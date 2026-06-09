"""SSH worker pool for distributing simulation tasks across Cooja containers.

Design notes:

* We use ``paramiko`` (optional dependency). Importing this module *without*
  paramiko installed must NOT break the surrogate path — we therefore import
  paramiko lazily, inside the methods that need it.

* Each container is exposed on a distinct SSH port (e.g. 2231..2236). We
  associate one :class:`SSHWorker` per port and dispatch tasks via a
  :class:`queue.Queue`. At most one simulation is active per container.

* Connection model mirrors ``wsn-design-space-exploration/batch_runner``:
  a fresh SSH connection is created for each task and always closed in
  a ``finally`` block, regardless of success or failure. When a password
  is provided, public-key and agent auth are disabled so the connection
  goes straight to password auth (no spurious "Authentication failed" noise
  in the logs).

* Failures are retried up to ``max_retries`` times before the task is marked
  FAILED. Other tasks keep going.
"""

from __future__ import annotations

import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Callable, Optional

from p2_population_estimator.logging_utils import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class SimulationTask:
    """One unit of work for an SSH worker."""

    task_id: str
    solution_id: str
    seed: int
    local_workdir: Path
    remote_workdir: PurePosixPath
    # The worker calls ``prepare_local`` to produce files that should be uploaded
    # before running the command. Returns the list of local file paths.
    prepare_local: Callable[[Path], list[Path]]
    # The remote command template. May reference ``{simulation_file}``,
    # ``{workdir}``, and any extra keys in ``placeholders``.
    command_template: str
    placeholders: dict[str, str] = field(default_factory=dict)
    # Absolute remote path to the log file to download after the run.
    # If None, falls back to remote_workdir / remote_log_filename.
    remote_log_path: Optional[PurePosixPath] = None
    # Filename of the Cooja output log on the remote host (used when remote_log_path is None).
    remote_log_filename: str = "COOJA.testlog"
    # Local filename where the downloaded remote log is saved.
    log_filename: str = "cooja.log"
    timeout_s: int = 900


@dataclass(slots=True)
class TaskResult:
    task_id: str
    worker_port: int
    status: str  # "ok" | "failed"
    duration_s: float
    log_path: Optional[Path]
    error: Optional[str] = None
    attempts: int = 0


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
class SSHWorker(threading.Thread):
    """One worker bound to a single SSH port. Pulls tasks from a queue.

    SSH connection model (mirrors wsn-dse batch_runner.create_ssh):
    - A brand-new ``SSHClient`` is created at the start of each task.
    - When a password is supplied, ``allow_agent`` and ``look_for_keys`` are
      disabled so Paramiko goes straight to password auth without generating
      spurious public-key failure log lines.
    - The connection is always closed in a ``finally`` block after the task.
    """

    def __init__(
        self,
        *,
        host: str,
        port: int,
        user: str,
        in_queue: "queue.Queue[Optional[SimulationTask]]",
        out_list: list[TaskResult],
        out_lock: threading.Lock,
        max_retries: int = 2,
        connect_timeout_s: float = 20.0,
        key_filename: Optional[str] = None,
        password: Optional[str] = None,
    ):
        super().__init__(daemon=True, name=f"worker-{port}")
        self.host = host
        self.port = port
        self.user = user
        self.in_queue = in_queue
        self.out_list = out_list
        self.out_lock = out_lock
        self.max_retries = max_retries
        self.connect_timeout_s = connect_timeout_s
        self.key_filename = key_filename
        self.password = password

    # ------------------------------------------------------------------ #
    def run(self) -> None:
        while True:
            task = self.in_queue.get()
            try:
                if task is None:
                    return
                result = self._handle_task(task)
                with self.out_lock:
                    self.out_list.append(result)
            finally:
                self.in_queue.task_done()

    # ------------------------------------------------------------------ #
    def _connect(self) -> object:
        """Open a fresh SSH connection (mirrors batch_runner.create_ssh)."""
        import paramiko  # lazy import

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        connect_kwargs: dict = dict(
            hostname=self.host,
            port=self.port,
            username=self.user,
            timeout=self.connect_timeout_s,
            banner_timeout=self.connect_timeout_s,
            auth_timeout=self.connect_timeout_s,
        )
        if self.password:
            # Password-only auth: skip agent and key-file attempts entirely.
            connect_kwargs["password"] = self.password
            connect_kwargs["allow_agent"] = False
            connect_kwargs["look_for_keys"] = False
        if self.key_filename:
            connect_kwargs["key_filename"] = self.key_filename

        client.connect(**connect_kwargs)
        log.info("SSH connected", kv={"port": self.port, "host": self.host})
        return client

    # ------------------------------------------------------------------ #
    def _handle_task(self, task: SimulationTask) -> TaskResult:
        t0 = time.perf_counter()
        attempts = 0
        last_err: Optional[str] = None
        while attempts <= self.max_retries:
            attempts += 1
            try:
                log_path = self._run_once(task)
                return TaskResult(
                    task_id=task.task_id,
                    worker_port=self.port,
                    status="ok",
                    duration_s=time.perf_counter() - t0,
                    log_path=log_path,
                    attempts=attempts,
                )
            except Exception as exc:  # noqa: BLE001
                last_err = f"{type(exc).__name__}: {exc}"
                log.warning(
                    "Task attempt failed",
                    kv={
                        "task_id": task.task_id,
                        "port": self.port,
                        "attempt": attempts,
                        "error": last_err,
                    },
                )
                time.sleep(min(2 ** attempts, 10))
        return TaskResult(
            task_id=task.task_id,
            worker_port=self.port,
            status="failed",
            duration_s=time.perf_counter() - t0,
            log_path=None,
            error=last_err,
            attempts=attempts,
        )

    # ------------------------------------------------------------------ #
    def _run_once(self, task: SimulationTask) -> Path:
        """Execute one simulation task, mirroring batch_runner.run_simulation."""
        # 1) Build local files
        task.local_workdir.mkdir(parents=True, exist_ok=True)
        local_files = task.prepare_local(task.local_workdir)

        # 2) Fresh SSH connection per task (matches batch_runner pattern)
        client = self._connect()
        try:
            # 3) Upload simulation files via SFTP
            sftp = client.open_sftp()
            try:
                self._mkdir_p(sftp, task.remote_workdir)
                for lf in local_files:
                    remote_path = str(PurePosixPath(task.remote_workdir) / lf.name)
                    sftp.put(str(lf), remote_path)
            finally:
                try:
                    sftp.close()
                except Exception:  # noqa: BLE001
                    pass

            # 4) Resolve command placeholders
            placeholders = dict(task.placeholders)
            # simulation.csc is uploaded to remote_workdir; use relative name so the
            # command template's own "cd {remote_cooja_dir}" sets the right CWD.
            placeholders.setdefault("simulation_file", "simulation.csc")
            placeholders.setdefault("workdir", str(task.remote_workdir))
            command = task.command_template.format(**placeholders)
            full_cmd = command  # template already contains the cd to the right directory
            log.info(
                "Running remote command",
                kv={"port": self.port, "task_id": task.task_id, "cmd": full_cmd},
            )

            # 5) Execute with PTY + poll exit status (matches batch_runner)
            _, stdout, _ = client.exec_command(full_cmd, get_pty=True)
            deadline = time.monotonic() + task.timeout_s
            while not stdout.channel.exit_status_ready():
                if time.monotonic() > deadline:
                    stdout.channel.close()
                    raise RuntimeError(
                        f"remote command timed out after {task.timeout_s}s"
                    )
                time.sleep(0.2)
            rc = stdout.channel.recv_exit_status()

            # 6) Download COOJA.testlog (matches batch_runner scp_get)
            remote_log = str(
                task.remote_log_path
                if task.remote_log_path is not None
                else PurePosixPath(task.remote_workdir) / task.remote_log_filename
            )
            log_path = task.local_workdir / task.log_filename
            sftp2 = client.open_sftp()
            try:
                sftp2.get(remote_log, str(log_path))
            except IOError as exc:
                raise RuntimeError(
                    f"Failed to download remote log '{remote_log}': {exc}"
                ) from exc
            finally:
                try:
                    sftp2.close()
                except Exception:  # noqa: BLE001
                    pass

            if rc != 0:
                raise RuntimeError(f"remote command exited with code {rc}")
            return log_path

        finally:
            # Always close connection after each task (matches batch_runner)
            try:
                client.close()
            except Exception:  # noqa: BLE001
                pass

    @staticmethod
    def _mkdir_p(sftp: object, path: PurePosixPath) -> None:
        parts = [p for p in str(path).split("/") if p]
        cur = "/"
        for part in parts:
            cur = cur.rstrip("/") + "/" + part
            try:
                sftp.stat(cur)  # type: ignore[attr-defined]
            except IOError:
                sftp.mkdir(cur)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Pool
# ---------------------------------------------------------------------------
class SSHPool:
    """A pool of one worker per SSH port."""

    def __init__(
        self,
        *,
        host: str,
        user: str,
        ports: list[int],
        max_retries: int = 2,
        key_filename: Optional[str] = None,
        password: Optional[str] = None,
    ):
        if not ports:
            raise ValueError("SSHPool needs at least one port")
        self.host = host
        self.user = user
        self.ports = list(ports)
        self.queue: "queue.Queue[Optional[SimulationTask]]" = queue.Queue()
        self.results: list[TaskResult] = []
        self._lock = threading.Lock()
        self.workers: list[SSHWorker] = []
        for p in ports:
            w = SSHWorker(
                host=host,
                port=p,
                user=user,
                in_queue=self.queue,
                out_list=self.results,
                out_lock=self._lock,
                max_retries=max_retries,
                key_filename=key_filename,
                password=password,
            )
            self.workers.append(w)

    def start(self) -> None:
        for w in self.workers:
            w.start()

    def submit(self, task: SimulationTask) -> None:
        self.queue.put(task)

    def submit_many(self, tasks: list[SimulationTask]) -> None:
        for t in tasks:
            self.queue.put(t)

    def join(self) -> list[TaskResult]:
        """Wait for all queued tasks to finish, then return collected results."""
        self.queue.join()
        return list(self.results)

    def shutdown(self) -> None:
        for _ in self.workers:
            self.queue.put(None)
        for w in self.workers:
            w.join(timeout=5.0)


def new_task_id(prefix: str = "task") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"
