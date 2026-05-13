"""SSH worker pool for distributing simulation tasks across Cooja containers.

Design notes:

* We use ``paramiko`` (optional dependency). Importing this module *without*
  paramiko installed must NOT break the surrogate path — we therefore import
  paramiko lazily, inside the methods that need it.

* Each container is exposed on a distinct SSH port (e.g. 2231..2236). We
  associate one :class:`SSHWorker` per port and dispatch tasks via a
  :class:`queue.Queue`. At most one simulation is active per container.

* Failures are retried up to ``max_retries`` times before the task is marked
  FAILED. Other tasks keep going.

* We make sure connections are always closed in ``shutdown``.
"""

from __future__ import annotations

import os
import queue
import stat
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Optional

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
    # The remote command (after files are uploaded). May reference
    # ``{simulation_file}`` and other placeholders. The command is formatted
    # with ``placeholders`` below.
    command_template: str
    placeholders: dict[str, str] = field(default_factory=dict)
    # Where to write the combined remote stdout/stderr locally.
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
    """One worker bound to a single SSH port. Pulls tasks from a queue."""

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
        self._client: Any = None  # paramiko.SSHClient when connected

    # ------------------------------------------------------------------ #
    def run(self) -> None:
        try:
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
        finally:
            self._close()

    # ------------------------------------------------------------------ #
    def _ensure_connected(self) -> Any:
        if self._client is not None:
            transport = self._client.get_transport()
            if transport is not None and transport.is_active():
                return self._client
        import paramiko  # lazy import

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(
            hostname=self.host,
            port=self.port,
            username=self.user,
            password=self.password,
            key_filename=self.key_filename,
            timeout=self.connect_timeout_s,
            banner_timeout=self.connect_timeout_s,
            auth_timeout=self.connect_timeout_s,
            allow_agent=True,
            look_for_keys=True,
        )
        self._client = client
        log.info("SSH worker connected", kv={"port": self.port, "host": self.host})
        return client

    def _close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:  # noqa: BLE001
                pass
            self._client = None

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
                # Drop the SSH client; force reconnect on next try.
                self._close()
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
        client = self._ensure_connected()
        # 1) Build local files
        task.local_workdir.mkdir(parents=True, exist_ok=True)
        local_files = task.prepare_local(task.local_workdir)
        # 2) Ensure remote dir
        sftp = client.open_sftp()
        try:
            self._mkdir_p(sftp, task.remote_workdir)
            # 3) Upload
            for lf in local_files:
                remote_path = PurePosixPath(task.remote_workdir) / lf.name
                sftp.put(str(lf), str(remote_path))
            # 4) Resolve placeholders
            placeholders = dict(task.placeholders)
            placeholders.setdefault(
                "simulation_file",
                str(PurePosixPath(task.remote_workdir) / "simulation.csc"),
            )
            placeholders.setdefault("workdir", str(task.remote_workdir))
            command = task.command_template.format(**placeholders)
            full_cmd = f"cd {task.remote_workdir} && {command}"
            log.info(
                "Running remote command",
                kv={"port": self.port, "task_id": task.task_id, "cmd": full_cmd},
            )
            # 5) Exec
            stdin, stdout, stderr = client.exec_command(full_cmd, timeout=task.timeout_s)
            # Drain streams
            out = stdout.read().decode("utf-8", errors="replace")
            err = stderr.read().decode("utf-8", errors="replace")
            rc = stdout.channel.recv_exit_status()
            # 6) Save log locally
            log_path = task.local_workdir / task.log_filename
            with log_path.open("w", encoding="utf-8") as fh:
                fh.write(f"# remote command: {full_cmd}\n# rc={rc}\n\n# STDOUT:\n")
                fh.write(out)
                fh.write("\n\n# STDERR:\n")
                fh.write(err)
            if rc != 0:
                raise RuntimeError(f"remote command exited with code {rc}")
            return log_path
        finally:
            try:
                sftp.close()
            except Exception:  # noqa: BLE001
                pass

    @staticmethod
    def _mkdir_p(sftp: Any, path: PurePosixPath) -> None:
        parts = [p for p in str(path).split("/") if p]
        cur = "/"
        for part in parts:
            cur = (cur.rstrip("/") + "/" + part)
            try:
                sftp.stat(cur)
            except IOError:
                sftp.mkdir(cur)


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
        # Sentinels
        for _ in self.workers:
            self.queue.put(None)
        for w in self.workers:
            w.join(timeout=5.0)


def new_task_id(prefix: str = "task") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"
