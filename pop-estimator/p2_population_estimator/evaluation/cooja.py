"""Cooja-backed evaluator.

This module is intentionally a *thin adapter*: the heavy lifting (SSH,
queueing, retries) lives in :mod:`ssh_pool`, and the metric parsing lives in
:mod:`parser`. File generation (simulation.csc, positions.dat) mirrors the
logic from ``wsn-design-space-exploration/batch_runner``.

Customisation hooks are exposed as small methods that you can override for
your specific firmware/simulation glue:

  - ``make_cooja_file_generator`` — factory that returns the default generator
  - Pass a custom ``file_generator`` to :class:`CoojaEvaluator` to override

If ``paramiko`` is not installed, instantiating this evaluator raises a
helpful ``RuntimeError`` *only when ``evaluate`` is actually called*; the
class can be imported in tests without paramiko.
"""

from __future__ import annotations

import time
from pathlib import Path, PurePosixPath
from typing import Callable, Optional

import numpy as np

from p2_population_estimator.evaluation.base import (
    BaseEvaluator,
    make_evaluation_result,
)
from p2_population_estimator.evaluation.parser import parse_log_files
from p2_population_estimator.evaluation.ssh_pool import (
    SSHPool,
    SimulationTask,
    TaskResult,
    new_task_id,
)
from p2_population_estimator.logging_utils import get_logger
from p2_population_estimator.models import (
    EvaluationResult,
    FullSolution,
    MobileNode,
    P2Problem,
    ScalarizationWeights,
    SimulationMetrics,
)

log = get_logger(__name__)

# FileGenerator receives (solution, seed, local_workdir, problem, remote_workdir)
# and returns the list of local files to upload.
FileGenerator = Callable[[FullSolution, int, Path, P2Problem, PurePosixPath], list[Path]]

_MOTE_INTERFACES = [
    "org.contikios.cooja.interfaces.Position",
    "org.contikios.cooja.interfaces.Battery",
    "org.contikios.cooja.contikimote.interfaces.ContikiVib",
    "org.contikios.cooja.contikimote.interfaces.ContikiMoteID",
    "org.contikios.cooja.contikimote.interfaces.ContikiRS232",
    "org.contikios.cooja.contikimote.interfaces.ContikiBeeper",
    "org.contikios.cooja.interfaces.IPAddress",
    "org.contikios.cooja.contikimote.interfaces.ContikiRadio",
    "org.contikios.cooja.contikimote.interfaces.ContikiButton",
    "org.contikios.cooja.contikimote.interfaces.ContikiPIR",
    "org.contikios.cooja.contikimote.interfaces.ContikiClock",
    "org.contikios.cooja.contikimote.interfaces.ContikiLED",
    "org.contikios.cooja.contikimote.interfaces.ContikiCFS",
    "org.contikios.cooja.contikimote.interfaces.ContikiEEPROM",
    "org.contikios.cooja.interfaces.Mote2MoteRelations",
    "org.contikios.cooja.interfaces.MoteAttributes",
]


class CoojaEvaluator(BaseEvaluator):
    """Evaluate one solution at a time using a pool of remote Cooja containers."""

    name = "cooja"

    def __init__(
        self,
        problem: P2Problem,
        weights: ScalarizationWeights,
        *,
        pool: SSHPool,
        output_dir: Path,
        remote_workdir_root: PurePosixPath,
        command_template: str,
        aggregation_method: str = "mean_with_std",
        simulation_timeout: int = 900,
        simulation_duration_s: int = 180,
        remote_cooja_dir: str = "/opt/contiki-ng/tools/cooja",
        file_generator: Optional[FileGenerator] = None,
    ):
        self.problem = problem
        self.weights = weights
        self.pool = pool
        self.output_dir = Path(output_dir)
        self.remote_workdir_root = remote_workdir_root
        self.command_template = command_template
        self.aggregation_method = aggregation_method
        self.simulation_timeout = simulation_timeout
        self.simulation_duration_s = simulation_duration_s
        self.remote_cooja_dir = remote_cooja_dir
        self.file_generator = file_generator or make_cooja_file_generator(
            simulation_duration_s=simulation_duration_s,
            remote_cooja_dir=remote_cooja_dir,
        )

    # ------------------------------------------------------------------ #
    def evaluate(self, solution: FullSolution, seeds: list[int]) -> EvaluationResult:
        t0 = time.perf_counter()
        if not seeds:
            raise ValueError("Cooja evaluator requires at least one seed")

        sol_dir = self.output_dir / "evaluations" / solution.solution_id
        sol_dir.mkdir(parents=True, exist_ok=True)

        submitted: list[SimulationTask] = []
        seed_to_task_id: dict[int, str] = {}
        for seed in seeds:
            task_id = new_task_id(f"sol-{solution.solution_id}-seed-{seed}")
            seed_to_task_id[seed] = task_id
            local_wd = sol_dir / str(seed)
            local_wd.mkdir(parents=True, exist_ok=True)
            # Files go directly into remote_cooja_dir (mirrors batch_runner).
            # One worker per container port ensures no concurrent overwrites.
            remote_wd = PurePosixPath(self.remote_cooja_dir)
            task = SimulationTask(
                task_id=task_id,
                solution_id=solution.solution_id,
                seed=seed,
                local_workdir=local_wd,
                remote_workdir=remote_wd,
                prepare_local=lambda workdir, _sol=solution, _seed=seed, _rwd=remote_wd: (
                    self.file_generator(_sol, _seed, workdir, self.problem, _rwd)
                ),
                command_template=self.command_template,
                placeholders={
                    "seed": str(seed),
                    "remote_cooja_dir": self.remote_cooja_dir,
                },
                remote_log_path=PurePosixPath(self.remote_cooja_dir) / "COOJA.testlog",
                timeout_s=self.simulation_timeout,
            )
            submitted.append(task)
            self.pool.submit(task)

        target_ids = {t.task_id for t in submitted}
        results_for_self: dict[str, TaskResult] = {}
        while target_ids - set(results_for_self):
            time.sleep(0.2)
            with self.pool._lock:  # noqa: SLF001
                snapshot = list(self.pool.results)
            for r in snapshot:
                if r.task_id in target_ids and r.task_id not in results_for_self:
                    results_for_self[r.task_id] = r

        per_seed: list[SimulationMetrics] = []
        any_failed = False
        for seed in seeds:
            tid = seed_to_task_id[seed]
            res = results_for_self[tid]
            if res.status == "ok" and res.log_path is not None:
                try:
                    metrics = parse_log_files([str(res.log_path)])
                except Exception as exc:  # noqa: BLE001
                    log.error(
                        "Failed to parse Cooja log",
                        kv={"solution_id": solution.solution_id, "seed": seed, "error": str(exc)},
                    )
                    any_failed = True
                    metrics = SimulationMetrics()
            else:
                log.error(
                    "Cooja simulation task failed",
                    kv={
                        "solution_id": solution.solution_id,
                        "seed": seed,
                        "error": res.error,
                    },
                )
                any_failed = True
                metrics = SimulationMetrics()
            per_seed.append(metrics)

        eval_result = make_evaluation_result(
            solution=solution,
            per_seed=per_seed,
            weights=self.weights,
            num_candidates=len(self.problem.candidates),
            aggregation_method=self.aggregation_method,
            duration_s=time.perf_counter() - t0,
        )
        if any_failed:
            eval_result.status = "failed"
            eval_result.error = "At least one seed failed; metrics may be missing."
        return eval_result

    def shutdown(self) -> None:
        self.pool.shutdown()


# ---------------------------------------------------------------------------
# Real file generator (mirrors wsn-dse batch_runner logic)
# ---------------------------------------------------------------------------

def make_cooja_file_generator(
    *,
    simulation_duration_s: int = 180,
    remote_cooja_dir: str = "/opt/contiki-ng/tools/cooja",
) -> FileGenerator:
    """Return a FileGenerator that produces ``simulation.csc`` and ``positions.dat``.

    Mirrors the batch_runner pipeline exactly:
    * Files are uploaded to ``remote_cooja_dir`` (the Cooja installation dir).
    * Firmware (``root.c``, ``node.c``) is expected to be pre-installed there
      via ``update_firmware.py`` — NOT uploaded per simulation.
    * ``positions.dat`` is uploaded only when mobile nodes are present.
    """

    def _generate(
        solution: FullSolution,
        seed: int,
        workdir: Path,
        problem: P2Problem,
        _remote_wd: PurePosixPath,  # files go to remote_workdir (= remote_cooja_dir)
    ) -> list[Path]:
        files: list[Path] = []

        # Topology: sink (server, mote 1) + selected relays (clients, motes 2..R)
        selected_indices = [j for j, b in enumerate(solution.bits) if b]
        fixed_positions: list[tuple[float, float]] = [
            (problem.sink.x, problem.sink.y)
        ] + [(problem.candidates[j].x, problem.candidates[j].y) for j in selected_indices]

        # positions.dat — uploaded only when mobile nodes exist;
        # CSC references remote_cooja_dir/positions.dat (same dir as firmware).
        if problem.mobile_nodes:
            pos_path = workdir / "positions.dat"
            _write_positions_dat(pos_path, fixed_positions, problem.mobile_nodes)
            files.append(pos_path)

        # simulation.csc — always uploaded
        csc_path = workdir / "simulation.csc"
        _write_simulation_csc(
            path=csc_path,
            seed=seed,
            fixed_positions=fixed_positions,
            mobile_nodes=problem.mobile_nodes,
            tx_range=problem.radius_of_reach,
            interference_range=problem.radius_of_inter,
            simulation_duration_s=simulation_duration_s,
            remote_cooja_dir=remote_cooja_dir,
        )
        files.append(csc_path)

        return files

    return _generate


# ---------------------------------------------------------------------------
# positions.dat generation (ported from wsn-dse parse_json_pos_dat.py)
# ---------------------------------------------------------------------------

def _write_positions_dat(
    path: Path,
    fixed_positions: list[tuple[float, float]],
    mobile_nodes: list[MobileNode],
) -> None:
    """Write Cooja Mobility plugin positions file.

    Format (one entry per line):
        mote_index  time_seconds  x  y
    Fixed motes are listed once at t=0; mobile motes are listed once per
    time step, with position computed from their ``path_segments``.
    """
    with path.open("w", encoding="utf-8") as f:
        f.write("# Fixed positions\n")
        for i, (x, y) in enumerate(fixed_positions):
            f.write(f"{i} 0.00000000 {x:.2f} {y:.2f}\n")
        f.write("\n")

        if not mobile_nodes:
            return

        f.write("# Mobile nodes\n")
        mote_index = len(fixed_positions)
        max_steps = 0
        mobile_trajectories: list[tuple[int, "np.ndarray", "np.ndarray", float]] = []

        for mn in mobile_nodes:
            x_all: list["np.ndarray"] = []
            y_all: list["np.ndarray"] = []
            seg_dists: list[float] = []

            for x_expr, y_expr in mn.path_segments:
                t_vals = np.linspace(0, 1, 100)
                x_vals = np.array([eval(x_expr, {"t": float(t), "np": np}) for t in t_vals])  # noqa: S307
                y_vals = np.array([eval(y_expr, {"t": float(t), "np": np}) for t in t_vals])  # noqa: S307
                x_all.append(x_vals)
                y_all.append(y_vals)
                seg_dists.append(float(np.sum(np.sqrt(np.diff(x_vals) ** 2 + np.diff(y_vals) ** 2))))

            total_dist = sum(seg_dists)
            total_duration = total_dist / mn.speed if mn.speed > 0 else total_dist
            total_steps = max(1, int(total_duration / mn.time_step))
            max_steps = max(max_steps, total_steps)

            x_full: list[float] = []
            y_full: list[float] = []
            for x_vals, y_vals, seg_dist in zip(x_all, y_all, seg_dists):
                proportion = (seg_dist / total_dist) if total_dist > 0 else (1.0 / len(x_all))
                seg_steps = max(1, int(proportion * total_steps))
                interp_t = np.linspace(0, 1, seg_steps)
                x_full.extend(np.interp(interp_t, np.linspace(0, 1, len(x_vals)), x_vals))
                y_full.extend(np.interp(interp_t, np.linspace(0, 1, len(y_vals)), y_vals))

            xa = np.array(x_full)
            ya = np.array(y_full)

            if mn.is_round_trip:  # closed paths naturally loop; only reverse for round-trips
                xa = np.concatenate([xa, xa[::-1]])
                ya = np.concatenate([ya, ya[::-1]])

            mobile_trajectories.append((mote_index, xa, ya, mn.time_step))
            mote_index += 1

        for step in range(2 * max_steps):
            for mote_id, xa, ya, time_step in mobile_trajectories:
                if step < len(xa):
                    f.write(f"{mote_id} {step * time_step:.8f} {xa[step]:.2f} {ya[step]:.2f}\n")
            f.write("\n")


# ---------------------------------------------------------------------------
# simulation.csc generation (ported from wsn-dse replace_xml.py)
# ---------------------------------------------------------------------------

def _write_simulation_csc(
    *,
    path: Path,
    seed: int,
    fixed_positions: list[tuple[float, float]],
    mobile_nodes: list[MobileNode],
    tx_range: float,
    interference_range: float,
    simulation_duration_s: int,
    remote_cooja_dir: str,
) -> None:
    """Write a complete Cooja simulation file (.csc), mirroring simulation_template.xml.

    Topology:
      - Mote 1: sink  (server firmware ``root.c`` pre-installed in remote_cooja_dir)
      - Motes 2..R+1: selected relay candidates (``node.c``)
      - Motes R+2..R+M+1: mobile nodes at their trajectory start positions

    Timeout note (matches batch_runner replace_xml.py):
      ``time`` in Cooja JS is in **milliseconds**; ``TIMEOUT`` also in ms.
      ``timeOut = simulation_duration_s * 1000`` ms drives the while loop.
      ``TIMEOUT(timeOut + 11000)`` gives 11 s of tolerance for the last YIELD.
    """
    timeout_ms = simulation_duration_s * 1000       # ms of simulated time
    timeout_tol_ms = timeout_ms + 11000             # hard cap (ms), 11 s buffer

    script = (
        f"        log.log(\"Initializing simulation script\\n\");\n"
        f"        var initTime = time;\n"
        f"        const timeOut = {timeout_ms};\n"
        f"        TIMEOUT({timeout_tol_ms});\n"
        f"        sim.startSimulation();\n"
        f"        while (time < initTime + timeOut) {{\n"
        f"          YIELD();\n"
        f"          log.log(\"[Mote:\" + id + \"] \" + msg + \"\\n\");\n"
        f"        }}\n"
        f"        sim.stopSimulation();\n"
        f"        log.log(\"Final simulation time: \" + sim.getSimulationTime() + \" ms\\n\");\n"
    )

    interfaces_xml = "\n".join(
        f"      <moteinterface>{iface}</moteinterface>" for iface in _MOTE_INTERFACES
    )

    # Compute mobile start positions
    mobile_starts: list[tuple[float, float]] = []
    for mn in mobile_nodes:
        x0, y0 = _eval_segment_start(mn.path_segments[0])
        mobile_starts.append((x0, y0))

    # Server mote = sink (mote 1)
    server_mote_xml = _mote_xml(1, fixed_positions[0][0], fixed_positions[0][1])

    # Client motes: relays (motes 2..R+1) + mobile starts (motes R+2..)
    client_positions = fixed_positions[1:] + mobile_starts
    client_motes_xml = "\n".join(
        _mote_xml(i + 2, x, y) for i, (x, y) in enumerate(client_positions)
    )

    # Mobility plugin — positions.dat is always in remote_cooja_dir (same as firmware)
    mobility_plugin_xml = ""
    if mobile_nodes:
        positions_path = f"{remote_cooja_dir}/positions.dat"
        mobility_plugin_xml = (
            f"  <plugin>\n"
            f"    org.contikios.cooja.plugins.Mobility\n"
            f"    <plugin_config>\n"
            f"      <positions>{positions_path}</positions>\n"
            f"    </plugin_config>\n"
            f"    <bounds x=\"7\" y=\"517\" height=\"200\" width=\"500\" />\n"
            f"  </plugin>\n"
        )

    content = (
        f'<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<simconf version="2023090101">\n'
        f'  <simulation>\n'
        f'    <title>P2 pop-est seed={seed}</title>\n'
        f'    <randomseed>{seed}</randomseed>\n'
        f'    <motedelay_us>1000000</motedelay_us>\n'
        f'    <radiomedium>\n'
        f'      org.contikios.cooja.radiomediums.UDGM\n'
        f'      <transmitting_range>{tx_range}</transmitting_range>\n'
        f'      <interference_range>{interference_range}</interference_range>\n'
        f'      <success_ratio_tx>1.0</success_ratio_tx>\n'
        f'      <success_ratio_rx>1.0</success_ratio_rx>\n'
        f'    </radiomedium>\n'
        f'    <events>\n'
        f'      <logoutput>40000</logoutput>\n'
        f'    </events>\n'
        f'    <motetype>\n'
        f'      org.contikios.cooja.contikimote.ContikiMoteType\n'
        f'      <description>server</description>\n'
        f'      <source>{remote_cooja_dir}/root.c</source>\n'
        f'      <commands>$(MAKE) -j$(CPUS) root.cooja TARGET=cooja</commands>\n'
        f'{interfaces_xml}\n'
        f'{server_mote_xml}\n'
        f'    </motetype>\n'
        f'    <motetype>\n'
        f'      org.contikios.cooja.contikimote.ContikiMoteType\n'
        f'      <description>client</description>\n'
        f'      <source>{remote_cooja_dir}/node.c</source>\n'
        f'      <commands>$(MAKE) -j$(CPUS) node.cooja TARGET=cooja</commands>\n'
        f'{interfaces_xml}\n'
        f'{client_motes_xml}\n'
        f'    </motetype>\n'
        f'  </simulation>\n'
        f'  <plugin>\n'
        f'    org.contikios.cooja.plugins.ScriptRunner\n'
        f'    <plugin_config>\n'
        f'      <script><![CDATA[\n'
        f'{script}'
        f'      ]]></script>\n'
        f'    </plugin_config>\n'
        f'  </plugin>\n'
        f'  <plugin>\n'
        f'    org.contikios.cooja.plugins.LogListener\n'
        f'    <plugin_config>\n'
        f'      <filter />\n'
        f'      <formatted_time />\n'
        f'      <coloring />\n'
        f'      <append>{remote_cooja_dir}/loglistener_append.txt</append>\n'
        f'    </plugin_config>\n'
        f'  </plugin>\n'
        f'{mobility_plugin_xml}'
        f'</simconf>\n'
    )

    path.write_text(content, encoding="utf-8")


def _mote_xml(mote_id: int, x: float, y: float) -> str:
    return (
        f"      <mote>\n"
        f"        <interface_config>\n"
        f"          org.contikios.cooja.interfaces.Position\n"
        f"          <pos x=\"{x}\" y=\"{y}\" />\n"
        f"        </interface_config>\n"
        f"        <interface_config>\n"
        f"          org.contikios.cooja.contikimote.interfaces.ContikiMoteID\n"
        f"          <id>{mote_id}</id>\n"
        f"        </interface_config>\n"
        f"      </mote>"
    )


def _eval_segment_start(segment: tuple[str, str]) -> tuple[float, float]:
    x_expr, y_expr = segment
    x0 = float(eval(x_expr, {"t": 0.0, "np": np}))  # noqa: S307
    y0 = float(eval(y_expr, {"t": 0.0, "np": np}))  # noqa: S307
    return x0, y0
