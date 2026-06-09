"""Data models used across the package.

We use ``@dataclass`` (plain stdlib) rather than Pydantic to keep the runtime
dependency footprint minimal. Validation is performed in ``io.py`` explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional


# ---------------------------------------------------------------------------
# Geometry primitives
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Point:
    x: float
    y: float

    def as_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)


@dataclass(frozen=True, slots=True)
class Region:
    """Axis-aligned bounding box [xmin, ymin, xmax, ymax]."""

    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def contains(self, p: Point) -> bool:
        return self.xmin <= p.x <= self.xmax and self.ymin <= p.y <= self.ymax

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin


# ---------------------------------------------------------------------------
# Mobile nodes (trajectories)
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class MobileNode:
    """A mobile sensor whose position is given by piecewise parametric segments.

    ``path_segments`` is a list of [expr_x(t), expr_y(t)] string pairs, where
    ``t`` runs in ``[0, 1]`` over the segment. ``speed`` and ``time_step``
    keep the original semantics from the SimLab specification.
    """

    name: str
    speed: float
    time_step: float
    is_closed: bool
    is_round_trip: bool
    path_segments: list[tuple[str, str]]
    source_code: Optional[str] = None
    raw: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Problem / Instance
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class P2Problem:
    name: str
    radius_of_reach: float
    radius_of_inter: float
    region: Region
    sink: Point
    candidates: list[Point]
    mobile_nodes: list[MobileNode]
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class P2Instance:
    """Top-level wrapper around the parsed JSON. ``raw`` preserves the original
    JSON so downstream Cooja artefact generators can read additional fields
    (e.g. simulation duration, repository options) without re-parsing."""

    problem: P2Problem
    raw: dict[str, Any] = field(default_factory=dict)
    source_path: Optional[str] = None


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class CandidateBlock:
    """A structural block of candidate indices (Q_i)."""

    block_id: int
    indices: list[int]  # original indices into Q
    method: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def k(self) -> int:
        """Block size k_i = |Q_i|."""
        return len(self.indices)


@dataclass(slots=True)
class BlockPattern:
    """A configuration for a block, i.e. the bits H_i restricted to Q_i.

    ``bits`` has length ``k_i`` and is aligned with ``CandidateBlock.indices``.
    """

    block_id: int
    bits: list[int]  # length k_i, values in {0,1}
    label: str  # e.g. "H_star", "H_local"

    @property
    def s(self) -> int:
        """Number of active bits s_i."""
        return sum(self.bits)

    @property
    def k(self) -> int:
        return len(self.bits)


@dataclass(slots=True)
class FullSolution:
    """A complete binary vector x in {0,1}^J."""

    solution_id: str
    bits: list[int]  # length J
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def relay_count(self) -> int:
        return sum(self.bits)


# ---------------------------------------------------------------------------
# Simulation results
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class SimulationSeed:
    seed: int


@dataclass(slots=True)
class SimulationMetrics:
    """One simulation's metrics. Unknown metrics default to ``None`` so the
    surrogate and Cooja back-ends can fill different subsets."""

    latency: Optional[float] = None
    energy: Optional[float] = None
    throughput: Optional[float] = None
    packet_delivery_ratio: Optional[float] = None
    connected_ratio: Optional[float] = None
    relay_count: Optional[int] = None
    # Surrogate-only convenience fields
    mean_hop_count: Optional[float] = None
    mean_distance_to_mobile: Optional[float] = None
    redundancy: Optional[float] = None
    # Free-form extras (raw parsed values)
    extras: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "latency": self.latency,
            "energy": self.energy,
            "throughput": self.throughput,
            "packet_delivery_ratio": self.packet_delivery_ratio,
            "connected_ratio": self.connected_ratio,
            "relay_count": self.relay_count,
            "mean_hop_count": self.mean_hop_count,
            "mean_distance_to_mobile": self.mean_distance_to_mobile,
            "redundancy": self.redundancy,
        }
        if self.extras:
            d["extras"] = dict(self.extras)
        return d


@dataclass(slots=True)
class AggregatedMetrics:
    """Aggregation of per-seed metrics produced by Psi_a."""

    method: str
    mean: SimulationMetrics
    std: SimulationMetrics
    n: int
    se: SimulationMetrics


@dataclass(slots=True)
class EvaluationResult:
    """Result of evaluating one FullSolution against one or more seeds."""

    solution_id: str
    per_seed: list[SimulationMetrics]
    aggregated: AggregatedMetrics
    F: float  # scalarised objective F(x)
    duration_s: float
    status: Literal["ok", "failed"] = "ok"
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Statistical / final results
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class BlockComparisonResult:
    """Per-block result of the d_i_hat / sigma_BB_i_hat procedure."""

    block_id: int
    k_i: int
    s_i_star: int
    alpha: float
    pi_i_star: Optional[float]
    d_i_hat: float
    sigma_BB_i_hat: float
    delta_samples: list[float]
    F_star_samples: list[float]
    F_local_samples: list[float]
    n_i_uniform: Optional[float]
    n_i_uniform_ceil: Optional[int]
    n_i_bernoulli: Optional[float]
    n_i_bernoulli_ceil: Optional[int]
    status: str  # "ok" | "invalid_non_positive_d" | "degenerate_zero_variance" | ...
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PopulationEstimateResult:
    """Final result bundle persisted to JSON."""

    experiment_config: dict[str, Any]
    instance_summary: dict[str, Any]
    partition_summary: dict[str, Any]
    block_results: list[BlockComparisonResult]
    global_estimate: dict[str, Any]
    failed_evaluations: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    reproducibility_info: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Experiment configuration (consumed by experiment.run)
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class ScalarizationWeights:
    """Weights for the F(x) scalarisation. All defaults are documented in
    ``config.py``. ``required_metrics`` lists metric names that MUST be present
    for the chosen scalarisation; failure to satisfy this list raises a
    ``ValueError`` at evaluation time."""

    w_connected: float = 1.0
    w_relays: float = 0.05
    w_hops: float = 0.05
    w_dist: float = 0.05
    w_redundancy: float = 0.02
    # Cooja-mode metrics (used only when present)
    w_latency: float = 0.0
    w_energy: float = 0.0
    w_throughput: float = 0.0
    required_metrics: tuple[str, ...] = ("connected_ratio", "relay_count")


@dataclass(slots=True)
class ExperimentConfig:
    instance_path: str
    output_dir: str
    mode: Literal["surrogate", "cooja"]
    partition_method: Literal["grid", "kmeans", "radial_to_sink"]
    num_blocks: int
    num_complements: int
    alpha: float
    rho: float
    seeds: list[int]
    random_seed: int
    hstar_method: str = "structural_greedy"
    hlocal_method: str = "deceptive_low_cost"
    complement_method: str = "bernoulli"
    aggregation_method: Literal["mean", "median", "trimmed_mean", "mean_with_std"] = "mean_with_std"
    weights: ScalarizationWeights = field(default_factory=ScalarizationWeights)
    # Cooja-specific
    ssh_host: str = "localhost"
    ssh_user: str = ""
    ssh_password: Optional[str] = None
    ssh_ports: list[int] = field(default_factory=lambda: [2231, 2232, 2233, 2234, 2235, 2236])
    remote_workdir: str = "/tmp/popest"
    simulation_timeout: int = 900
    simulation_duration: int = 180
    remote_cooja_dir: str = "/opt/contiki-ng/tools/cooja"
    cooja_command_template: str = (
        "cd {remote_cooja_dir} && "
        "/opt/java/openjdk/bin/java --enable-preview "
        "-Xms4g -Xmx4g "
        "-jar build/libs/cooja.jar --no-gui {simulation_file}"
    )
    max_retries: int = 2
    # External H_star / H_local files (optional)
    hstar_external_path: Optional[str] = None
    hlocal_external_path: Optional[str] = None
    # Firmware directory to upload alongside simulation files (cooja mode)
    firmware_local_dir: Optional[str] = None
    # Overwrite protection
    force_overwrite: bool = False
