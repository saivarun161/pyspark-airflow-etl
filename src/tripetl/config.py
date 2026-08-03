"""Pipeline configuration: where data lives and how strict the gates are."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_WAREHOUSE = "warehouse"


@dataclass(frozen=True)
class PipelineConfig:
    """Everything a stage needs to know that is not the data itself.

    Paths are derived from a single ``warehouse`` root rather than configured
    one by one, so a run can be redirected somewhere else (a temp directory in
    a test, a dated prefix in production) by changing one value.
    """

    warehouse: str = DEFAULT_WAREHOUSE
    input_path: str | None = None
    input_format: str = "parquet"

    app_name: str = "tripetl"
    master: str = "local[*]"
    shuffle_partitions: int = 8

    #: When false, a failing gate is reported but does not stop the run. Useful
    #: for a first look at an unfamiliar extract, where you want the whole
    #: quality picture rather than the first thing that trips.
    enforce_gates: bool = True

    #: Rows failing a quarantine-eligible rule are written here instead of
    #: being dropped. A row that vanishes without trace is a bug you find
    #: three months later in a revenue number.
    quarantine_enabled: bool = True

    write_mode: str = "overwrite"

    #: Sample-generator settings, used when no ``input_path`` is given.
    sample_rows: int = 20_000
    sample_seed: int = 20260803
    sample_dirty_rate: float = 0.04

    extra_conf: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.sample_dirty_rate <= 1.0:
            raise ValueError(
                f"sample_dirty_rate must be between 0 and 1, got {self.sample_dirty_rate}"
            )
        if self.sample_rows < 0:
            raise ValueError(f"sample_rows must be non-negative, got {self.sample_rows}")

    # -- derived paths ----------------------------------------------------

    @property
    def bronze_path(self) -> str:
        return self._under("bronze", "trips")

    @property
    def silver_path(self) -> str:
        return self._under("silver", "trips")

    @property
    def gold_path(self) -> str:
        return self._under("gold", "daily_zone_metrics")

    @property
    def reports_dir(self) -> str:
        return self._under("_quality")

    def quarantine_path(self, stage: str) -> str:
        return self._under("_quarantine", stage)

    def report_path(self, stage: str) -> str:
        return str(Path(self.reports_dir) / f"{stage}.json")

    def _under(self, *parts: str) -> str:
        return str(Path(self.warehouse).joinpath(*parts))

    def session_kwargs(self) -> dict[str, object]:
        return {
            "app_name": self.app_name,
            "master": self.master,
            "shuffle_partitions": self.shuffle_partitions,
            "extra_conf": self.extra_conf,
        }
