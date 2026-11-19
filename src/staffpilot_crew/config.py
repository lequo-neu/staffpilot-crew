"""Crew settings: the graph limits, the router's models and the file locations.

The packaged defaults live in ``default_crew.toml``. ``config/crew.toml`` in the working
folder overrides them section by section, and the file named in ``STAFFPILOT_CREW_CONFIG``
wins over both. Model ids are only ever read from here, never written in the code.
"""

from __future__ import annotations

from importlib import resources
import os
from pathlib import Path
import tomllib
from typing import Any

from pydantic import BaseModel, Field


CONFIG_ENV_VAR = "STAFFPILOT_CREW_CONFIG"
STEP_TYPES = ("classify", "reason", "decide", "write")
DIFFICULTIES = ("simple", "moderate", "hard")


class GraphSettings(BaseModel):
    optimizer_retries: int = Field(2, ge=0, le=5)
    writer_retries: int = Field(2, ge=0, le=5)
    recursion_limit: int = Field(60, ge=10)
    monitor_gap: float = Field(0.10, ge=0.0, le=1.0)


class ModelEntry(BaseModel):
    name: str
    label: str
    provider: str
    spec: str
    prices: tuple[float, float] | None = None
    roles: list[str] = Field(default_factory=list)
    prior: dict[str, float] = Field(default_factory=dict)
    measured: dict[str, float] = Field(default_factory=dict)
    measured_source: str = ""

    def accuracy(self, step: str) -> tuple[float, str]:
        """The accuracy the router uses for a step, and where it comes from."""
        if step in self.measured:
            return self.measured[step], "measured"
        return self.prior.get(step, 0.0), "prior"

    @property
    def cost_rank(self) -> float:
        """Order from cheapest to dearest. A model with no known price sorts after priced ones."""
        if self.prices is None:
            return float("inf")
        return self.prices[0] + self.prices[1]

    def cost(self, tokens_in: int, tokens_out: int) -> float | None:
        if self.prices is None:
            return None
        return (tokens_in * self.prices[0] + tokens_out * self.prices[1]) / 1_000_000


class RouterSettings(BaseModel):
    bar: dict[str, float] = Field(default_factory=lambda: {"simple": 0.80, "moderate": 0.88, "hard": 0.95})
    hard_countries: int = Field(3, ge=2)
    models: list[ModelEntry] = Field(default_factory=list)

    def model(self, name: str) -> ModelEntry:
        for entry in self.models:
            if entry.name == name:
                return entry
        raise KeyError(f"No model named {name!r} in the crew config")


class AutomationSettings(BaseModel):
    uplift_share_low: float = Field(0.0, ge=0.0, le=1.0)
    uplift_share_high: float = Field(0.5, ge=0.0, le=1.0)


class WorkflowSettings(BaseModel):
    finance_cost_share: float = Field(0.05, ge=0.0)
    finance_cost_usd: float = Field(100_000, ge=0.0)
    finance_hires_per_country: int = Field(5, ge=0)
    stale_days: int = Field(7, ge=1)
    due_soon_days: int = Field(7, ge=0)


class PathSettings(BaseModel):
    app_db: str = "outputs/crew_app.sqlite"
    checkpoints: str = "outputs/crew_checkpoints.sqlite"
    eval_results: str = "eval/results"


class CrewSettings(BaseModel):
    graph: GraphSettings = Field(default_factory=GraphSettings)
    router: RouterSettings = Field(default_factory=RouterSettings)
    automation: AutomationSettings = Field(default_factory=AutomationSettings)
    workflow: WorkflowSettings = Field(default_factory=WorkflowSettings)
    paths: PathSettings = Field(default_factory=PathSettings)

    def with_updates(self, **sections: dict[str, Any]) -> CrewSettings:
        data = self.model_dump()
        for name, values in sections.items():
            data[name] = {**data[name], **values}
        return CrewSettings.model_validate(data)


def _read(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_crew_settings(path: str | Path | None = None) -> CrewSettings:
    """Packaged defaults, then config/crew.toml, then the file in STAFFPILOT_CREW_CONFIG or `path`."""
    data = tomllib.loads(resources.files("staffpilot_crew").joinpath("default_crew.toml").read_text())
    candidates = [Path("config/crew.toml"), os.environ.get(CONFIG_ENV_VAR), path]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            data = _merge(data, _read(Path(candidate)))
    settings = CrewSettings.model_validate(data)
    return with_measured_accuracy(settings)


def with_measured_accuracy(settings: CrewSettings) -> CrewSettings:
    """Replace prior guesses with the accuracies `staffpilot-crew eval` measured, when there are any."""
    table = Path(settings.paths.eval_results) / "routing_table.csv"
    if not table.is_file():
        return settings
    import csv

    with table.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for entry in settings.router.models:
        for row in rows:
            if row["model"] == entry.name and row["step"] in STEP_TYPES:
                entry.measured[row["step"]] = float(row["accuracy"])
                entry.measured_source = f"{row['source']}, {row['questions']} questions"
    return settings
