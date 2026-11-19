"""Pick a model for each step from the step type, the request's difficulty and the user's mode.

Modes:

* ``best_quality``: the model the team assigned to that step in the proposal, if it can be
  reached; otherwise the most accurate model that can.
* ``lowest_cost``: the cheapest hosted model whose accuracy on the step clears the bar for
  the request's difficulty.
* ``private``: only the local model, so no data leaves the computer.

When no model can be reached the deterministic offline backend answers, and the reason says
so. Every choice records whether the accuracy behind it was measured or is a prior guess.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from staffpilot_crew.config import CrewSettings, ModelEntry
from staffpilot_crew.models import OFFLINE


Mode = Literal["best_quality", "lowest_cost", "private"]
MODES: tuple[Mode, ...] = ("best_quality", "lowest_cost", "private")
MODE_LABELS = {"best_quality": "Best quality", "lowest_cost": "Lowest cost", "private": "Private"}


@dataclass
class Difficulty:
    countries: int
    constraints: int
    conflicts: list[str] = field(default_factory=list)
    level: str = "simple"

    def describe(self) -> str:
        parts = [f"{self.countries} {'country' if self.countries == 1 else 'countries'}"]
        parts.append(f"{self.constraints} {'limit' if self.constraints == 1 else 'limits'}")
        if self.conflicts:
            parts.append("limits that pull against each other: " + "; ".join(self.conflicts))
        return f"{self.level} ({', '.join(parts)})"


def assess(countries: int, constraints: int, conflicts: list[str], hard_countries: int = 3) -> Difficulty:
    if countries >= hard_countries or conflicts:
        level = "hard"
    elif countries == 2 or constraints >= 2:
        level = "moderate"
    else:
        level = "simple"
    return Difficulty(countries, constraints, list(conflicts), level)


@dataclass
class RouteChoice:
    step: str
    model: str
    label: str
    spec: str
    reason: str
    accuracy: float | None = None
    accuracy_source: str = ""  # "measured", "prior" or "" for the offline backend
    escalated: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def offline_choice(step: str, reason: str) -> RouteChoice:
    return RouteChoice(step, OFFLINE, "Deterministic offline backend", OFFLINE, reason)


class Router:
    def __init__(self, crew: CrewSettings, available: list[str], local_ready: bool = False) -> None:
        self.crew = crew
        self.available = set(available)
        self.local_ready = local_ready

    def _entries(self, mode: Mode) -> list[ModelEntry]:
        models = self.crew.router.models
        if mode == "private":
            return [m for m in models if m.provider == "local" and (m.name in self.available or self.local_ready)]
        return [m for m in models if m.provider != "local" and m.name in self.available]

    def _choice(self, entry: ModelEntry, step: str, reason: str, escalated: bool = False) -> RouteChoice:
        accuracy, source = entry.accuracy(step)
        return RouteChoice(step, entry.name, entry.label, entry.spec, reason, accuracy, source, escalated)

    def choose(self, step: str, difficulty: Difficulty, mode: Mode = "best_quality") -> RouteChoice:
        entries = self._entries(mode)
        if not entries:
            local = [m for m in self.crew.router.models if m.provider == "local"]
            if mode != "private" and local and self.local_ready:
                return self._choice(local[0], step, "no hosted model has a key; using the local model")
            why = "no local model server is running" if mode == "private" else "no model has a key"
            return offline_choice(step, f"{why}; the deterministic offline backend answers")
        if mode == "private":
            return self._choice(entries[0], step, "private mode: only the local model is used")
        bar = self.crew.router.bar[difficulty.level]
        if mode == "best_quality":
            assigned = [m for m in entries if step in m.roles]
            if assigned:
                return self._choice(assigned[0], step, f"assigned to the {step} step in the proposal")
            best = max(entries, key=lambda m: (m.accuracy(step)[0], -m.cost_rank))
            accuracy, source = best.accuracy(step)
            return self._choice(best, step, f"most accurate available on {step} ({accuracy:.0%}, {source})")
        passing = [m for m in entries if m.accuracy(step)[0] >= bar]
        if passing:
            cheapest = min(passing, key=lambda m: (m.cost_rank, -m.accuracy(step)[0]))
            accuracy, source = cheapest.accuracy(step)
            return self._choice(
                cheapest,
                step,
                f"cheapest model at or above the {bar:.0%} bar for a {difficulty.level} request "
                f"({accuracy:.0%}, {source})",
            )
        best = max(entries, key=lambda m: m.accuracy(step)[0])
        accuracy, source = best.accuracy(step)
        return self._choice(best, step, f"no model reaches the {bar:.0%} bar; the most accurate one ({accuracy:.0%}, {source})")

    def escalate(self, current: RouteChoice, difficulty: Difficulty, mode: Mode = "best_quality") -> RouteChoice:
        """A stronger model for the same step, or the same one with a note when there is none."""
        entries = self._entries(mode) if mode == "private" else self._entries("best_quality")
        now = current.accuracy if current.accuracy is not None else -1.0
        stronger = [m for m in entries if m.name != current.model and m.accuracy(current.step)[0] > now]
        if not stronger:
            note = "no stronger model is available; retrying with the same one"
            return RouteChoice(**{**current.as_dict(), "reason": note, "escalated": False})
        # The model assigned to this step wins a tie; then the cheaper one.
        best = max(
            stronger, key=lambda m: (m.accuracy(current.step)[0], current.step in m.roles, -m.cost_rank)
        )
        accuracy, source = best.accuracy(current.step)
        return self._choice(
            best, current.step, f"escalated after a failed number check ({accuracy:.0%}, {source})", escalated=True
        )
