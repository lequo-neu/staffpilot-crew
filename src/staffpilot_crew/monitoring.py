"""Review each monthly update without being asked.

When a manager loads the real sign ups of the months so far, the gap with the forecast is
measured first. A gap larger than ``graph.monitor_gap`` (or any change in headcount nobody
planned) starts a crew run by itself; the run ends, as always, in option cards that wait for
approval. A small gap is reported and nothing is started.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from staffpilot_crew.graph import CrewRunner


MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December")


@dataclass
class MonthlyReview:
    country: str
    as_of: int
    gap: float
    started: bool
    message: str
    thread: str = ""
    values: dict[str, Any] = field(default_factory=dict)


def signup_gap(runner: CrewRunner, country: str, as_of: int, actual_signups: list[float]) -> float:
    forecast = np.asarray(runner.services.planner.forecast.demand(country)[: as_of - 1], dtype=float)
    expected = float(forecast.sum())
    return float(np.sum(actual_signups)) / expected - 1 if expected else 0.0


def review_monthly_update(
    runner: CrewRunner,
    country: str,
    as_of: int,
    actual_signups: list[float],
    headcount_change: int = 0,
    user: str = "",
    mode: str = "best_quality",
    thread: str | None = None,
) -> MonthlyReview:
    planner = runner.services.planner
    resolved = planner.resolve_country(country)
    if resolved is None:
        raise ValueError(f"No data for {country}")
    if not 2 <= as_of <= 12 or len(actual_signups) != as_of - 1:
        raise ValueError("Give one count of sign ups for each month before the update month (February to December)")
    if any(v is None or v < 0 for v in actual_signups):
        raise ValueError("Sign up counts must be whole numbers of zero or more")
    gap = signup_gap(runner, resolved, as_of, actual_signups)
    threshold = runner.services.crew.graph.monitor_gap
    seen = f"{MONTHS[0]} to {MONTHS[as_of - 2]}" if as_of > 2 else MONTHS[0]
    if abs(gap) < threshold and headcount_change == 0:
        message = (
            f"Sign ups in {resolved} from {seen} were {abs(gap):.0%} {'above' if gap >= 0 else 'below'} the forecast, "
            f"inside the {threshold:.0%} band, so no review was started."
        )
        return MonthlyReview(resolved, as_of, gap, False, message)
    thread = thread or runner.new_thread()
    monthly = {"country": resolved, "as_of": as_of, "actual_signups": [float(v) for v in actual_signups], "headcount_change": int(headcount_change)}
    request = f"Monthly update for {resolved} as of {MONTHS[as_of - 1]}"
    values = runner.run(request, thread=thread, monthly=monthly, user=user, mode=mode)
    reason = f"sign ups {abs(gap):.0%} {'above' if gap >= 0 else 'below'} the forecast" if abs(gap) >= threshold else "a headcount change"
    message = f"A review of {resolved} was started by itself after {reason}; its proposal waits for approval."
    return MonthlyReview(resolved, as_of, gap, True, message, thread, values)
