"""Explain a scenario that cannot meet the manager's limits, and say which limit has to move.

The hackathon's optimizer treats the service goal as a soft floor and has no budget at all,
so it never stops with "no solution": it returns the best plan it can and notes the shortfall.
The crew therefore decides feasibility itself: a plan meets the limits when it serves the
goal, stays within the budget and never hires more than the cap.

When the limits cannot all be met, a small linear program of the same year (a month by month
capacity model, solved with PuLP and CBC) gives each limit a penalized slack variable. It is
solved once per limit, with only that limit allowed to move, and the smallest move is the
estimate. Each estimate is then checked with the hackathon's own optimizer and day by day
simulator, and adjusted if the check fails, so the numbers shown to the manager are always
the planner's, never the estimate's.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any

import numpy as np
import pulp

from staffpilot_crew.constraints import Constraints


LIMITS = ("max_hires_per_month", "budget_usd", "service_goal")
# Penalty per unit of slack: a point of service goal, one hire a month, a thousand dollars.
WEIGHTS = {"service_goal": 50.0, "max_hires_per_month": 10.0, "budget_usd": 1.0}
DAYS_PER_MONTH = 30.4
SERVICE_TOLERANCE = 0.005


@dataclass
class YearInputs:
    """One country's year, as the capacity model needs it."""

    demand: np.ndarray  # eligible sign ups each month
    start_agents: int
    salary: float
    capacity_per_agent: int
    support_days: int
    fire_cost_share: float


@dataclass
class Relaxation:
    limit: str
    original: float | None
    estimate: float
    relaxed: float
    verified: bool
    kpis: dict[str, float] = field(default_factory=dict)
    peak_hires: float = 0.0
    note: str = ""

    @property
    def change(self) -> float:
        return self.relaxed - (self.original or 0.0)

    def describe(self) -> str:
        if self.limit == "max_hires_per_month":
            return f"raise the hiring cap from {self.original:.0f} to {self.relaxed:.0f} a month"
        if self.limit == "budget_usd":
            return f"raise the budget from ${self.original:,.0f} to ${self.relaxed:,.0f}"
        return f"lower the service goal from {self.original:.0%} to {self.relaxed:.0%}"

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "change": self.change, "description": self.describe()}


def relaxed_model(inputs: YearInputs, limits: Constraints, allow: str) -> dict[str, float] | None:
    """Smallest move of the `allow` limit (others fixed) that the capacity model can meet.

    Returns the slack of each limit, or None when even the moved limit cannot be met.
    """
    months = len(inputs.demand)
    throughput = inputs.capacity_per_agent * DAYS_PER_MONTH / inputs.support_days
    problem = pulp.LpProblem("relaxed_limits", pulp.LpMinimize)
    active = [pulp.LpVariable(f"active_{t}", lowBound=0) for t in range(months + 1)]
    hires = [pulp.LpVariable(f"hires_{t}", lowBound=0) for t in range(months)]
    notices = [pulp.LpVariable(f"notices_{t}", lowBound=0) for t in range(months)]
    served = [pulp.LpVariable(f"served_{t}", lowBound=0) for t in range(months)]
    slack = {name: pulp.LpVariable(f"slack_{name}", lowBound=0) for name in LIMITS}
    for name in LIMITS:
        if name != allow:
            problem += slack[name] == 0

    problem += active[0] == inputs.start_agents
    for t in range(months):
        problem += active[t + 1] == active[t] + hires[t] - notices[t]
        problem += served[t] <= throughput * active[t]
        problem += pulp.lpSum(served[: t + 1]) <= float(inputs.demand[: t + 1].sum())
    total_demand = float(inputs.demand.sum())
    goal = limits.service_goal or 0.0
    problem += pulp.lpSum(served) + slack["service_goal"] * total_demand >= goal * total_demand
    if limits.max_hires_per_month is not None:
        for t in range(months):
            problem += hires[t] <= limits.max_hires_per_month + slack["max_hires_per_month"]
    # Agents in their first month are paid too, as in the hackathon's default assumptions.
    cost = pulp.lpSum(inputs.salary / 12 * (active[t] + hires[t]) for t in range(months)) + pulp.lpSum(
        inputs.fire_cost_share * inputs.salary * n for n in notices
    )
    if limits.budget_usd is not None:
        problem += cost <= limits.budget_usd + slack["budget_usd"]
    penalty = (
        WEIGHTS["service_goal"] * 100 * slack["service_goal"]
        + WEIGHTS["max_hires_per_month"] * slack["max_hires_per_month"]
        + WEIGHTS["budget_usd"] * slack["budget_usd"] / 1000
    )
    problem += penalty + 1e-9 * cost
    problem.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=20))
    if pulp.LpStatus[problem.status] != "Optimal":
        return None
    return {name: float(var.value() or 0.0) for name, var in slack.items()}


def meets(kpis: dict[str, float], limits: Constraints, max_hires: float | None = None) -> list[str]:
    """The limits a planner result breaks, in plain words. Empty when it meets them all."""
    broken = []
    if limits.service_goal is not None and kpis["service_rate"] < limits.service_goal - SERVICE_TOLERANCE:
        broken.append(f"serves {kpis['service_rate']:.1%} of eligible advertisers, below the {limits.service_goal:.0%} goal")
    if limits.budget_usd is not None and kpis["total_cost"] > limits.budget_usd:
        broken.append(f"costs ${kpis['total_cost']:,.0f}, above the ${limits.budget_usd:,.0f} budget")
    if limits.max_hires_per_month is not None and max_hires is not None and max_hires > limits.max_hires_per_month:
        broken.append(f"hires {max_hires:.0f} in one month, above the cap of {limits.max_hires_per_month}")
    return broken


def find_relaxations(inputs: YearInputs, limits: Constraints, evaluate) -> list[Relaxation]:
    """One relaxation per limit that can move, each checked with the planner.

    `evaluate(limits)` runs the hackathon planner under the given limits and returns its KPIs
    and the largest monthly hires. The list is sorted with the smallest penalty first.
    """
    found: list[tuple[float, Relaxation]] = []
    for limit in LIMITS:
        original = getattr(limits, limit)
        if original is None:
            continue
        slack = relaxed_model(inputs, limits, allow=limit)
        if slack is None:
            continue
        relaxation = _check(limit, original, slack[limit], limits, evaluate)
        if relaxation is not None:
            penalty = WEIGHTS[limit] * {
                "service_goal": 100 * max(0.0, original - relaxation.relaxed),
                "max_hires_per_month": relaxation.change,
                "budget_usd": relaxation.change / 1000,
            }[limit]
            found.append((penalty, relaxation))
    return [r for _, r in sorted(found, key=lambda pair: pair[0])]


def _check(limit: str, original: float, slack: float, limits: Constraints, evaluate) -> Relaxation | None:
    if limit == "max_hires_per_month":
        estimate = float(math.ceil(original + slack - 1e-6))
        value = estimate
        for _ in range(8):
            trial = limits.model_copy(update={"max_hires_per_month": int(value)})
            kpis, peak = evaluate(trial)
            if not meets(kpis, trial.model_copy(update={"budget_usd": None}), peak):
                note = "" if value == estimate else f"the capacity model estimated {estimate:.0f}; the planner needed {value:.0f}"
                return Relaxation(limit, original, estimate, value, True, kpis, peak, note)
            value = math.ceil(value * 1.25 + 1)
        return Relaxation(limit, original, estimate, value, False, kpis, peak, "the planner did not reach the goal")
    if limit == "budget_usd":
        estimate = original + slack
        trial = limits.model_copy(update={"budget_usd": None})
        kpis, peak = evaluate(trial)
        if meets(kpis, trial, peak):
            return None  # the goal cannot be met under the other limits at any budget
        needed = max(estimate, kpis["total_cost"])
        note = "" if needed == estimate else "the planner's plan costs more than the capacity model estimated"
        return Relaxation(limit, original, estimate, round(needed, -3), True, kpis, peak, note)
    # The service goal: the share the planner reaches under the other limits.
    estimate = max(0.0, original - slack)
    goal = math.floor(estimate * 100) / 100
    for _ in range(10):
        trial = limits.model_copy(update={"service_goal": goal})
        kpis, peak = evaluate(trial)
        if not meets(kpis, trial, peak):
            return Relaxation(limit, original, estimate, goal, True, kpis, peak)
        goal = min(goal - 0.01, math.floor(kpis["service_rate"] * 100) / 100)
        if goal <= 0:
            break
    return None
