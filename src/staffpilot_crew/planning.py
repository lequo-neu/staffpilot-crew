"""The crew's only door into the hackathon planner.

Every number an agent reports comes from here: a scenario for one country is run through the
hackathon's optimizer and day by day simulator, and its KPIs are copied into plain floats so
they can be stored in the graph state and checked against the FactLedger.

``Planner.scenario`` returns totals but not the waiting times, so options are run with
``Planner.run`` on the context that ``Planner._scenario_context`` builds. That method is
private, but the package is pinned to the submitted commit, so it cannot change under us
(see docs/DECISIONS.md).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from staffpilot.planner import RECOMMENDED_POLICY, Planner
from staffpilot.scenarios import DemandChange, ScenarioParams

from staffpilot_crew.constraints import Constraints
from staffpilot_crew.relax import YearInputs


KPI_KEYS = (
    "net_value",
    "total_cost",
    "uplift_value",
    "served",
    "arrivals",
    "dropped",
    "service_rate",
    "mean_wait_days",
    "max_wait_days",
    "hires",
    "notices",
    "start_agents",
    "end_agents",
    "avg_active",
    "max_overflow",
    "peak_load",
)
OPTION_LABELS = {
    "recommended": "Recommended (at least 98% served)",
    "profit_only": "Profit only",
    "within_limits": "Within your limits",
    "relaxed": "Relaxed limits",
    "lifted_cap": "Without the hiring cap",
}


def country_params(scenario: dict[str, Any] | None, country: str, limits: Constraints) -> ScenarioParams:
    """The request's scenario narrowed to one country, with the manager's limits applied."""
    base = ScenarioParams.model_validate(scenario or {})
    adjustments = [
        change.model_copy(update={"countries": [country]})
        for change in base.adjustments
        if not change.countries or country in change.countries
    ]
    if limits.automation_share:
        adjustments.append(DemandChange(countries=[country], months=[], multiplier=1 - limits.automation_share))
    return base.model_copy(
        update={
            "countries": [country],
            "adjustments": adjustments,
            "max_hires_per_month": limits.max_hires_per_month,
            "service_target": limits.service_goal,
        }
    )


def evaluate(planner: Planner, params: ScenarioParams, policy: str) -> tuple[dict[str, float], float, list[str]]:
    """KPIs of one country under `params`, the largest hires in a month, and the planner's notes."""
    kpis, peak, notes, _ = evaluate_full(planner, params, policy)
    return kpis, peak, notes


def evaluate_full(
    planner: Planner, params: ScenarioParams, policy: str
) -> tuple[dict[str, float], float, list[str], dict[str, list[float]]]:
    """As `evaluate`, plus the hires and notices decided in each month of the year."""
    country = params.countries[0]
    if params.service_target is not None:
        policy = "optimized"  # the target in the context sets the floor, as Planner.scenario does
    ctx = planner._scenario_context(params, [country])
    outcome = planner.run(policy, [country], ctx, use_cache=False).outcomes[country]
    kpis = {key: float(outcome.kpis.get(key, 0.0)) for key in KPI_KEYS}
    peak = float(np.max(outcome.plan.hires[:12])) if len(outcome.plan.hires) else 0.0
    return kpis, peak, list(outcome.plan.notes), schedule(outcome.plan)


def schedule(plan: Any) -> dict[str, list[float]]:
    """Hires and notices decided in each month of the year, and the agents active in it."""
    return {
        "hires": [float(v) for v in plan.hires[:12]],
        "notices": [float(v) for v in plan.notices[:12]],
        "active": [float(v) for v in plan.active[:12]],
    }


def year_inputs(planner: Planner, params: ScenarioParams) -> YearInputs:
    country = params.countries[0]
    ctx = planner._scenario_context(params, [country])
    demand = np.asarray(ctx.forecast.demand(country)[:12], dtype=float)
    factors = (ctx.multipliers or {}).get(country)
    if factors is not None:
        demand = demand * np.asarray(factors[:12], dtype=float)
    row = ctx.dataset.countries.loc[country]
    rules = ctx.settings.rules
    return YearInputs(
        demand, int(row["agents"]), float(row["salary"]), rules.capacity_per_agent, rules.support_days, rules.fire_cost_share
    )


def option(
    country: str,
    name: str,
    kpis: dict[str, float],
    peak: float,
    notes: list[str],
    limits: Constraints,
    automation: tuple[float, float] = (0.0, 0.5),
    relaxation: dict[str, Any] | None = None,
    months: dict[str, list[float]] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schedule": months or {},
        "country": country,
        "name": name,
        "label": OPTION_LABELS[name],
        "kpis": kpis,
        "peak_hires": peak,
        "notes": notes,
        "relaxed": relaxation is not None,
        "relaxation": relaxation,
    }
    if limits.automation_share:
        share = limits.automation_share
        handled = kpis["arrivals"] * share / (1 - share)
        per_advertiser = kpis["uplift_value"] / kpis["served"] if kpis["served"] else 0.0
        low, high = (handled * per_advertiser * s for s in automation)
        record["automation"] = {
            "share": share,
            "handled": handled,
            "value_low": low,
            "value_high": high,
            "net_value_low": kpis["net_value"] + low,
            "net_value_high": kpis["net_value"] + high,
        }
    return record


def option_facts(record: dict[str, Any]) -> dict[str, float]:
    """FactLedger entries of one option, under the prefix "<Country>.<option>".

    The names follow the ledger's formatting rules: a name with "rate" or "share" prints as a
    percent, one with "cost", "value" or "budget" as dollars.
    """
    facts = dict(record["kpis"])
    facts["peak_hires"] = record["peak_hires"]
    for kind in ("hires", "notices"):
        for month, value in enumerate((record.get("schedule") or {}).get(kind, []), start=1):
            if value:
                facts[f"{kind}_m{month}"] = value
    if record.get("automation"):
        auto = record["automation"]
        facts.update(
            {
                "ai_share": auto["share"],
                "ai_handled": auto["handled"],
                "ai_value_low": auto["value_low"],
                "ai_value_high": auto["value_high"],
                "net_value_ai_low": auto["net_value_low"],
                "net_value_ai_high": auto["net_value_high"],
            }
        )
    relaxation = record.get("relaxation")
    if relaxation:
        name = {
            "max_hires_per_month": "hiring_cap",
            "budget_usd": "budget",
            "service_goal": "service_goal_share",
        }[relaxation["limit"]]
        facts[f"relaxed_{name}"] = relaxation["relaxed"]
        facts[f"original_{name}"] = relaxation["original"] or 0.0
    return facts


def recommended_policy() -> str:
    return RECOMMENDED_POLICY
