"""The parts under the graph: reducers, the relaxed model, automation share and monitoring."""

from __future__ import annotations

import numpy as np
import pytest

from staffpilot_crew import planning
from staffpilot_crew.constraints import Constraints
from staffpilot_crew.graph import CrewRunner
from staffpilot_crew.monitoring import review_monthly_update
from staffpilot_crew.relax import YearInputs, find_relaxations, meets, relaxed_model
from staffpilot_crew.services import CrewServices
from staffpilot_crew.state import latest_by_country, merge_dicts


def test_reducers_merge_parallel_writes():
    assert merge_dicts({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}
    old = {"Alpha": {"version": 2, "v": "new"}}
    assert latest_by_country(old, {"Alpha": {"version": 1, "v": "stale"}})["Alpha"]["v"] == "new"
    assert latest_by_country(old, {"Alpha": {"version": 3, "v": "newer"}, "Beta": {"version": 0}}) == {
        "Alpha": {"version": 3, "v": "newer"},
        "Beta": {"version": 0},
    }


INPUTS = YearInputs(np.full(12, 100.0), start_agents=5, salary=36_000, capacity_per_agent=10, support_days=60, fire_cost_share=0.4)


def test_relaxed_model_moves_only_the_limit_it_is_allowed_to_move():
    limits = Constraints(service_goal=0.98, max_hires_per_month=1)
    cap = relaxed_model(INPUTS, limits, allow="max_hires_per_month")
    assert cap["max_hires_per_month"] > 0
    assert cap["service_goal"] == 0
    goal = relaxed_model(INPUTS, limits, allow="service_goal")
    assert goal["service_goal"] > 0
    assert goal["max_hires_per_month"] == 0


def test_relaxed_model_finds_no_slack_when_the_limits_can_be_met():
    loose = Constraints(service_goal=0.5, max_hires_per_month=50)
    assert relaxed_model(INPUTS, loose, allow="max_hires_per_month") == {
        "max_hires_per_month": 0.0,
        "budget_usd": 0.0,
        "service_goal": 0.0,
    }


def test_meets_names_each_broken_limit():
    kpis = {"service_rate": 0.5, "total_cost": 2_000_000.0}
    broken = meets(kpis, Constraints(service_goal=0.98, budget_usd=1_000_000, max_hires_per_month=2), max_hires=5)
    assert len(broken) == 3


def test_every_relaxation_is_checked_with_the_planner(planner):
    limits = Constraints(service_goal=0.98, max_hires_per_month=1)
    base = planning.country_params(None, "Beta", limits)

    def check(trial):
        kpis, peak, _ = planning.evaluate(planner, planning.country_params(None, "Beta", trial), "optimized")
        return kpis, peak

    found = find_relaxations(planning.year_inputs(planner, base), limits, check)
    assert {r.limit for r in found} == {"max_hires_per_month", "service_goal"}
    for relaxation in found:
        assert relaxation.verified
        trial = limits.model_copy(update={relaxation.limit: relaxation.relaxed})
        assert not meets(relaxation.kpis, trial.model_copy(update={"budget_usd": None}), relaxation.peak_hires)


def test_automation_share_lowers_human_demand_and_reports_a_range(planner, crew_settings):
    services = CrewServices.build(planner, crew_settings)
    runner = CrewRunner(services)
    values = runner.run("What if an AI assistant onboards 30% of advertisers in Alpha?", thread="ai")
    assert values["user_constraints"]["automation_share"] == pytest.approx(0.3)
    plain = planning.evaluate(planner, planning.country_params(None, "Alpha", Constraints()), "service_98")[0]
    auto = values["country_results"]["Alpha"]["options"][0]
    assert auto["kpis"]["arrivals"] < plain["arrivals"] * 0.8
    assert auto["automation"]["value_low"] == 0.0
    assert auto["automation"]["net_value_high"] > auto["automation"]["net_value_low"]
    assert "handled by an AI assistant" in values["answer"]
    assert values["unverified"] == []


def test_monitoring_starts_a_review_only_for_a_real_gap(planner, crew_settings):
    runner = CrewRunner(CrewServices.build(planner, crew_settings))
    forecast = planner.forecast.demand("Alpha")[:2]
    quiet = review_monthly_update(runner, "Alpha", 3, [float(v) for v in forecast])
    assert not quiet.started
    assert "no review was started" in quiet.message
    busy = review_monthly_update(runner, "Alpha", 3, [float(v * 1.25) for v in forecast], user="kevin")
    assert busy.started
    assert busy.gap == pytest.approx(0.25, abs=0.01)
    assert busy.values["waiting"]
    assert busy.values["chosen"] == "update"


@pytest.mark.parametrize("bad", [[100.0], [100.0, -1.0]])
def test_monitoring_rejects_bad_inputs(planner, crew_settings, bad):
    runner = CrewRunner(CrewServices.build(planner, crew_settings))
    with pytest.raises(ValueError, match=r"sign ups|Sign up"):
        review_monthly_update(runner, "Alpha", 3, bad)
