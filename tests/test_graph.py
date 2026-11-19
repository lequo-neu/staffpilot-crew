"""The graph end to end, offline: the normal path, parallel countries, bounded loops,
infeasible limits, the number check, and pausing and resuming for the manager."""

from __future__ import annotations

from langgraph.errors import GraphRecursionError
import pytest
from staffpilot.agent.backends import OfflineBackend

from staffpilot_crew import agents
from staffpilot_crew.auth import User
from staffpilot_crew.graph import CrewRunner, sqlite_checkpointer
from staffpilot_crew.services import CrewServices


MANAGER = User("morgan", "Morgan Lee", "staffing_manager")


def runner_for(planner, crew_settings, providers=None, factory=None, checkpointer=None):
    services = CrewServices.build(planner, crew_settings, providers, factory)
    return CrewRunner(services, checkpointer)


def actions(values, agent):
    return [e for e in values["trace"] if e["agent"] == agent]


def test_one_country_runs_to_the_approval_pause(planner, crew_settings):
    values = runner_for(planner, crew_settings).run("What is the plan for Alpha?", thread="one")
    assert values["waiting"] and values["next"] == ["approval"]
    assert values["approval_status"] == "Waiting for approval"
    assert values["chosen"] == "recommended"
    assert values["plan"]["countries"] == ["Alpha"]
    assert "{{" not in values["answer"]
    assert values["unverified"] == []
    assert "Nothing is applied until you approve" in values["answer"]
    # Offline, every step says which backend answered and why.
    assert all(r["model"] == "offline" for r in values["routing_trace"])


def test_countries_run_in_parallel_branches_and_merge(planner, crew_settings):
    values = runner_for(planner, crew_settings).run("Compare Alpha, Beta and Gamma", thread="three")
    assert sorted(values["country_results"]) == ["Alpha", "Beta", "Gamma"]
    assert sorted(values["analyses"]) == ["Alpha", "Beta", "Gamma"]
    assert len(actions(values, "Analyst")) == 3
    assert len(actions(values, "Optimizer")) == 3
    assert values["plan"]["difficulty"]["level"] == "hard"
    # Totals over the three countries come from the per country facts.
    card = next(c for c in values["options"] if c["name"] == "recommended")
    by_country = sum(values["fact_ledger"][f"{c}.recommended.net_value"] for c in ("Alpha", "Beta", "Gamma"))
    assert card["totals"]["net_value"] == pytest.approx(by_country)


def test_a_request_without_countries_asks_back_before_any_planning(planner, crew_settings):
    values = runner_for(planner, crew_settings).run("What if sign ups rise 20%?", thread="vague")
    assert values["approval_status"] == "Needs details"
    assert "Which countries" in values["answer"]
    assert not values["waiting"]
    assert not values.get("country_results")


def test_an_out_of_scope_request_is_answered_with_what_the_crew_can_do(planner, crew_settings):
    values = runner_for(planner, crew_settings).run("What should we do?", thread="scope")
    assert values["approval_status"] == "Needs details"
    assert "staffing plans" in values["answer"]


def test_all_countries_can_be_asked_for_in_words(planner, crew_settings):
    values = runner_for(planner, crew_settings).run("What if sign ups rise 20% in all countries?", thread="all")
    assert values["plan"]["countries"] == ["Alpha", "Beta", "Gamma"]
    assert values["plan"]["scenario"]["demand_multiplier"] == pytest.approx(1.2)


def test_infeasible_limits_get_a_relaxed_option_checked_by_the_planner(planner, crew_settings):
    values = runner_for(planner, crew_settings).run(
        "Beta: serve at least 98% with no more than 1 hire per month", thread="tight"
    )
    result = values["country_results"]["Beta"]
    assert result["strict_issues"] and "below the 98% goal" in result["strict_issues"][0]
    relaxation = result["relaxations"][0]
    assert relaxation["limit"] == "max_hires_per_month"
    assert relaxation["verified"]
    assert relaxation["relaxed"] > 1
    relaxed = next(o for o in result["options"] if o["name"] == "relaxed")
    assert relaxed["relaxed"]
    assert relaxed["kpis"]["service_rate"] >= 0.975
    assert relaxed["peak_hires"] <= relaxation["relaxed"]
    assert values["chosen"] == "relaxed"
    assert "raise the hiring cap from 1 to" in values["answer"]


def test_ask_for_changes_resumes_from_the_optimizer(planner, crew_settings):
    runner = runner_for(planner, crew_settings)
    first = runner.run("Beta: serve at least 98% with no more than 1 hire per month", thread="change")
    before = len(first["trace"])
    after = runner.ask_for_changes("change", "allow at most 20 hires a month", user=MANAGER)
    new = after["trace"][before:]
    agents_after = [e["agent"] for e in new]
    assert agents_after[0] == "Staffing Manager"
    assert agents_after[1] == "Optimizer"
    assert "Planner" not in agents_after and "Analyst" not in agents_after
    assert after["user_constraints"]["max_hires_per_month"] == 20
    assert after["retry_counts"] == {"optimizer": 0, "writer": 0}
    assert after["country_results"]["Beta"]["version"] > first["country_results"]["Beta"]["version"]
    assert after["approval_status"] == "Waiting for approval"
    assert after["chosen"] == "within_limits"


def test_approve_logs_the_decision_and_ends_the_run(planner, crew_settings):
    runner = runner_for(planner, crew_settings)
    runner.run("What is the plan for Gamma?", thread="ok")
    values = runner.approve("ok", MANAGER)
    assert values["approval_status"] == "Approved"
    assert not values["waiting"]
    rows = runner.services.log.rows()
    assert len(rows) == 1
    assert (rows[0]["username"], rows[0]["full_name"], rows[0]["role"]) == ("morgan", "Morgan Lee", "staffing_manager")
    assert rows[0]["countries"] == "Gamma"
    assert rows[0]["status"] == "approved"


def test_a_paused_run_survives_a_restart_with_the_sqlite_checkpointer(planner, crew_settings):
    path = crew_settings.paths.checkpoints
    runner_for(planner, crew_settings, checkpointer=sqlite_checkpointer(path)).run("Plan for Alpha", thread="saved")
    restarted = runner_for(planner, crew_settings, checkpointer=sqlite_checkpointer(path))
    assert restarted.values("saved")["waiting"]
    assert restarted.approve("saved", MANAGER)["approval_status"] == "Approved"


def test_the_optimizer_loop_stops_after_two_retries(planner, crew_settings, monkeypatch):
    monkeypatch.setattr(agents, "rule_issues", lambda services, record, limits: ["a rule that never passes"])
    values = runner_for(planner, crew_settings).run("Plan for Alpha", thread="loop")
    assert values["retry_counts"]["optimizer"] == 2
    assert len(actions(values, "Optimizer")) == 3  # the first run and two retries
    assert values["open_issues"] == ["Alpha: a rule that never passes"]
    assert "Still open after two more tries" in values["answer"]
    assert values["waiting"]


def test_recursion_limit_is_a_hard_stop(planner, crew_settings):
    with pytest.raises(GraphRecursionError):
        runner_for(planner, crew_settings).run("Plan for Alpha", thread="short", recursion_limit=3)


class Scripted(OfflineBackend):
    """An online model stand in: offline reading, scripted prose, token counts reported."""

    def __init__(self, text):
        self.text = text
        self.usage = {"input": 0, "output": 0}
        self.calls = 0

    def compose(self, question, route, outputs, facts, feedback=""):
        self.calls += 1
        self.usage["input"] += 500
        self.usage["output"] += 50
        return self.text


MADE_UP = "Net value is $99.9M this year."
GROUNDED = "Net value is {{Alpha.recommended.net_value}} for a cost of {{Alpha.recommended.total_cost}}."


def scripted_factory(texts):
    made = {}

    def factory(entry, settings):
        made[entry.name] = Scripted(texts[entry.name])
        return made[entry.name]

    return factory, made


def test_writer_escalates_on_the_second_retry_and_then_passes(planner, crew_settings):
    factory, _made = scripted_factory({"gpt_nano": MADE_UP, "gemini_flash": GROUNDED})
    runner = runner_for(planner, crew_settings, ["openai", "google"], factory)
    values = runner.run("Plan for Alpha", thread="escalate", mode="lowest_cost")
    writers = [r for r in values["routing_trace"] if r["agent"] == "Writer"]
    assert [w["model"] for w in writers] == ["gpt_nano", "gpt_nano", "gemini_flash"]
    assert writers[-1]["escalated"]
    assert values["retry_counts"]["writer"] == 2
    assert values["unverified"] == []
    assert "$99.9M" not in values["answer"]
    # Token counts become a measured cost for priced models only.
    costs = {c["model"]: c["cost_source"] for c in values["calls"] if c["agent"] == "Writer"}
    assert costs == {"gpt_nano": "measured", "gemini_flash": "unknown price"}


def test_numbers_that_never_get_a_source_are_left_out(planner, crew_settings):
    factory, _ = scripted_factory({"gpt_nano": MADE_UP, "gemini_flash": MADE_UP})
    values = runner_for(planner, crew_settings, ["openai", "google"], factory).run(
        "Plan for Alpha", thread="flag", mode="lowest_cost"
    )
    assert values["retry_counts"]["writer"] == 2
    assert "$99.9M" not in values["answer"]
    assert values["flagged"] == [MADE_UP]
    assert "could not be traced to the planner" in values["answer"]
    assert values["waiting"]


def test_audience_changes_the_writer_output(planner, crew_settings):
    runner = runner_for(planner, crew_settings)
    memo = runner.run("Write a memo for the Alpha country manager", thread="memo")
    assert memo["audience"] == "country_manager"
    assert "## Staffing memo: Alpha" in memo["answer"]
    finance = runner.run("A note for finance on Alpha", thread="finance")
    assert "## Finance note: Alpha" in finance["answer"]
    assert finance["unverified"] == []


def test_monthly_update_proposes_next_month_for_approval(planner, crew_settings):
    runner = runner_for(planner, crew_settings)
    demand = planner.forecast.demand("Beta")[:2]
    monthly = {"country": "Beta", "as_of": 3, "actual_signups": [float(v * 1.3) for v in demand], "headcount_change": 0}
    values = runner.run("Monthly update for Beta", thread="monthly", monthly=monthly)
    assert values["plan"]["intent"] == "monthly_update"
    assert values["chosen"] == "update"
    assert values["fact_ledger"]["Beta.analysis.signup_change"] == pytest.approx(0.3, abs=0.01)
    assert "For March in Beta: hire" in values["answer"]
    assert values["waiting"]
