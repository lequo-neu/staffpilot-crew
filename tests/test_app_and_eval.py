"""The app's pieces and the evaluation, offline."""

from __future__ import annotations

import pandas as pd
import pytest

from staffpilot_crew import app, evaluation
from staffpilot_crew.graph import CrewRunner
from staffpilot_crew.services import CrewServices


@pytest.fixture
def runner(planner, crew_settings):
    return CrewRunner(CrewServices.build(planner, crew_settings))


def test_the_app_builds(runner):
    demo = app.build_crew_app(runner.services, runner, "Data: sample.", app.EXAMPLES["sample"], planner_pages=False)
    assert demo is not None


def test_cards_trace_and_status_show_the_run(runner):
    values = runner.run("Beta: serve at least 98% with no more than 1 hire per month", thread="cards")
    html = app.run_cards(values)
    assert "Relaxed limits" in html and "Best match for your request" in html and "Limits moved" in html
    assert "Verified numbers" in html
    # Each figure shows its change against the plan the year started with.
    assert "class='delta'" in html and "In brackets: the change against the plan" in html
    frame = app.trace_frame(values)
    assert list(frame.columns) == app.TRACE_COLUMNS
    assert {"Planner", "Analyst", "Optimizer", "Decision critic", "Writer"} <= set(frame["Agent"])
    assert "Options ready" in app.status_text(values)
    assert app.cost_text(runner, ["cards"]) == ""
    assert "Written by" not in app.trace_html(values)
    assert app.answer_markdown(values, "Beta?").startswith("<div class='crew-question'>You asked: Beta?</div>")


def test_unverified_numbers_are_flagged_on_the_cards(runner):
    values = runner.run("What is the plan for Alpha?", thread="flagged")
    assert "Verified numbers" in app.run_cards(values)
    assert "Unverified" in app.run_cards({**values, "flagged": ["Net value is $99.9M."]})


def test_changes_are_written_plainly():
    assert app.money_change(320_000) == "+$320K"
    assert app.money_change(-1_250_000) == "-$1.2M"
    assert app.money_change(0.2) == "no change"
    assert app.count_change(6) == "+6"
    assert app.count_change(-7) == "-7"


def test_copy_formats_are_ready_to_paste(runner):
    values = runner.run("What is the plan for Gamma?", thread="copy")
    user = app.User("jordan", "Jordan Kim", "people_team")
    email = app.copy_text(values, "Hiring notice email", user)
    assert email.startswith("Subject: Hiring and notice plan for Gamma")
    assert email.rstrip().endswith("Jordan Kim")
    assert "**" not in app.copy_text(values, "Country memo", user)
    assert app.copy_text({}, "Country memo", user) == ""


def test_role_view_rewrites_the_answer_without_a_model(runner):
    values = runner.run("What is the plan for Alpha?", thread="roles")
    memo = app.role_view(values, "country_manager")
    finance = app.role_view(values, "finance")
    assert memo.startswith("## Staffing memo: Alpha")
    assert finance.startswith("## Finance note: Alpha")
    assert "{{" not in memo + finance


def test_offline_evaluation_scores_every_sample_request(planner, crew_settings):
    services = CrewServices.build(planner, crew_settings)
    rows, summary = evaluation.evaluate(services, "sample")
    assert len(rows) == len(evaluation.load_questions("sample"))
    assert summary.loc[0, "setup"] == "Offline backend (no model)"
    assert summary.loc[0, "task_accuracy"] == 1.0
    assert summary.loc[0, "cost_note"] == "no model calls"
    folder = evaluation.save(rows, summary, services)
    assert (folder / "crew_eval_summary.csv").exists()
    assert evaluation.pareto_chart(summary) is not None


def test_pareto_frontier_keeps_only_setups_nobody_beats():
    summary = pd.DataFrame(
        {
            "setup": ["router", "cheap", "dear", "worse"],
            "kind": ["router", "single", "single", "single"],
            "task_accuracy": [0.95, 0.80, 0.96, 0.70],
            "cost_per_request_usd": [0.004, 0.001, 0.02, 0.01],
        }
    )
    assert evaluation.pareto(summary, "cost_per_request_usd") == [True, True, True, False]


def test_single_model_baseline_sends_every_step_to_one_model(planner, crew_settings):
    from staffpilot.agent.backends import OfflineBackend

    services = CrewServices.build(planner, crew_settings, ["openai"], lambda entry, s: OfflineBackend())
    setup = evaluation.Setup("Only GPT 5 nano", "single", model="gpt_nano")
    rows = evaluation.run_setup(services, setup, evaluation.load_questions("sample")[:2])
    assert all(r["models"] == "gpt_nano" for r in rows)
    table = evaluation.routing_table(pd.DataFrame(rows), services)
    assert set(table["step"]) == {"classify", "write"}
    assert services.router_override is None
