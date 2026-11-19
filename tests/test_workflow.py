"""Proposals between roles: states, notes, the timeline, stale numbers, deadlines, scenarios and Today."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
import sqlite3

import pytest

from staffpilot_crew.auth import User
from staffpilot_crew.desk import ProposalDesk, today
from staffpilot_crew.graph import CrewRunner
from staffpilot_crew.services import CrewServices
from staffpilot_crew.workflow import (
    APPROVED,
    DONE,
    DRAFT,
    EXPORT_COLUMNS,
    HIRING,
    PENDING_APPROVAL,
    PENDING_FINANCE,
    RETURNED,
    WITHDRAWN,
    WorkflowError,
    WorkflowStore,
    finance_reasons,
    hiring_deadlines,
    hiring_rows,
)


MORGAN = User("morgan", "Morgan Lee", "staffing_manager")
PRIYA = User("priya", "Priya Shah", "finance")
JORDAN = User("jordan", "Jordan Kim", "people_team")
BETA_CM = User("bea", "Bea Ortiz", "country_manager", "Beta")
REVIEWER = User("reviewer", "Avery Quinn", "reviewer")
ADMIN = User("admin", "Administrator", "admin")


class Clock:
    def __init__(self):
        self.now = datetime(2026, 11, 16, 9, 0, tzinfo=UTC)

    def __call__(self):
        return self.now


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def desk(planner, crew_settings, clock):
    services = CrewServices.build(planner, crew_settings)
    store = WorkflowStore(crew_settings.paths.app_db, crew_settings.workflow, clock)
    return ProposalDesk(store, CrewRunner(services))


def run(desk, request):
    thread = desk.runner.new_thread()
    desk.runner.run(request, thread=thread)
    return thread


def test_the_tables_exist(desk, crew_settings):
    with sqlite3.connect(crew_settings.paths.app_db) as db:
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert {"proposals", "proposal_events", "saved_scenarios", "data_versions"} <= tables


def test_a_proposal_needs_a_note_and_a_role_that_may_propose(desk):
    thread = run(desk, "What is the plan for Beta?")
    with pytest.raises(WorkflowError, match="note"):
        desk.create(MORGAN, thread, "  ")
    for user in (PRIYA, JORDAN, REVIEWER):
        with pytest.raises(WorkflowError, match="cannot create proposals"):
            desk.create(user, thread, "a note")
    with pytest.raises(WorkflowError, match="Beta only"):
        desk.create(BETA_CM, run(desk, "What is the plan for Alpha?"), "a note")
    proposal = desk.create(BETA_CM, thread, "Beta needs its plan approved")
    assert proposal.state == DRAFT
    assert proposal.events[0]["action"] == "create"
    assert proposal.verified


def test_the_full_path_with_a_timeline_and_a_decision_log_row(desk):
    proposal = desk.create(MORGAN, run(desk, "Beta needs to stay within a budget of $300k while serving at least 98%"), "First draft", submit=True)
    assert proposal.state == PENDING_FINANCE
    assert "the request sets a budget" in proposal.finance_reasons
    with pytest.raises(WorkflowError, match="note"):
        desk.act(PRIYA, proposal.id, "confirm_budget", "")
    with pytest.raises(WorkflowError, match="Staffing Manager"):
        desk.act(PRIYA, proposal.id, "approve", "trying")
    proposal = desk.act(PRIYA, proposal.id, "confirm_budget", "The budget move is covered in Q2")
    assert proposal.state == PENDING_APPROVAL
    proposal = desk.act(MORGAN, proposal.id, "approve", "Go ahead")
    assert proposal.state == APPROVED and proposal.approved_by == "Morgan Lee"
    assert desk.runner.services.log.rows()[0]["full_name"] == "Morgan Lee"
    with pytest.raises(WorkflowError, match="cannot do this"):
        desk.act(MORGAN, proposal.id, "start_hiring", "not mine")
    proposal = desk.act(JORDAN, proposal.id, "start_hiring", "Requisitions opened")
    proposal = desk.act(JORDAN, proposal.id, "mark_done", "All hires signed")
    assert proposal.state == DONE
    timeline = [(e["actor_name"], e["actor_role"], e["from_state"], e["to_state"]) for e in proposal.events]
    assert timeline == [
        ("Morgan Lee", "staffing_manager", None, DRAFT),
        ("Morgan Lee", "staffing_manager", DRAFT, PENDING_FINANCE),
        ("Priya Shah", "finance", PENDING_FINANCE, PENDING_APPROVAL),
        ("Morgan Lee", "staffing_manager", PENDING_APPROVAL, APPROVED),
        ("Jordan Kim", "people_team", APPROVED, HIRING),
        ("Jordan Kim", "people_team", HIRING, DONE),
    ]
    assert all(e["note"] and e["timestamp"] for e in proposal.events)


def test_a_small_change_skips_finance(desk):
    proposal = desk.create(MORGAN, run(desk, "What is the plan for Gamma?"), "Standard plan", submit=True)
    assert proposal.finance_reasons == []
    assert proposal.state == PENDING_APPROVAL


def test_finance_thresholds(desk):
    snapshot = {
        "chosen_totals": {"total_cost": 1_100_000.0},
        "baseline_totals": {"total_cost": 1_000_000.0},
        "schedules": {"Beta": {"hires": [3, 4, 0]}},
        "baseline_schedules": {"Beta": {"hires": [1, 0, 0]}},
        "limits": {},
    }
    reasons = finance_reasons(snapshot, desk.store.settings)
    assert len(reasons) == 2
    assert "10.0%" in reasons[0] and "+6" in reasons[1]
    snapshot["chosen_totals"]["total_cost"] = 1_010_000.0
    snapshot["schedules"]["Beta"]["hires"] = [2, 0, 0]
    assert finance_reasons(snapshot, desk.store.settings) == []


def test_return_revise_and_withdraw(desk):
    proposal = desk.create(BETA_CM, run(desk, "Beta: serve at least 98% with no more than 1 hire per month"), "Can we keep hiring low?", submit=True)
    returned = desk.act(MORGAN if proposal.state == PENDING_APPROVAL else PRIYA, proposal.id, "return", "The cap of 1 cannot serve 98%; try a higher cap")
    assert returned.state == RETURNED
    with pytest.raises(WorkflowError, match="cannot do this"):
        desk.revise(JORDAN, proposal.id, run(desk, "Beta with at most 20 hires a month"), "not mine")
    revised = desk.revise(BETA_CM, proposal.id, run(desk, "Beta: serve at least 98% with no more than 20 hires per month"), "Raised the cap to 20")
    assert revised.state == DRAFT
    assert revised.limits["max_hires_per_month"] == 20
    withdrawn = desk.act(BETA_CM, proposal.id, "withdraw", "No longer needed")
    assert withdrawn.state == WITHDRAWN
    assert [e["action"] for e in withdrawn.events] == ["create", "submit", "return", "revise", "withdraw"]


def test_new_data_makes_a_proposal_stale_and_locks_approval(desk):
    proposal = desk.create(MORGAN, run(desk, "What is the plan for Gamma?"), "Standard plan", submit=True)
    assert desk.stale(proposal) == []
    desk.store.clock.now += timedelta(minutes=5)
    desk.store.mark_data_changed("Gamma", "monthly check for March")
    assert desk.stale(proposal) == ["Gamma has newer data (monthly check for March)"]
    assert desk.allowed(MORGAN, proposal, "approve").startswith("The numbers are out of date")
    with pytest.raises(WorkflowError, match="out of date"):
        desk.act(MORGAN, proposal.id, "approve", "Go")
    desk.store.clock.now += timedelta(minutes=1)
    refreshed = desk.refresh(MORGAN, proposal.id, "Numbers refreshed after the March check")
    assert desk.stale(refreshed) == []
    assert refreshed.events[-1]["action"] == "refresh"
    assert desk.act(MORGAN, proposal.id, "approve", "Go").state == APPROVED


def test_waiting_too_long_makes_a_proposal_stale(desk, clock):
    proposal = desk.create(MORGAN, run(desk, "What is the plan for Gamma?"), "Standard plan")
    clock.now += timedelta(days=8)
    assert any("8 days old" in r for r in desk.stale(proposal))


def test_hiring_deadlines_follow_the_lead_time():
    schedules = {"Beta": {"hires": [0, 4, 0, 2, 0, 0, 0, 0, 0, 0, 0, 3]}}
    found = hiring_deadlines(schedules, 2026, 1, date(2026, 3, 28), due_soon_days=7)
    assert [(d.month, d.start_month, d.deadline.isoformat(), d.status) for d in found] == [
        (2, 3, "2026-02-01", "Overdue"),
        (4, 5, "2026-04-01", "Due soon"),
        (12, 13, "2026-12-01", "On track"),
    ]
    assert found[0].as_row()["Needed from"] == "March"
    assert all(d.status == "Done" for d in hiring_deadlines(schedules, 2026, 1, date(2026, 3, 28), done=True))


def test_the_hiring_plan_exports_with_the_agreed_columns(desk):
    proposal = desk.create(MORGAN, run(desk, "What is the plan for Beta?"), "Standard plan")
    rows = hiring_rows(proposal, desk.planner.year, desk.lead_months())
    assert rows
    assert tuple(rows[0]) == EXPORT_COLUMNS
    assert rows[0]["country"] == "Beta"
    assert sum(r["recommended_hires"] for r in rows) == sum(proposal.snapshot["schedules"]["Beta"]["hires"])


def test_saved_scenarios_rename_duplicate_and_belong_to_their_owner(desk):
    thread = run(desk, "What is the plan for Alpha?")
    values = desk.runner.values(thread)
    first = desk.store.save_scenario(MORGAN, "Alpha baseline", values, "best_quality", "manager")
    with pytest.raises(WorkflowError, match="name"):
        desk.store.save_scenario(MORGAN, " ", values, "best_quality", "manager")
    desk.store.rename_scenario(MORGAN, first, "Alpha baseline, draft 1")
    copy = desk.store.duplicate_scenario(MORGAN, first)
    mine = desk.store.scenarios(MORGAN)
    assert [s["name"] for s in mine] == ["Alpha baseline, draft 1 (copy)", "Alpha baseline, draft 1"]
    assert next(s for s in mine if s["id"] == copy)["thread"] is None
    assert next(s for s in mine if s["id"] == first)["thread"] == thread
    assert desk.store.scenarios(PRIYA) == []
    with pytest.raises(WorkflowError, match="not one of yours"):
        desk.store.rename_scenario(PRIYA, first, "mine now")
    assert len(desk.store.scenarios(ADMIN)) == 2


def test_today_lists_what_waits_on_each_role(desk):
    beta = desk.create(BETA_CM, run(desk, "Beta needs to stay within a budget of $300k while serving at least 98%"), "Beta budget", submit=True)
    gamma = desk.create(MORGAN, run(desk, "What is the plan for Gamma?"), "Gamma plan", submit=True)
    draft = desk.create(BETA_CM, run(desk, "What is the plan for Beta?"), "Beta draft")
    assert [p.id for p in today(desk, PRIYA).action_items] == [beta.id]
    assert [p.id for p in today(desk, MORGAN).action_items] == [gamma.id]
    assert [p.id for p in today(desk, BETA_CM).action_items] == [draft.id]
    assert today(desk, JORDAN).action_items == []
    assert {p.id for p in today(desk, JORDAN, "All countries").action_items} == {beta.id, gamma.id, draft.id}
    desk.act(MORGAN, gamma.id, "approve", "ok")
    view = today(desk, JORDAN)
    assert [p.id for p in view.action_items] == [gamma.id]
    assert all(d.country == "Gamma" for d in view.deadlines)
    assert {d.status for d in view.deadlines} <= {"On track", "Due soon", "Overdue"}
    assert isinstance(view.alerts, list)


def test_the_proposal_page_pieces(desk, tmp_path):
    from staffpilot_crew import app

    proposal = desk.create(MORGAN, run(desk, "Beta needs to stay within a budget of $300k while serving at least 98%"), "First draft", submit=True)
    header = app.proposal_header(desk, proposal)
    assert "Pending Finance" in header and "Goes to Finance because" in header
    frame = app.proposal_rows(desk, [proposal])
    assert list(frame.columns) == app.PROPOSAL_COLUMNS
    assert frame.iloc[0]["Needs"] == "Finance review"
    timeline = app.timeline_html(proposal)
    assert "Morgan Lee" in timeline and "First draft" in timeline
    path = app.export_csv(desk, proposal)
    with open(path, encoding="utf-8") as handle:
        assert handle.readline().strip() == ",".join(EXPORT_COLUMNS)
    desk.store.clock.now += timedelta(minutes=1)
    desk.store.mark_data_changed("Beta", "monthly check for March")
    assert "Numbers out of date" in app.proposal_header(desk, desk.store.get(proposal.id))
    assert app.proposal_rows(desk, [desk.store.get(proposal.id)]).iloc[0]["Needs"].startswith("Refresh")


def test_the_app_builds_with_the_desk(desk):
    from staffpilot_crew import app

    demo = app.build_crew_app(desk.runner.services, desk.runner, "Data: sample.", app.EXAMPLES["sample"], False, None, "disabled", desk)
    assert demo is not None
