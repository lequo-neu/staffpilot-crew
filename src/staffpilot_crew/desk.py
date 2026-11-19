"""The working desk: proposals tied to crew runs, and the Today page for each role.

``WorkflowStore`` keeps the rules and the records; ``ProposalDesk`` adds what needs the crew:
approving a proposal resumes its paused LangGraph run (so the decision log is written the
same way as before), and refreshing a stale proposal runs its request again on the latest
data. ``today`` gathers what one user should look at first: the proposals waiting on them,
the alerts for their countries and the hiring deadlines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from staffpilot.alerts import plan_alerts

from staffpilot_crew.auth import User
from staffpilot_crew.graph import CrewRunner
from staffpilot_crew.workflow import (
    APPROVED,
    DONE,
    HIRING,
    OPEN_STATES,
    PENDING_APPROVAL,
    PENDING_FINANCE,
    WAITING_ON,
    Deadline,
    Proposal,
    WorkflowError,
    WorkflowStore,
    hiring_deadlines,
)


class ProposalDesk:
    def __init__(self, store: WorkflowStore, runner: CrewRunner) -> None:
        self.store = store
        self.runner = runner

    @property
    def planner(self):
        return self.runner.services.planner

    def data_version(self) -> str:
        return self.planner.dataset.fingerprint()

    def lead_months(self) -> int:
        return int(self.planner.settings.rules.ramp_months)

    def stale(self, proposal: Proposal) -> list[str]:
        return self.store.stale_reasons(proposal, self.data_version())

    def allowed(self, user: User | None, proposal: Proposal, action: str) -> str | None:
        reason = self.store.allowed(user, proposal, action)
        if reason is None and action == "approve" and self.stale(proposal):
            return "The numbers are out of date; refresh them before approving."
        return reason

    def create(self, user: User, thread: str, note: str, option: str | None = None, submit: bool = False) -> Proposal:
        values = self.runner.values(thread)
        if not values.get("waiting"):
            raise WorkflowError("This run is no longer waiting for a decision; ask the question again.")
        proposal = self.store.create(user, values, note, self.data_version(), option)
        if submit:
            proposal = self.store.act(user, proposal.id, "submit", note)
        return proposal

    def act(self, user: User, proposal_id: int, action: str, note: str) -> Proposal:
        proposal = self.store.get(proposal_id)
        reason = self.allowed(user, proposal, action)
        if reason:
            raise WorkflowError(reason)
        if action == "approve":
            # The paused run is approved too, so the decision log gets its row as before.
            values = self.runner.values(proposal.thread) if proposal.thread else {}
            if not values.get("waiting"):
                raise WorkflowError("The run behind this proposal is closed; refresh its numbers first.")
            self.runner.approve(proposal.thread, user, option=proposal.option_name)
            return self.store.act(user, proposal_id, action, note, approved_by=user.full_name)
        return self.store.act(user, proposal_id, action, note)

    def refresh(self, user: User, proposal_id: int, note: str) -> Proposal:
        """Run the proposal's request again on the latest data, keeping its limits and its choice."""
        proposal = self.store.get(proposal_id)
        reason = self.store.allowed(user, proposal, "refresh")
        if reason:
            raise WorkflowError(reason)
        values = self.runner.run(proposal.request, constraints=proposal.limits, home_country=proposal.countries[0] if len(proposal.countries) == 1 else None)
        if not values.get("options"):
            raise WorkflowError("The request no longer gives options; open it on Decide and ask again.")
        names = [c["name"] for c in values["options"]]
        values["chosen"] = proposal.option_name if proposal.option_name in names else values["chosen"]
        return self.store.refresh(user, proposal_id, values, note, self.data_version())

    def revise(self, user: User, proposal_id: int, thread: str, note: str) -> Proposal:
        """Put the numbers of a new run (after changing the limits) on a returned proposal."""
        values = self.runner.values(thread)
        if not values.get("options"):
            raise WorkflowError("Ask a question on Decide first, then save its numbers on the proposal.")
        return self.store.refresh(user, proposal_id, values, note, self.data_version(), action="revise")

    def deadlines(self, proposals: list[Proposal], today: date) -> list[Deadline]:
        found: list[Deadline] = []
        for proposal in proposals:
            found += hiring_deadlines(
                proposal.snapshot.get("schedules", {}),
                self.planner.year,
                self.lead_months(),
                today,
                self.store.settings.due_soon_days,
                done=proposal.state == DONE,
            )
        return sorted(found, key=lambda d: (d.deadline, d.country))


# The Today page.

SCOPES = ("My scope", "All countries", "Overdue or urgent")


@dataclass
class TodayView:
    action_items: list[Proposal] = field(default_factory=list)
    alerts: list[dict[str, Any]] = field(default_factory=list)
    deadlines: list[Deadline] = field(default_factory=list)
    stale: dict[int, list[str]] = field(default_factory=dict)


def waiting_on(user: User, proposal: Proposal) -> bool:
    """True when the proposal's next step belongs to this user."""
    owners = WAITING_ON.get(proposal.state, ())
    if user.role == "admin":
        return proposal.state in WAITING_ON
    if "creator" in owners:
        return proposal.creator_username == user.username
    return user.role in owners


def in_scope(user: User, countries: list[str]) -> bool:
    if user.role == "country_manager":
        return user.assigned_country in countries
    return True


def today(desk: ProposalDesk, user: User, scope: str = "My scope", on: date | None = None, states: list[str] | None = None) -> TodayView:
    on = on or desk.store.clock().date()
    view = TodayView()
    everything = desk.store.list()
    if scope == "My scope":
        mine = [p for p in everything if waiting_on(user, p) and in_scope(user, p.countries)]
    else:
        mine = [p for p in everything if p.state in WAITING_ON and (scope != "My scope")]
    if states:
        mine = [p for p in mine if p.state in states]
    view.stale = {p.id: desk.stale(p) for p in everything if p.state in OPEN_STATES}
    hiring = [p for p in everything if p.state in (APPROVED, HIRING, DONE)]
    if user.role == "country_manager" and scope == "My scope":
        hiring = [p for p in hiring if in_scope(user, p.countries)]
    view.deadlines = desk.deadlines(hiring, on)
    if scope == "Overdue or urgent":
        urgent = {(d.country) for d in view.deadlines if d.status in ("Overdue", "Due soon")}
        mine = [p for p in mine if view.stale.get(p.id) or set(p.countries) & urgent or p.state in (PENDING_FINANCE, PENDING_APPROVAL)]
        view.deadlines = [d for d in view.deadlines if d.status in ("Overdue", "Due soon")]
    view.action_items = mine
    view.alerts = [
        {"level": a.level, "countries": list(a.countries), "message": a.message}
        for a in plan_alerts(desk.planner)
        if a.countries and (scope != "My scope" or user.role != "country_manager" or user.assigned_country in a.countries)
    ]
    return view
