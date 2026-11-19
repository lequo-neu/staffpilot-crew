"""Proposals that move between roles, their timeline, stale numbers, deadlines and saved scenarios.

A proposal is a plan someone wants approved: the option they chose from a crew run, with a
snapshot of its numbers. It moves through these states:

    Draft -> Pending Finance (only when it moves the budget enough) -> Pending approval
          -> Approved -> Hiring triggered -> Done

with two side exits: Returned (sent back with a reason; the creator revises it into a new
Draft) and Withdrawn (the creator takes it back). Every action needs a note, and every action
is a row in ``proposal_events``, which is the proposal's timeline.

A proposal is stale when its country's data changed after its numbers were made, when the
dataset itself changed, or when it has waited longer than ``workflow.stale_days``. A stale
proposal cannot be approved until its numbers are refreshed.

Everything lives in the app's SQLite file next to the users and the decision log.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
import json
from pathlib import Path
import sqlite3
from typing import Any

from staffpilot_crew.auth import User
from staffpilot_crew.config import WorkflowSettings


DRAFT = "Draft"
PENDING_FINANCE = "Pending Finance"
PENDING_APPROVAL = "Pending approval"
APPROVED = "Approved"
HIRING = "Hiring triggered"
DONE = "Done"
RETURNED = "Returned"
WITHDRAWN = "Withdrawn"
STATES = (DRAFT, PENDING_FINANCE, PENDING_APPROVAL, APPROVED, HIRING, DONE, RETURNED, WITHDRAWN)
OPEN_STATES = (DRAFT, RETURNED, PENDING_FINANCE, PENDING_APPROVAL)
CLOSED_STATES = (DONE, WITHDRAWN)


@dataclass(frozen=True)
class Action:
    name: str
    label: str
    from_states: tuple[str, ...]
    roles: tuple[str, ...]
    creator_may: bool = False  # the proposal's creator may do it whatever their role
    to_state: str | None = None  # None: decided when it runs (submit)


ACTIONS: dict[str, Action] = {
    a.name: a
    for a in (
        Action("submit", "Submit for approval", (DRAFT,), ("admin",), creator_may=True),
        Action("confirm_budget", "Confirm budget", (PENDING_FINANCE,), ("finance", "admin"), to_state=PENDING_APPROVAL),
        Action("approve", "Approve", (PENDING_APPROVAL,), ("staffing_manager", "admin"), to_state=APPROVED),
        Action("return", "Return with a reason", (PENDING_FINANCE, PENDING_APPROVAL), ("finance", "staffing_manager", "admin"), to_state=RETURNED),
        Action("start_hiring", "Start hiring", (APPROVED,), ("people_team", "admin"), to_state=HIRING),
        Action("mark_done", "Mark as done", (HIRING,), ("people_team", "admin"), to_state=DONE),
        Action("withdraw", "Withdraw", OPEN_STATES, ("admin",), creator_may=True, to_state=WITHDRAWN),
        Action("revise", "Save the new numbers as a draft", (RETURNED, DRAFT), ("admin",), creator_may=True, to_state=DRAFT),
        Action("refresh", "Refresh the numbers", OPEN_STATES, ("staffing_manager", "finance", "admin"), creator_may=True),
    )
}
# Only one role works on each open state; the Today page lists those for that role.
WAITING_ON = {
    DRAFT: ("creator",),
    RETURNED: ("creator",),
    PENDING_FINANCE: ("finance",),
    PENDING_APPROVAL: ("staffing_manager",),
    APPROVED: ("people_team",),
    HIRING: ("people_team",),
}

SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS proposals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        state TEXT NOT NULL,
        creator_username TEXT NOT NULL,
        creator_name TEXT NOT NULL,
        creator_role TEXT NOT NULL,
        countries TEXT NOT NULL,
        request TEXT NOT NULL,
        thread TEXT,
        option_name TEXT NOT NULL,
        option_label TEXT NOT NULL,
        limits TEXT NOT NULL,
        snapshot TEXT NOT NULL,
        verified INTEGER NOT NULL,
        needs_finance INTEGER NOT NULL,
        finance_reasons TEXT NOT NULL,
        data_version TEXT NOT NULL,
        snapshot_at TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        approved_by TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS proposal_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        proposal_id INTEGER NOT NULL REFERENCES proposals(id),
        timestamp TEXT NOT NULL,
        actor_username TEXT NOT NULL,
        actor_name TEXT NOT NULL,
        actor_role TEXT NOT NULL,
        action TEXT NOT NULL,
        from_state TEXT,
        to_state TEXT NOT NULL,
        note TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS saved_scenarios (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        owner_username TEXT NOT NULL,
        countries TEXT NOT NULL,
        request TEXT NOT NULL,
        mode TEXT NOT NULL,
        audience TEXT NOT NULL,
        constraints TEXT NOT NULL,
        thread TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS data_versions (
        country TEXT PRIMARY KEY,
        updated_at TEXT NOT NULL,
        reason TEXT NOT NULL
    )
    """,
)


class WorkflowError(ValueError):
    """An action that is not allowed now; the message says why in plain words."""


@dataclass
class Proposal:
    id: int
    title: str
    state: str
    creator_username: str
    creator_name: str
    creator_role: str
    countries: list[str]
    request: str
    thread: str | None
    option_name: str
    option_label: str
    limits: dict[str, Any]
    snapshot: dict[str, Any]
    verified: bool
    needs_finance: bool
    finance_reasons: list[str]
    data_version: str
    snapshot_at: str
    created_at: str
    updated_at: str
    approved_by: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Proposal:
        return cls(
            row["id"],
            row["title"],
            row["state"],
            row["creator_username"],
            row["creator_name"],
            row["creator_role"],
            [c for c in row["countries"].split(",") if c],
            row["request"],
            row["thread"],
            row["option_name"],
            row["option_label"],
            json.loads(row["limits"]),
            json.loads(row["snapshot"]),
            bool(row["verified"]),
            bool(row["needs_finance"]),
            json.loads(row["finance_reasons"]),
            row["data_version"],
            row["snapshot_at"],
            row["created_at"],
            row["updated_at"],
            row["approved_by"],
        )

    @property
    def totals(self) -> dict[str, float]:
        return self.snapshot.get("chosen_totals", {})

    @property
    def baseline(self) -> dict[str, float]:
        return self.snapshot.get("baseline_totals", {})


# What a crew run becomes when it is saved as a proposal.


BASELINE_KEYS = ("net_value", "total_cost", "uplift_value", "served", "arrivals", "dropped", "hires", "notices", "start_agents", "end_agents", "mean_wait_days", "max_wait_days")


def baseline_totals(results: dict[str, dict[str, Any]], countries: list[str]) -> dict[str, float]:
    """The plan the year started with, summed over the countries, like an option card's totals."""
    rows = [results[c]["baseline"] for c in countries if results.get(c, {}).get("baseline")]
    if not rows:
        return {}
    total = {k: sum(r.get(k, 0.0) for r in rows) for k in BASELINE_KEYS if k not in {"mean_wait_days", "max_wait_days"}}
    total["service_rate"] = total["served"] / total["arrivals"] if total["arrivals"] else 1.0
    total["max_wait_days"] = max(r.get("max_wait_days", 0.0) for r in rows)
    served = sum(r.get("served", 0.0) for r in rows)
    total["mean_wait_days"] = sum(r.get("mean_wait_days", 0.0) * r.get("served", 0.0) for r in rows) / served if served else 0.0
    return total


def chosen_schedule(result: dict[str, Any], name: str) -> dict[str, list[float]]:
    by_name = {o["name"]: o for o in result["options"]}
    for pick in (name, "within_limits", "recommended"):
        if pick in by_name and by_name[pick].get("schedule"):
            return by_name[pick]["schedule"]
    return {}


def snapshot_from_run(values: dict[str, Any], option: str | None = None) -> dict[str, Any]:
    """Everything a proposal keeps from a run: the cards, the choice, the baseline and the schedules."""
    cards = values.get("options") or []
    if not cards:
        raise WorkflowError("This run has no options yet; ask a question first.")
    name = option or values.get("chosen") or cards[0]["name"]
    card = next((c for c in cards if c["name"] == name), None)
    if card is None:
        raise WorkflowError(f"No option called {name!r} in this run.")
    countries = values["plan"]["countries"]
    results = values.get("country_results") or {}
    return {
        "countries": countries,
        "cards": cards,
        "chosen": name,
        "chosen_label": card["label"],
        "chosen_totals": card["totals"],
        "baseline_totals": baseline_totals(results, countries),
        "schedules": {c: chosen_schedule(results[c], name) for c in countries if c in results},
        "baseline_schedules": {c: results[c].get("baseline_schedule", {}) for c in countries if c in results},
        "answer": values.get("answer", ""),
        "unverified": list(values.get("unverified") or []),
        "flagged": list(values.get("flagged") or []),
        "limits": values.get("user_constraints") or {},
    }


def finance_reasons(snapshot: dict[str, Any], settings: WorkflowSettings) -> list[str]:
    """Why Finance has to see a proposal first; empty when it can go straight to approval."""
    reasons = []
    chosen, base = snapshot["chosen_totals"], snapshot["baseline_totals"]
    if base:
        change = chosen["total_cost"] - base["total_cost"]
        share = abs(change) / base["total_cost"] if base["total_cost"] else 0.0
        if abs(change) > settings.finance_cost_usd or share > settings.finance_cost_share:
            reasons.append(f"staffing cost changes by ${change:+,.0f} ({share:.1%}) against the plan")
    for country, months in snapshot["schedules"].items():
        before = sum(snapshot["baseline_schedules"].get(country, {}).get("hires", []))
        after = sum(months.get("hires", []))
        if abs(after - before) > settings.finance_hires_per_country:
            reasons.append(f"{country}: hires change by {after - before:+.0f} against the plan")
    if snapshot.get("limits", {}).get("budget_usd") is not None:
        reasons.append("the request sets a budget")
    return reasons


# Deadlines from lead times.

MONTH_NAMES = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December")


@dataclass
class Deadline:
    country: str
    month: int  # the month the hires are decided, 1 to 12
    hires: float
    start_month: int  # the month the new agents start
    deadline: date
    status: str  # On track, Due soon, Overdue, Done

    def as_row(self) -> dict[str, Any]:
        return {
            "Country": self.country,
            "Hires": f"{self.hires:.0f}",
            "Needed from": f"{MONTH_NAMES[self.start_month - 1]}" if self.start_month <= 12 else "next January",
            "Decide by": self.deadline.isoformat(),
            "Status": self.status,
        }


def hiring_deadlines(
    schedules: dict[str, dict[str, list[float]]],
    year: int,
    lead_months: int,
    today: date,
    due_soon_days: int = 7,
    done: bool = False,
) -> list[Deadline]:
    """For each month with hires, the date by which they must be decided so the agents start in time.

    Agents needed from the start of month M must be hired `lead_months` earlier, so the
    deadline is the first day of month M minus the lead time.
    """
    found = []
    for country, months in schedules.items():
        for index, hires in enumerate(months.get("hires", []), start=1):
            if not hires:
                continue
            start = index + lead_months
            deadline = date(year, index, 1)
            if done:
                status = "Done"
            elif deadline < today:
                status = "Overdue"
            elif (deadline - today).days <= due_soon_days:
                status = "Due soon"
            else:
                status = "On track"
            found.append(Deadline(country, index, hires, start, deadline, status))
    return sorted(found, key=lambda d: (d.deadline, d.country))


# The store.


class WorkflowStore:
    def __init__(self, path: str | Path, settings: WorkflowSettings | None = None, clock: Callable[[], datetime] | None = None) -> None:
        self.path = Path(path)
        self.settings = settings or WorkflowSettings()
        self.clock = clock or (lambda: datetime.now(UTC))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db, db:
            for statement in SCHEMA:
                db.execute(statement)

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        return db

    def _now(self) -> str:
        return self.clock().isoformat()

    # Proposals.

    def create(self, user: User, values: dict[str, Any], note: str, data_version: str, option: str | None = None, title: str | None = None) -> Proposal:
        from staffpilot_crew.auth import can_propose

        snapshot = snapshot_from_run(values, option)
        countries = snapshot["countries"]
        if not can_propose(user, countries):
            if user.role == "country_manager":
                raise WorkflowError(f"A country manager can propose changes for {user.assigned_country} only.")
            raise WorkflowError(f"{user.role_label} cannot create proposals; a country or staffing manager can.")
        note = require_note(note)
        reasons = finance_reasons(snapshot, self.settings)
        verified = not snapshot["unverified"] and not snapshot["flagged"]
        now = self._now()
        title = title or f"{', '.join(countries)}: {snapshot['chosen_label']}"
        with closing(self._connect()) as db, db:
            cursor = db.execute(
                "INSERT INTO proposals (title, state, creator_username, creator_name, creator_role, countries, request, thread, option_name, option_label, "
                "limits, snapshot, verified, needs_finance, finance_reasons, data_version, snapshot_at, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    title,
                    DRAFT,
                    user.username,
                    user.full_name,
                    user.role,
                    ",".join(countries),
                    values.get("request", ""),
                    values.get("thread"),
                    snapshot["chosen"],
                    snapshot["chosen_label"],
                    json.dumps(snapshot["limits"]),
                    json.dumps(snapshot),
                    int(verified),
                    int(bool(reasons)),
                    json.dumps(reasons),
                    data_version,
                    now,
                    now,
                    now,
                ),
            )
            proposal_id = cursor.lastrowid
            self._event(db, proposal_id, user, "create", None, DRAFT, note, now)
        return self.get(proposal_id)

    def get(self, proposal_id: int) -> Proposal:
        with closing(self._connect()) as db:
            row = db.execute("SELECT * FROM proposals WHERE id = ?", (proposal_id,)).fetchone()
            if row is None:
                raise WorkflowError(f"No proposal number {proposal_id}.")
            proposal = Proposal.from_row(row)
            proposal.events = [dict(e) for e in db.execute("SELECT * FROM proposal_events WHERE proposal_id = ? ORDER BY id", (proposal_id,))]
        return proposal

    def list(self, states: tuple[str, ...] | None = None, countries: list[str] | None = None) -> list[Proposal]:
        with closing(self._connect()) as db:
            rows = db.execute("SELECT * FROM proposals ORDER BY updated_at DESC, id DESC").fetchall()
        found = [Proposal.from_row(r) for r in rows]
        if states:
            found = [p for p in found if p.state in states]
        if countries:
            found = [p for p in found if set(p.countries) & set(countries)]
        return found

    def allowed(self, user: User | None, proposal: Proposal, action: str) -> str | None:
        """None when `user` may do `action` now, otherwise the reason they may not."""
        if user is None:
            return "Sign in first."
        rule = ACTIONS[action]
        is_creator = user.username == proposal.creator_username
        if user.role not in rule.roles and not (rule.creator_may and is_creator):
            if action == "approve":
                return "Final approval requires Staffing Manager role."
            return f"{user.role_label} cannot do this; it is for {', '.join(r.replace('_', ' ') for r in rule.roles if r != 'admin') or 'the creator'}."
        if proposal.state not in rule.from_states:
            return f"A proposal that is {proposal.state} cannot be changed this way."
        if action == "approve" and self.stale_reasons(proposal):
            return "The numbers are out of date; refresh them before approving."
        return None

    def act(self, user: User, proposal_id: int, action: str, note: str, approved_by: str | None = None) -> Proposal:
        proposal = self.get(proposal_id)
        reason = self.allowed(user, proposal, action)
        if reason:
            raise WorkflowError(reason)
        note = require_note(note)
        rule = ACTIONS[action]
        to_state = rule.to_state
        if action == "submit":
            to_state = PENDING_FINANCE if proposal.needs_finance else PENDING_APPROVAL
        if action == "refresh":
            raise WorkflowError("Use refresh() with the new numbers.")
        now = self._now()
        with closing(self._connect()) as db, db:
            db.execute(
                "UPDATE proposals SET state = ?, updated_at = ?, approved_by = COALESCE(?, approved_by) WHERE id = ?",
                (to_state, now, approved_by if action == "approve" else None, proposal_id),
            )
            self._event(db, proposal_id, user, action, proposal.state, to_state, note, now)
        return self.get(proposal_id)

    def refresh(self, user: User, proposal_id: int, values: dict[str, Any], note: str, data_version: str, action: str = "refresh") -> Proposal:
        """New numbers for an open proposal (a refresh), or a returned one revised into a draft."""
        proposal = self.get(proposal_id)
        reason = self.allowed(user, proposal, action)
        if reason:
            raise WorkflowError(reason)
        note = require_note(note)
        snapshot = snapshot_from_run(values, values.get("chosen"))
        reasons = finance_reasons(snapshot, self.settings)
        verified = not snapshot["unverified"] and not snapshot["flagged"]
        to_state = DRAFT if action == "revise" else proposal.state
        # New numbers that now need Finance go back to Finance, even when they were past it.
        if to_state == PENDING_APPROVAL and reasons and not proposal.needs_finance:
            to_state = PENDING_FINANCE
        now = self._now()
        with closing(self._connect()) as db, db:
            db.execute(
                "UPDATE proposals SET state = ?, thread = ?, option_name = ?, option_label = ?, limits = ?, snapshot = ?, verified = ?, "
                "needs_finance = ?, finance_reasons = ?, data_version = ?, snapshot_at = ?, updated_at = ? WHERE id = ?",
                (
                    to_state,
                    values.get("thread", proposal.thread),
                    snapshot["chosen"],
                    snapshot["chosen_label"],
                    json.dumps(snapshot["limits"]),
                    json.dumps(snapshot),
                    int(verified),
                    int(bool(reasons)),
                    json.dumps(reasons),
                    data_version,
                    now,
                    now,
                    proposal_id,
                ),
            )
            self._event(db, proposal_id, user, action, proposal.state, to_state, note, now)
        return self.get(proposal_id)

    def _event(self, db: sqlite3.Connection, proposal_id: int, user: User, action: str, from_state: str | None, to_state: str, note: str, now: str) -> None:
        db.execute(
            "INSERT INTO proposal_events (proposal_id, timestamp, actor_username, actor_name, actor_role, action, from_state, to_state, note) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (proposal_id, now, user.username, user.full_name, user.role, action, from_state, to_state, note),
        )

    # Stale numbers.

    def mark_data_changed(self, country: str, reason: str) -> None:
        with closing(self._connect()) as db, db:
            db.execute(
                "INSERT INTO data_versions (country, updated_at, reason) VALUES (?, ?, ?) "
                "ON CONFLICT(country) DO UPDATE SET updated_at = excluded.updated_at, reason = excluded.reason",
                (country, self._now(), reason),
            )

    def data_changes(self) -> dict[str, tuple[str, str]]:
        with closing(self._connect()) as db:
            return {r["country"]: (r["updated_at"], r["reason"]) for r in db.execute("SELECT * FROM data_versions")}

    def stale_reasons(self, proposal: Proposal, data_version: str | None = None) -> list[str]:
        """Why a proposal's numbers may no longer hold; empty when they are current."""
        if proposal.state not in OPEN_STATES:
            return []
        made = datetime.fromisoformat(proposal.snapshot_at)
        changes = self.data_changes()
        reasons = [
            f"{country} has newer data ({changes[country][1]})"
            for country in proposal.countries
            if country in changes and datetime.fromisoformat(changes[country][0]) > made
        ]
        if data_version is not None and data_version != proposal.data_version:
            reasons.append("the dataset changed after these numbers were made")
        age = (self.clock() - made).days
        if age > self.settings.stale_days:
            reasons.append(f"the numbers are {age} days old (the limit is {self.settings.stale_days})")
        return reasons

    # Saved scenarios.

    def save_scenario(self, user: User, name: str, values: dict[str, Any], mode: str, audience: str) -> int:
        name = (name or "").strip()
        if not name:
            raise WorkflowError("Give the scenario a name.")
        countries = (values.get("plan") or {}).get("countries", [])
        now = self._now()
        with closing(self._connect()) as db, db:
            cursor = db.execute(
                "INSERT INTO saved_scenarios (name, owner_username, countries, request, mode, audience, constraints, thread, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (name, user.username, ",".join(countries), values.get("request", ""), mode, audience, json.dumps(values.get("user_constraints") or {}), values.get("thread"), now, now),
            )
            return cursor.lastrowid

    def scenarios(self, user: User) -> list[dict[str, Any]]:
        """The user's own scenarios, newest first; the admin sees everyone's."""
        with closing(self._connect()) as db:
            if user.role == "admin":
                rows = db.execute("SELECT * FROM saved_scenarios ORDER BY updated_at DESC, id DESC").fetchall()
            else:
                rows = db.execute("SELECT * FROM saved_scenarios WHERE owner_username = ? ORDER BY updated_at DESC, id DESC", (user.username,)).fetchall()
        return [{**dict(r), "constraints": json.loads(r["constraints"]), "countries": [c for c in r["countries"].split(",") if c]} for r in rows]

    def scenario(self, user: User, scenario_id: int) -> dict[str, Any]:
        found = next((s for s in self.scenarios(user) if s["id"] == scenario_id), None)
        if found is None:
            raise WorkflowError("That scenario is not one of yours.")
        return found

    def rename_scenario(self, user: User, scenario_id: int, name: str) -> None:
        self.scenario(user, scenario_id)
        name = (name or "").strip()
        if not name:
            raise WorkflowError("Give the scenario a name.")
        with closing(self._connect()) as db, db:
            db.execute("UPDATE saved_scenarios SET name = ?, updated_at = ? WHERE id = ?", (name, self._now(), scenario_id))

    def duplicate_scenario(self, user: User, scenario_id: int, name: str | None = None) -> int:
        """A copy to change freely; the original stays as it is. The copy runs afresh when reopened."""
        original = self.scenario(user, scenario_id)
        now = self._now()
        with closing(self._connect()) as db, db:
            cursor = db.execute(
                "INSERT INTO saved_scenarios (name, owner_username, countries, request, mode, audience, constraints, thread, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)",
                (
                    (name or "").strip() or f"{original['name']} (copy)",
                    user.username,
                    ",".join(original["countries"]),
                    original["request"],
                    original["mode"],
                    original["audience"],
                    json.dumps(original["constraints"]),
                    now,
                    now,
                ),
            )
            return cursor.lastrowid

    def set_scenario_thread(self, scenario_id: int, thread: str) -> None:
        with closing(self._connect()) as db, db:
            db.execute("UPDATE saved_scenarios SET thread = ?, updated_at = ? WHERE id = ?", (thread, self._now(), scenario_id))


def require_note(note: str | None) -> str:
    note = (note or "").strip()
    if not note:
        raise WorkflowError("Write a short note: it goes on the proposal's timeline for the next person.")
    return note


# The hiring plan as a CSV for an HR system.

EXPORT_COLUMNS = ("country", "month", "current_headcount", "recommended_hires", "delta", "target_deadline", "approval_status", "approved_by")


def hiring_rows(proposal: Proposal, year: int, lead_months: int) -> list[dict[str, Any]]:
    """One row per country and month: the agents the plan started with that month, the hires
    this proposal decides, the change against the plan, and the date to decide them by."""
    rows = []
    for country in proposal.countries:
        months = proposal.snapshot.get("schedules", {}).get(country, {})
        base = proposal.snapshot.get("baseline_schedules", {}).get(country, {})
        hires = months.get("hires", [0.0] * 12)
        base_hires = base.get("hires", [0.0] * 12)
        active = base.get("active", [0.0] * 12)
        rows += [
            {
                "country": country,
                "month": f"{year}-{index + 1:02d}",
                "current_headcount": int(active[index]) if index < len(active) else "",
                "recommended_hires": int(hires[index]),
                "delta": int(hires[index] - base_hires[index]),
                "target_deadline": date(year, index + 1, 1).isoformat(),
                "approval_status": proposal.state,
                "approved_by": proposal.approved_by or "",
            }
            for index in range(12)
            if hires[index] or base_hires[index]
        ]
    return rows
