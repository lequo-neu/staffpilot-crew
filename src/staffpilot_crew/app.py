"""The StaffPilot Crew web app: a daily desk for each role.

Pages, in the order a user meets them:

* **Today**: what waits on me, the alerts for my countries and the hiring deadlines.
* **Proposals**: every proposal with its state, numbers, timeline and the next actions.
* **Workspace**: the options of the latest question, ready to become a proposal.
* **Saved scenarios**, **Monthly check**, **Decisions** and the **Full planner**.
* **Answer quality** and **Users** for the admin and reviewers only.

Questions to the crew go through the sidebar on the right, so the numbers stay in view.
The hackathon's own pages are served unchanged at /planner, built by
``staffpilot.app.build_app`` on the same planner.
"""

from __future__ import annotations

from collections.abc import Iterator
import csv
import html
import inspect
from pathlib import Path
import tempfile
from typing import Any

import gradio as gr
import pandas as pd
from staffpilot.agent.facts import FactLedger

from staffpilot_crew import agents, evaluation
from staffpilot_crew.auth import (
    GUEST,
    LOCAL_USER,
    ROLE_LABELS,
    AuthStore,
    PasswordError,
    User,
    auth_mode,
    can_propose,
    password_warning,
    sees_diagnostics,
    starter_prompts,
)
from staffpilot_crew.constraints import AUDIENCE_LABELS
from staffpilot_crew.desk import SCOPES, ProposalDesk, today
from staffpilot_crew.graph import CrewRunner
from staffpilot_crew.monitoring import MONTHS, review_monthly_update
from staffpilot_crew.router import MODE_LABELS
from staffpilot_crew.services import CrewServices
from staffpilot_crew.workflow import (
    ACTIONS,
    EXPORT_COLUMNS,
    STATES,
    Proposal,
    WorkflowError,
    WorkflowStore,
    baseline_totals,
    hiring_rows,
)


MODE_BY_LABEL = {label: mode for mode, label in MODE_LABELS.items()}
AUDIENCE_BY_LABEL = {label: name for name, label in AUDIENCE_LABELS.items()}
TRACE_COLUMNS = ["Agent", "Action", "Model", "Why this model", "Detail", "Seconds"]
PROPOSAL_ACTIONS = ("submit", "confirm_budget", "approve", "return", "start_hiring", "mark_done", "withdraw", "refresh")
COPY_FORMATS = ("Country memo", "Finance note", "Hiring notice email", "Staffing summary")
EXAMPLES = {
    "sample": [
        "What is the plan for Alpha?",
        "Compare a growth push in Alpha, Beta and Gamma where sign ups rise 20%",
        "Beta: serve at least 98% with no more than 1 hire per month",
        "What if an AI assistant onboards 30% of advertisers in Alpha?",
        "Write a memo for the Gamma country manager",
    ],
    "course": [
        "Sign ups in the USA came in 20% above forecast in March. What should we do, and what does it cost?",
        "Compare a Q3 growth push in the USA, Brazil and the UAE. Where does it pay?",
        "Keep hiring in Brazil to at most 5 people a month and serve at least 98%",
        "What if an AI assistant onboards 30% of small advertisers in Brazil?",
        "Write a short note to the Vietnam country manager that explains why they are hiring.",
    ],
}
CSS = """
.crew-cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 12px; }
.crew-card { border: 1px solid var(--border-color-primary); border-radius: 8px; padding: 12px 14px; background: var(--background-fill-primary); }
.crew-card.chosen { border: 2px solid var(--color-accent); }
.crew-card h4 { margin: 0 0 6px 0; font-size: 15px; }
.crew-card dl { display: grid; grid-template-columns: auto auto; gap: 2px 10px; margin: 0; font-size: 13px; }
.crew-card dt { color: var(--body-text-color-subdued); }
.crew-card dd { margin: 0; text-align: right; font-variant-numeric: tabular-nums; }
.crew-card .delta { color: var(--body-text-color-subdued); font-size: 12px; margin-left: 4px; }
.crew-tag { display: inline-block; font-size: 12px; padding: 1px 8px; border-radius: 10px; margin: 0 4px 6px 0; border: 1px solid var(--border-color-primary); }
.crew-verified { display: inline-block; font-size: 12px; padding: 2px 10px; border-radius: 10px; margin-bottom: 8px; }
.crew-verified.ok { background: #ecfdf3; color: #05603a; border: 1px solid #6ce9a6; }
.crew-verified.review { background: #fffaeb; color: #93370d; border: 1px solid #fec84b; }
.crew-issue { color: var(--error-text-color, #b42318); font-size: 12px; margin-top: 6px; }
.crew-moved, .crew-same { font-size: 12px; color: var(--body-text-color-subdued); margin: 4px 0; }
.crew-status { font-size: 14px; font-weight: 600; margin: 4px 0; }
.crew-banner { border: 1px solid #b54708; background: #fffaeb; color: #7a2e0e; border-radius: 8px; padding: 10px 14px; font-size: 14px; }
.crew-stale { border: 1px solid #b42318; background: #fef3f2; color: #7a271a; border-radius: 8px; padding: 10px 14px; font-size: 14px; margin: 6px 0; }
.crew-login { max-width: 440px; }
.crew-top { align-items: flex-start; }
.crew-account { text-align: right; }
.crew-account-buttons { justify-content: flex-end; gap: 8px; flex-wrap: nowrap; }
.crew-account-buttons button { flex: 0 0 auto !important; width: auto !important; min-width: max-content !important; white-space: nowrap; padding: 4px 12px; }
.crew-quiet, .crew-quiet p { color: var(--body-text-color-subdued); font-size: 13px; }
.crew-question { color: var(--body-text-color-subdued); font-size: 13px; }
.crew-answer { line-height: 1.5; }
.crew-timeline { list-style: none; margin: 0; padding: 0; font-size: 13px; }
.crew-timeline li { border-left: 2px solid var(--border-color-primary); padding: 2px 0 10px 12px; }
.crew-timeline .when { color: var(--body-text-color-subdued); font-size: 12px; }
.crew-alerts { margin: 0; padding-left: 18px; font-size: 13px; }
.crew-alerts li { margin-bottom: 6px; }
.crew-state { display: inline-block; font-size: 12px; padding: 2px 10px; border-radius: 10px; border: 1px solid var(--border-color-primary); margin-left: 6px; }
.crew-trace { margin: 0; padding-left: 20px; font-size: 12px; }
.crew-trace li { margin-bottom: 8px; }
.crew-trace .secs { float: right; color: var(--body-text-color-subdued); }
.crew-trace .meta { color: var(--body-text-color-subdued); }
.crew-trace.empty { color: var(--body-text-color-subdued); }
"""


class UserInputError(Exception):
    """Something the user can fix (an empty field, a missing note); shown as a short notice."""


def calm(fn, count: int):
    """Show a UserInputError or WorkflowError as a notice and leave every output as it was."""
    skips = tuple(gr.skip() for _ in range(count)) if count > 1 else gr.skip()
    expected = (UserInputError, WorkflowError, PasswordError, PermissionError)
    if inspect.isgeneratorfunction(fn):

        def stream(*args):
            try:
                yield from fn(*args)
            except expected as problem:
                gr.Warning(str(problem))
                yield skips

        return stream

    def plain(*args):
        try:
            return fn(*args)
        except expected as problem:
            gr.Warning(str(problem))
            return skips

    return plain


# Numbers and their change against the plan the year started with.


def money(value: float) -> str:
    sign = "-" if value < 0 else ""
    value = abs(value)
    return f"{sign}${value / 1e6:,.1f}M" if value >= 1e6 else f"{sign}${value:,.0f}"


def money_change(value: float) -> str:
    if abs(value) < 0.5:
        return "no change"
    sign = "+" if value > 0 else "-"
    value = abs(value)
    return f"{sign}${value / 1e6:,.1f}M" if value >= 1e6 else f"{sign}${value / 1e3:,.0f}K" if value >= 1e3 else f"{sign}${value:,.0f}"


def count_change(value: float, unit: str = "") -> str:
    if abs(value) < 0.05:
        return "no change"
    return f"{value:+,.0f}{unit}" if abs(value) >= 1 else f"{value:+.1f}{unit}"


def cards_html(cards: list[dict[str, Any]], chosen: str | None, baseline: dict[str, float] | None = None, verified: bool | None = None) -> str:
    """Option cards with the change against the plan the year started with."""
    if not cards:
        return ""
    base = baseline or {}
    badge = ""
    if verified is True:
        badge = "<div class='crew-verified ok'>Verified numbers: every figure matches the planner</div>"
    elif verified is False:
        badge = "<div class='crew-verified review'>Unverified: some figures could not be matched; review before deciding</div>"
    blocks = []
    for card in cards:
        t = card["totals"]

        def row(label: str, shown: str, change: str = "") -> str:
            delta = f"<span class='delta'>({change})</span>" if change else ""
            return f"<dt>{html.escape(label)}</dt><dd>{html.escape(shown)}{delta}</dd>"

        def diff(key: str, kind: str = "count", t: dict[str, float] = t) -> str:
            if key not in base:
                return ""
            value = t[key] - base[key]
            if kind == "money":
                return money_change(value)
            if kind == "rate":
                return "no change" if abs(value) < 0.0005 else f"{value * 100:+.1f} pt"
            if kind == "days":
                return "no change" if abs(value) < 0.05 else f"{value:+.1f} days"
            return count_change(value)

        rows = [
            row("Net value", money(t["net_value"]), diff("net_value", "money")),
            row("Staffing cost", money(t["total_cost"]), diff("total_cost", "money")),
            row("Advertisers served", f"{t['service_rate']:.1%}", diff("service_rate", "rate")),
            row("Average wait", f"{t['mean_wait_days']:.1f} days", diff("mean_wait_days", "days")),
            row("Longest wait", f"{t['max_wait_days']:.0f} days", diff("max_wait_days", "days")),
            row("Team at year end", f"{t['end_agents']:.0f}", diff("end_agents")),
            row("Hires", f"{t['hires']:.0f}", diff("hires")),
            row("Notices", f"{t['notices']:.0f}", diff("notices")),
        ]
        if card.get("automation"):
            a = card["automation"]
            rows.append(row("Net value with the AI share", f"{money(a['net_value_low'])} to {money(a['net_value_high'])}"))
        tags = []
        if card["name"] == chosen:
            tags.append("Best match for your request")
        if card["relaxed"]:
            tags.append("Limits moved")
        moved = "".join(f"<div class='crew-moved'>{html.escape(m)}</div>" for m in card.get("relaxations", []))
        same = f"<div class='crew-same'>Same plan as: {html.escape(', '.join(card['same_as']))}</div>" if card.get("same_as") else ""
        issues = "".join(f"<div class='crew-issue'>{html.escape(i)}</div>" for i in card.get("issues", []))
        tag_html = "".join(f"<span class='crew-tag'>{html.escape(x)}</span>" for x in tags)
        cls = " chosen" if card["name"] == chosen else ""
        blocks.append(f"<div class='crew-card{cls}'><h4>{html.escape(card['label'])}</h4>{tag_html}{moved}<dl>{''.join(rows)}</dl>{same}{issues}</div>")
    note = "<div class='crew-quiet'>In brackets: the change against the plan the year started with.</div>" if base else ""
    return f"{badge}<div class='crew-cards'>{''.join(blocks)}</div>{note}"


def run_cards(values: dict[str, Any]) -> str:
    cards = values.get("options") or []
    if not cards:
        return ""
    countries = (values.get("plan") or {}).get("countries", [])
    base = baseline_totals(values.get("country_results") or {}, countries)
    verified = not values.get("unverified") and not values.get("flagged")
    return cards_html(cards, values.get("chosen"), base, verified)


def trace_html(values: dict[str, Any]) -> str:
    """The steps behind an answer: who did what, and which model wrote it when one was used."""
    events = values.get("trace", [])
    if not events:
        return "<div class='crew-trace empty'>The steps behind each answer appear here.</div>"
    items = []
    for e in events:
        used_model = e.get("model_name") not in (None, "", "offline")
        model = f"<div class='meta'>Written by {html.escape(e['model'])}: {html.escape(e.get('reason', ''))}</div>" if used_model else ""
        items.append(
            f"<li><b>{html.escape(e.get('agent', ''))}</b> {html.escape(e.get('action', ''))}"
            f"<span class='secs'>{e.get('seconds', 0.0):.2f} s</span>"
            f"<div>{html.escape(e.get('detail', ''))}</div>{model}</li>"
        )
    return f"<ol class='crew-trace'>{''.join(items)}</ol>"


def trace_frame(values: dict[str, Any]) -> pd.DataFrame:
    rows = [[e.get("agent", ""), e.get("action", ""), e.get("model", ""), e.get("reason", ""), e.get("detail", ""), e.get("seconds", 0.0)] for e in values.get("trace", [])]
    return pd.DataFrame(rows, columns=TRACE_COLUMNS)


STATUS_TEXT = {
    "Waiting for approval": "Options ready: see Workspace to turn one into a proposal.",
    "Re-optimizing": "Planning again with your new limits.",
    "Approved": "Approved and saved under Decisions. Nothing was changed in any other system.",
    "Needs details": "One more detail is needed; see the question below.",
    "Planning": "Working on it.",
}


def status_text(values: dict[str, Any], user: User | None = None) -> str:
    text = STATUS_TEXT.get(values.get("approval_status") or "", "")
    if values.get("approval_status") == "Waiting for approval" and user is not None and not can_propose(user, (values.get("plan") or {}).get("countries")):
        text = "Options ready on Workspace. Proposals for these countries come from a country or staffing manager."
    return f"<div class='crew-status'>{html.escape(text)}</div>" if text else ""


def cost_text(runner: CrewRunner, threads: list[str]) -> str:
    """The cost of the model calls in this session; empty when no model was called."""
    calls = [c for t in threads for c in runner.values(t).get("calls", [])]
    model_calls = [c for c in calls if c.get("model") != "offline"]
    if not model_calls:
        return ""
    measured = sum(c["cost_usd"] for c in model_calls if c.get("cost_source") == "measured")
    unpriced = sum(1 for c in model_calls if c.get("cost_source") in {"unknown price", "no token counts"})
    tokens = sum(c.get("tokens_in", 0) + c.get("tokens_out", 0) for c in model_calls)
    text = f"Model cost this session: ${measured:,.4f} for {tokens:,} tokens in {len(model_calls)} calls."
    if unpriced:
        text += f" {unpriced} calls have no known price, so the real cost is higher."
    return text


def answer_markdown(values: dict[str, Any], request: str) -> str:
    answer = values.get("answer") or ""
    if not request and not answer:
        return ""
    return f"<div class='crew-question'>You asked: {html.escape(request)}</div>\n\n{answer}"


def role_view(values: dict[str, Any], audience: str) -> str:
    """The same decision written for another reader, from the ledger, without a model call."""
    if not values.get("options"):
        return values.get("answer", "")
    draft = agents.template_draft({**values, "audience": audience})
    text, _ = FactLedger(dict(values.get("fact_ledger") or {})).render(draft)
    return text


def copy_text(values: dict[str, Any], kind: str, user: User | None) -> str:
    """Plain text ready to paste: a memo, a finance note, a hiring notice email or a summary."""
    if not values.get("options"):
        return ""
    audience = {"Country memo": "country_manager", "Finance note": "finance", "Hiring notice email": "people", "Staffing summary": "manager"}[kind]
    body = role_view(values, audience).replace("**", "")
    if kind != "Hiring notice email":
        return body
    countries = ", ".join((values.get("plan") or {}).get("countries", []))
    sender = user.full_name if user else "StaffPilot"
    return f"Subject: Hiring and notice plan for {countries}\n\nHello,\n\n{body}\n\nBest regards,\n{sender}"


def user_bar(user: User | None) -> str:
    if user is None:
        return ""
    country = user.assigned_country or "All countries"
    return f"**{html.escape(user.full_name)}** | {html.escape(user.role_label)} | {html.escape(country)}"


def banner_html(user: User | None) -> str:
    text = password_warning(user)
    if not text:
        return ""
    title = "Change the default password." if user and user.weak_password else "Password older than 3 months."
    return f"<div class='crew-banner' role='alert'><b>{title}</b> {html.escape(text)}</div>"


def users_frame(store: AuthStore | None) -> pd.DataFrame:
    """Every account, for the admin's Users tab. Hashes and salts are never shown."""
    columns = ["Username", "Full name", "Role", "Country", "Password age (days)", "Over 90 days"]
    if store is None:
        return pd.DataFrame(columns=columns)
    rows = [[u.username, u.full_name, u.role_label, u.assigned_country or "all", u.password_age_days(), "yes" if u.password_expired() else "no"] for u in store.users()]
    return pd.DataFrame(rows, columns=columns)


# Proposals on the page.

PROPOSAL_COLUMNS = ["#", "Proposal", "State", "Countries", "Net value change", "Hires change", "Needs", "Updated"]


def proposal_rows(desk: ProposalDesk, proposals: list[Proposal]) -> pd.DataFrame:
    rows = []
    for p in proposals:
        t, b = p.totals, p.baseline
        stale = desk.stale(p)
        needs = "Refresh: numbers out of date" if stale else ("Finance review" if p.state == "Pending Finance" else "")
        rows.append(
            [
                p.id,
                p.title,
                p.state,
                ", ".join(p.countries),
                money_change(t.get("net_value", 0) - b.get("net_value", 0)) if b else "",
                count_change(t.get("hires", 0) - b.get("hires", 0)) if b else "",
                needs,
                p.updated_at[:16].replace("T", " "),
            ]
        )
    return pd.DataFrame(rows, columns=PROPOSAL_COLUMNS)


def timeline_html(proposal: Proposal) -> str:
    items = []
    for e in proposal.events:
        change = f"{e['from_state']} to {e['to_state']}" if e["from_state"] and e["from_state"] != e["to_state"] else e["to_state"]
        items.append(
            f"<li><div class='when'>{html.escape(e['timestamp'][:16].replace('T', ' '))} UTC</div>"
            f"<b>{html.escape(e['actor_name'])}</b> ({html.escape(ROLE_LABELS.get(e['actor_role'], e['actor_role']))}): "
            f"{html.escape(ACTIONS[e['action']].label if e['action'] in ACTIONS else 'Created')}, {html.escape(change)}"
            f"<div>{html.escape(e['note'])}</div></li>"
        )
    return f"<ul class='crew-timeline'>{''.join(items)}</ul>"


def proposal_header(desk: ProposalDesk, proposal: Proposal) -> str:
    parts = [
        f"### #{proposal.id} {html.escape(proposal.title)} <span class='crew-state'>{html.escape(proposal.state)}</span>",
        (f"<div class='crew-quiet'>Created by {html.escape(proposal.creator_name)} ({html.escape(ROLE_LABELS.get(proposal.creator_role, proposal.creator_role))}). "
        f"Request: {html.escape(proposal.request)}</div>"),
    ]
    if proposal.finance_reasons:
        parts.append("<div class='crew-quiet'>Goes to Finance because " + html.escape("; ".join(proposal.finance_reasons)) + ".</div>")
    stale = desk.stale(proposal)
    if stale:
        parts.append(
            "<div class='crew-stale' role='alert'><b>Numbers out of date.</b> "
            + html.escape("; ".join(stale))
            + ". Approval is locked until you refresh the numbers.</div>"
        )
    return "\n\n".join(parts)


def export_csv(desk: ProposalDesk, proposal: Proposal) -> str:
    rows = hiring_rows(proposal, desk.planner.year, desk.lead_months())
    path = Path(tempfile.mkdtemp(prefix="staffpilot_")) / f"hiring_plan_{proposal.id}_{'_'.join(proposal.countries)}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXPORT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


def build_crew_app(
    services: CrewServices,
    runner: CrewRunner,
    data_note: str,
    examples: list[str],
    planner_pages: bool,
    store: AuthStore | None = None,
    mode_name: str | None = None,
    desk: ProposalDesk | None = None,
) -> gr.Blocks:
    countries = services.planner.countries
    mode_name = mode_name or auth_mode()
    if store is None and mode_name != "disabled":
        raise ValueError("Sign in needs an AuthStore; pass one or set STAFFPILOT_AUTH_MODE=disabled")
    desk = desk or ProposalDesk(WorkflowStore(services.crew.paths.app_db, services.crew.workflow), runner)
    starters_count = 4
    year = services.planner.year

    def who(current: dict | None) -> User | None:
        return User.from_dict(current)

    def need_user(current: dict | None) -> User:
        user = who(current)
        if user is None:
            raise UserInputError("Sign in first.")
        return user

    # Asking the crew (sidebar) and the Workspace.

    def run_outputs(values: dict[str, Any], request: str, threads: list[str], user: User | None):
        has_options = bool(values.get("options"))
        names = [(c["label"], c["name"]) for c in values.get("options") or []]
        may_propose = has_options and can_propose(user, (values.get("plan") or {}).get("countries"))
        return (
            answer_markdown(values, request),
            trace_html(values),
            status_text(values, user),
            run_cards(values),
            gr.update(choices=names, value=values.get("chosen") if names else None),
            gr.update(visible=has_options),
            gr.update(visible=may_propose),
            gr.update(visible=has_options and not may_propose),
            cost_text(runner, threads),
            copy_text(values, "Staffing summary", user) if has_options else "",
        )

    def stream_run(inputs: Any, thread: str, request: str, threads: list[str], user: User | None) -> Iterator[tuple]:
        values: dict[str, Any] = {"trace": []}
        for update in runner.stream(inputs, thread):
            for node, change in update.items():
                if node == "__interrupt__" or not isinstance(change, dict):
                    continue
                values["trace"] = values.get("trace", []) + change.get("trace", [])
                values["approval_status"] = change.get("approval_status", values.get("approval_status"))
                yield (*run_outputs(values, request, threads, user), thread, threads, gr.skip())
        final = runner.values(thread)
        go = gr.Tabs(selected="workspace") if final.get("options") else gr.skip()
        yield (*run_outputs(final, request, threads, user), thread, threads, go)

    def send(request: str, mode_label: str, audience_label: str, threads: list[str], current: dict | None):
        user = need_user(current)
        request = (request or "").strip()
        if not request:
            raise UserInputError("Type what you need to decide, or pick one of the common questions.")
        thread = runner.new_thread()
        threads = [*threads, thread]
        inputs = runner.initial(request, mode=MODE_BY_LABEL[mode_label], audience=AUDIENCE_BY_LABEL[audience_label], user=user.full_name, home_country=user.assigned_country)
        yield from stream_run(inputs, thread, request, threads, user)

    def change_limits(thread: str | None, text: str, threads: list[str], current: dict | None):
        user = need_user(current)
        if not thread:
            raise UserInputError("Ask a question first.")
        if not (text or "").strip():
            raise UserInputError("Write the new limit, for example: at most 5 hires a month.")
        values = runner.values(thread)
        if not values.get("waiting"):
            raise UserInputError("This run is closed; ask the question again.")
        inputs = runner.resume_inputs("change", thread, text=text.strip(), user=user.as_dict())
        yield from stream_run(inputs, thread, values.get("request", ""), threads, user)

    def switch_view(thread: str | None, audience_label: str):
        if not thread:
            return gr.skip()
        values = runner.values(thread)
        if not values.get("options"):
            return gr.skip()
        return answer_markdown({"answer": role_view(values, AUDIENCE_BY_LABEL[audience_label])}, values.get("request", ""))

    def show_copy(thread: str | None, kind: str, current: dict | None):
        if not thread:
            return ""
        return copy_text(runner.values(thread), kind, who(current))

    def load_run(thread: str | None, threads: list[str], current: dict | None):
        """Put an existing run (a monthly check, a reopened scenario) on Workspace and in the sidebar."""
        if not thread:
            return (gr.skip(),) * 13
        values = runner.values(thread)
        go = gr.Tabs(selected="workspace") if values.get("options") else gr.skip()
        return (*run_outputs(values, values.get("request", ""), threads, who(current)), thread, threads, go)

    # Proposals.

    def make_proposal(thread: str | None, option: str | None, note: str, submit: bool, current: dict | None):
        user = need_user(current)
        if not thread:
            raise UserInputError("Ask a question first.")
        proposal = desk.create(user, thread, note, option, submit=submit)
        gr.Info(f"Proposal #{proposal.id} is {proposal.state}.")
        return (proposal.id, "", gr.Tabs(selected="proposals"), *proposal_view(proposal.id, "", user), *today_outputs(user, "My scope", []))

    def proposal_view(proposal_id: int | None, note: str, user: User | None):
        """Header, cards, timeline, copy text and the action buttons of one proposal."""
        hidden = [gr.update(visible=False)] * len(PROPOSAL_ACTIONS)
        if not proposal_id:
            return ("Pick a proposal from the list.", "", "", gr.update(visible=False), "", *hidden)
        proposal = desk.store.get(int(proposal_id))
        snap = proposal.snapshot
        cards = cards_html(snap.get("cards", []), snap.get("chosen"), snap.get("baseline_totals"), proposal.verified)
        has_note = bool((note or "").strip())
        buttons = []
        for action in PROPOSAL_ACTIONS:
            reason = desk.allowed(user, proposal, action)
            role_may = reason is None or reason.startswith("The numbers are out of date")
            buttons.append(gr.update(visible=role_may, interactive=reason is None and has_note))
        return (
            proposal_header(desk, proposal),
            cards,
            timeline_html(proposal),
            gr.update(value=export_csv(desk, proposal), visible=True),
            snap.get("answer", ""),
            *buttons,
        )

    def act(action: str):
        def handler(proposal_id: int | None, note: str, current: dict | None, scope: str, states: list[str]):
            user = need_user(current)
            if not proposal_id:
                raise UserInputError("Pick a proposal first.")
            if action == "refresh":
                proposal = desk.refresh(user, int(proposal_id), note)
            else:
                proposal = desk.act(user, int(proposal_id), action, note)
            gr.Info(f"Proposal #{proposal.id} is now {proposal.state}.")
            return ("", *proposal_view(proposal.id, "", user), *today_outputs(user, scope, states), proposals_frame(user, scope, states), log_frame())

        return handler

    def revise(proposal_id: int | None, thread: str | None, note: str, current: dict | None):
        user = need_user(current)
        if not proposal_id:
            raise UserInputError("Open the proposal on Proposals first, then ask the changed question in the sidebar.")
        if not thread:
            raise UserInputError("Ask the changed question in the sidebar first.")
        proposal = desk.revise(user, int(proposal_id), thread, note)
        gr.Info(f"Proposal #{proposal.id} has the new numbers and is a Draft again.")
        return (proposal.id, "", gr.Tabs(selected="proposals"), *proposal_view(proposal.id, "", user))

    def proposals_frame(user: User | None, scope: str, states: list[str]):
        if user is None:
            return proposal_rows(desk, [])
        found = desk.store.list(tuple(states) if states else None)
        if scope == "My scope" and user.role == "country_manager":
            found = [p for p in found if user.assigned_country in p.countries]
        if scope == "Overdue or urgent":
            view = today(desk, user, scope)
            ids = {p.id for p in view.action_items}
            found = [p for p in found if p.id in ids]
        return proposal_rows(desk, found)

    def open_selected(frame: pd.DataFrame, current: dict | None, event: gr.SelectData):
        user = who(current)
        try:
            proposal_id = int(frame.iloc[event.index[0], 0])
        except (IndexError, ValueError, TypeError):
            return (gr.skip(),) * (3 + 5 + len(PROPOSAL_ACTIONS))
        return (proposal_id, gr.Tabs(selected="proposals"), "", *proposal_view(proposal_id, "", user))

    # Today.

    def today_outputs(user: User | None, scope: str, states: list[str]):
        if user is None:
            return (proposal_rows(desk, []), "", pd.DataFrame(), "")
        view = today(desk, user, scope, states=states or None)
        alerts = view.alerts
        changes = desk.store.data_changes()
        for country, (when, reason) in changes.items():
            if user.role != "country_manager" or user.assigned_country == country:
                alerts = [{"level": "info", "countries": [country], "message": f"New data for {country}: {reason} ({when[:10]})."}, *alerts]
        alert_html = (
            "<ul class='crew-alerts'>" + "".join(f"<li>{html.escape(a['message'])}</li>" for a in alerts[:8]) + "</ul>"
            if alerts
            else "<div class='crew-quiet'>No alerts for your countries.</div>"
        )
        deadlines = pd.DataFrame([d.as_row() for d in view.deadlines], columns=["Country", "Hires", "Needed from", "Decide by", "Status"])
        waiting = len(view.action_items)
        summary = f"**{waiting} {'proposal waits' if waiting == 1 else 'proposals wait'} on you.**" if scope == "My scope" else f"**{waiting} open proposals.**"
        if not waiting and scope == "My scope":
            summary = "**Nothing waits on you today.** Ask a question in the sidebar to start a new plan."
        return (proposal_rows(desk, view.action_items), alert_html, deadlines, summary)

    def today_refresh(current: dict | None, scope: str, states: list[str]):
        return today_outputs(who(current), scope, states)

    def look_into(country: str | None, current: dict | None):
        if not country:
            raise UserInputError("Pick a country first.")
        return f"What is the plan for {country}?"

    # Saved scenarios.

    def scenarios_frame(user: User | None) -> pd.DataFrame:
        columns = ["#", "Name", "Countries", "Request", "Updated"]
        if user is None:
            return pd.DataFrame(columns=columns)
        rows = [[s["id"], s["name"], ", ".join(s["countries"]), s["request"], s["updated_at"][:16].replace("T", " ")] for s in desk.store.scenarios(user)]
        return pd.DataFrame(rows, columns=columns)

    def save_scenario(thread: str | None, name: str, mode_label: str, audience_label: str, current: dict | None):
        user = need_user(current)
        if not thread:
            raise UserInputError("Ask a question first; the scenario keeps its request, limits and results.")
        values = runner.values(thread)
        desk.store.save_scenario(user, name, values, MODE_BY_LABEL[mode_label], AUDIENCE_BY_LABEL[audience_label])
        gr.Info("Scenario saved.")
        return "", scenarios_frame(user)

    def pick_scenario(frame: pd.DataFrame, event: gr.SelectData):
        try:
            row = frame.iloc[event.index[0]]
            return int(row.iloc[0]), str(row.iloc[1])
        except (IndexError, ValueError, TypeError):
            return gr.skip(), gr.skip()

    def rename_scenario(scenario_id: int | None, name: str, current: dict | None):
        user = need_user(current)
        if not scenario_id:
            raise UserInputError("Pick a scenario first.")
        desk.store.rename_scenario(user, int(scenario_id), name)
        return scenarios_frame(user)

    def duplicate_scenario(scenario_id: int | None, name: str, current: dict | None):
        user = need_user(current)
        if not scenario_id:
            raise UserInputError("Pick a scenario first.")
        desk.store.duplicate_scenario(user, int(scenario_id), name if name and not name.endswith("(copy)") else None)
        gr.Info("Copy made; the original is unchanged.")
        return scenarios_frame(user)

    def reopen_scenario(scenario_id: int | None, threads: list[str], current: dict | None):
        user = need_user(current)
        if not scenario_id:
            raise UserInputError("Pick a scenario first.")
        scenario = desk.store.scenario(user, int(scenario_id))
        thread = scenario["thread"]
        if not thread or not runner.values(thread).get("options"):
            thread = runner.new_thread()
            runner.run(scenario["request"], thread=thread, mode=scenario["mode"], audience=scenario["audience"], user=user.full_name, constraints=scenario["constraints"], home_country=user.assigned_country)
            desk.store.set_scenario_thread(scenario["id"], thread)
        threads = [*threads, thread]
        return load_run(thread, threads, current)

    # Monthly check, decisions, answer quality.

    def log_frame() -> pd.DataFrame:
        columns = ["When", "Approved by", "Role", "Countries", "Decision", "Net value", "Served", "Hires", "Notices", "Limits", "Question"]
        rows = [
            [
                r["time"].removesuffix(" UTC"),
                r["full_name"],
                ROLE_LABELS.get(r["role"], r["role"]),
                r["countries"],
                r["option"],
                money(float(r["net_value"] or 0)),
                f"{float(r['service_rate'] or 0):.1%}",
                f"{float(r['hires'] or 0):.0f}",
                f"{float(r['notices'] or 0):.0f}",
                r["limits"] or "none",
                r["request"],
            ]
            for r in reversed(services.log.rows())
        ]
        return pd.DataFrame(rows, columns=columns)

    def monthly_table(country: str, month: str) -> pd.DataFrame:
        as_of = MONTHS.index(month) + 1
        forecast = services.planner.forecast.demand(country)[: as_of - 1]
        return pd.DataFrame({"Month": list(MONTHS[: as_of - 1]), "Forecast": [round(float(v)) for v in forecast], "Actual sign ups": [round(float(v)) for v in forecast]})

    def review(country: str, month: str, table: pd.DataFrame, headcount: float | None, mode_label: str, threads: list[str], current: dict | None):
        user = need_user(current)
        if table is None or table.empty:
            raise UserInputError("Fill in the actual sign ups for each month.")
        values = []
        for raw in table["Actual sign ups"].tolist():
            if raw is None or str(raw).strip() == "":
                raise UserInputError("Every month needs a count of sign ups; none may be left empty.")
            try:
                number = float(raw)
            except ValueError as exc:
                raise UserInputError(f"{raw!r} is not a number. Use whole numbers of zero or more.") from exc
            if number < 0 or not number.is_integer():
                raise UserInputError("Sign up counts must be whole numbers of zero or more.")
            values.append(number)
        if headcount is None or not float(headcount).is_integer():
            raise UserInputError("The change in headcount must be a whole number (0 when nobody left or joined).")
        result = review_monthly_update(runner, country, MONTHS.index(month) + 1, values, int(headcount), user.full_name, MODE_BY_LABEL[mode_label])
        # Proposals for this country made before these numbers are now out of date.
        desk.store.mark_data_changed(result.country, f"monthly check for {month}, sign ups {result.gap:+.0%} against the forecast")
        if not result.started:
            return result.message, None, threads
        threads = [*threads, result.thread]
        return f"{result.message} Its options are on **Workspace**.", result.thread, threads

    def eval_view():
        path = Path(services.crew.paths.eval_results) / "crew_eval_summary.csv"
        if not path.exists():
            return pd.DataFrame(), None, "Not run yet. The check without models takes a few seconds; the check with models runs from a terminal with `staffpilot-crew eval --online`."
        summary = pd.read_csv(path)
        note = "This run used no models, so it checks how requests are read and options chosen, not how well each model writes." if (summary["model_calls"] == 0).all() else ""
        shown = summary.rename(
            columns={
                "setup": "Setup",
                "questions": "Requests",
                "task_accuracy": "Right answers",
                "grounded_rate": "Every number traced",
                "cost_per_request_usd": "Cost per request ($)",
                "seconds_per_request": "Seconds per request",
            }
        )[["Setup", "Requests", "Right answers", "Every number traced", "Cost per request ($)", "Seconds per request"]]
        return shown.round(3), evaluation.pareto_chart(summary), note

    def run_offline_eval():
        rows, summary = evaluation.evaluate(services, "sample" if "sample" in data_note else "course")
        evaluation.save(rows, summary, services)
        return eval_view()

    # Sign in and personalisation.

    def personalize(user: User | None):
        prompts = starter_prompts(user, countries) if user else []
        starters = [gr.update(value=p, visible=True) for p in prompts[:starters_count]]
        starters += [gr.update(value="", visible=False)] * (starters_count - len(starters))
        banner = banner_html(user)
        diagnostics = sees_diagnostics(user)
        return (
            user.as_dict() if user else None,
            gr.update(visible=user is None and mode_name != "disabled"),
            gr.update(visible=user is not None),
            gr.update(visible=user is not None),
            user_bar(user),
            gr.update(value=banner, visible=bool(banner)),
            gr.update(value=AUDIENCE_LABELS[user.audience] if user else "Staffing manager"),
            gr.update(visible=bool(user) and user.role != "guest" and mode_name != "disabled"),
            gr.update(visible=bool(user) and mode_name != "disabled"),
            gr.update(visible=bool(user) and user.role == "admin"),
            users_frame(store) if user and user.role == "admin" else users_frame(None),
            gr.update(visible=diagnostics),
            gr.update(visible=diagnostics),
            gr.update(visible=diagnostics),
            gr.update(placeholder=prompts[0] if prompts else examples[0], value=""),
            gr.update(value=user.assigned_country if user and user.assigned_country in countries else countries[0]),
            scenarios_frame(user),
            *today_outputs(user, "My scope", []),
            proposals_frame(user, "My scope", []),
            # A new session starts on Today with nothing of the last user's open.
            gr.Tabs(selected="today"),
            None,
            *proposal_view(None, "", user),
            *starters,
        )

    def sign_in(username: str, password: str):
        user = store.authenticate(username, password)
        if user is None:
            return (*personalize(None), "Wrong username or password.", "")
        return (*personalize(user), "", "")

    def sign_out():
        return (*personalize(None), "Signed out.", "")

    def as_guest():
        return (*personalize(GUEST), "", "")

    def start(current: dict | None):
        if mode_name == "disabled":
            return (*personalize(who(current) or LOCAL_USER), "", "")
        return (*personalize(who(current)), "", "")

    def save_password(old: str, new: str, confirm: str, current: dict | None):
        user = need_user(current)
        updated = store.change_password(user.username, old or "", new or "", confirm or "")
        return (updated.as_dict(), gr.update(value="", visible=False), "Password changed. The next change is due in 90 days.", gr.update(visible=False), "", "", "")

    with gr.Blocks(title="StaffPilot Crew") as demo:
        current = gr.State(None)
        thread = gr.State(None)
        threads = gr.State([])
        proposal_id = gr.State(None)
        scenario_id = gr.State(None)

        with gr.Row(elem_classes="crew-top"):
            sample_note = "\n\n<span class='crew-quiet'>Using a synthetic sample of three made up countries.</span>" if "sample" in data_note else ""
            gr.Markdown(f"## StaffPilot\nStaffing decisions for {year}: see what needs you, compare the options, decide together.{sample_note}")
            with gr.Column(scale=0, min_width=300, elem_classes="crew-account"):
                bar = gr.Markdown(user_bar(LOCAL_USER if mode_name == "disabled" else None))
                with gr.Row(elem_classes="crew-account-buttons"):
                    password_button = gr.Button("Change password", size="sm", visible=False, scale=0, min_width=0)
                    sign_out_button = gr.Button("Sign out", size="sm", visible=False, scale=0, min_width=0)

        with gr.Column(visible=mode_name != "disabled", elem_classes="crew-login") as login_panel:
            gr.Markdown("### Sign in")
            username = gr.Textbox(label="Username", autofocus=True)
            password = gr.Textbox(label="Password", type="password")
            with gr.Row():
                sign_in_button = gr.Button("Sign in", variant="primary")
                guest_button = gr.Button("Look around as a guest", visible=mode_name == "optional")
            login_message = gr.Markdown()
            gr.Markdown("No account yet? Ask your administrator.", elem_classes="crew-quiet")

        with gr.Sidebar(label="Ask StaffPilot", position="right", open=True, width=420, visible=mode_name == "disabled") as sidebar:
            gr.Markdown("### Ask StaffPilot")
            request = gr.Textbox(label="What do you need to decide?", placeholder=examples[0], lines=3)
            ask = gr.Button("Get options", variant="primary")
            starters = [gr.Button("", size="sm", visible=False) for _ in range(starters_count)]
            status = gr.HTML()
            audience = gr.Radio(list(AUDIENCE_BY_LABEL), value="Staffing manager", label="Show the answer as")
            answer = gr.Markdown(elem_classes="crew-answer", buttons=["copy"])
            with gr.Accordion("How this answer was worked out", open=False, visible=False) as trace_box:
                trace = gr.HTML(trace_html({}))
                cost = gr.Markdown()
            with gr.Accordion("Model settings", open=False, visible=False) as settings_box:
                mode = gr.Radio(
                    list(MODE_BY_LABEL),
                    value="Best quality",
                    label="Language models",
                    info="Best quality uses the strongest model for each step, Lowest cost the cheapest good enough one, Private keeps everything on this computer. Without keys, answers come from templates and cost nothing.",
                )

        with gr.Column(visible=mode_name == "disabled") as main_panel:
            banner = gr.HTML(visible=False)
            password_message = gr.Markdown()
            with gr.Column(visible=False, elem_classes="crew-login") as password_panel:
                gr.Markdown("### Change password\nAt least 8 characters, and different from the current one.")
                old_password = gr.Textbox(label="Current password", type="password")
                new_password = gr.Textbox(label="New password", type="password")
                confirm_password = gr.Textbox(label="New password again", type="password")
                with gr.Row():
                    save_button = gr.Button("Save new password", variant="primary")
                    cancel_button = gr.Button("Cancel")

            with gr.Tabs(selected="today") as tabs:
                with gr.Tab("Today", id="today"):
                    with gr.Row():
                        scope = gr.Radio(list(SCOPES), value="My scope", label="Show", scale=2)
                        state_filter = gr.Dropdown(list(STATES), multiselect=True, label="States", scale=2)
                        today_button = gr.Button("Refresh", size="sm", scale=0, min_width=90)
                    today_summary = gr.Markdown()
                    gr.Markdown("#### Waiting on you")
                    action_table = gr.Dataframe(proposal_rows(desk, []), interactive=False, wrap=True)
                    gr.Markdown("<span class='crew-quiet'>Select a row to open the proposal.</span>")
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=3):
                            gr.Markdown("#### Alerts")
                            alerts = gr.HTML()
                            with gr.Row():
                                look_country = gr.Dropdown(countries, label="Look into a country", scale=3)
                                look_button = gr.Button("Ask about it", size="sm", scale=1)
                        with gr.Column(scale=2):
                            gr.Markdown("#### Hiring deadlines")
                            deadlines = gr.Dataframe(interactive=False, wrap=True)

                with gr.Tab("Proposals", id="proposals"):
                    with gr.Row():
                        p_scope = gr.Radio(list(SCOPES), value="My scope", label="Show", scale=2)
                        p_states = gr.Dropdown(list(STATES), multiselect=True, label="States", scale=2)
                    p_table = gr.Dataframe(proposal_rows(desk, []), interactive=False, wrap=True)
                    p_header = gr.Markdown("Pick a proposal from the list.")
                    p_cards = gr.HTML()
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=3):
                            gr.Markdown("#### Next step")
                            p_note = gr.Textbox(label="Note for the timeline (required)", placeholder="Why you are doing this, for the next person", lines=2)
                            with gr.Row():
                                action_buttons = {a: gr.Button(ACTIONS[a].label, size="sm", visible=False, interactive=False) for a in PROPOSAL_ACTIONS}
                            revise_button = gr.Button("Use the sidebar's latest options as new numbers", size="sm")
                            with gr.Row():
                                p_export = gr.DownloadButton("Download the hiring plan (CSV)", visible=False, size="sm")
                            p_copy = gr.Textbox(label="Proposal text", lines=6, buttons=["copy"], interactive=False)
                        with gr.Column(scale=2):
                            gr.Markdown("#### Timeline")
                            p_timeline = gr.HTML()

                with gr.Tab("Workspace", id="workspace"):
                    gr.Markdown("Ask a question in the sidebar; its options appear here.", elem_classes="crew-quiet")
                    cards = gr.HTML()
                    with gr.Column(visible=False) as decide_box:
                        with gr.Column(visible=False) as propose_box:
                            gr.Markdown("#### Turn an option into a proposal")
                            with gr.Row(equal_height=True):
                                option = gr.Dropdown(label="Option", choices=[], scale=2)
                                w_note = gr.Textbox(label="Note (required)", placeholder="Why this option", scale=3)
                            with gr.Row():
                                draft_button = gr.Button("Save as draft", size="sm")
                                submit_button = gr.Button("Submit for approval", variant="primary", size="sm")
                        view_only = gr.Markdown(visible=False, elem_classes="crew-quiet", value="You can compare and copy these options; a country or staffing manager turns them into a proposal.")
                        with gr.Row(equal_height=True):
                            change_text = gr.Textbox(label="Change the limits", placeholder="At most 5 hires a month", scale=3)
                            change_button = gr.Button("Plan again", scale=1)
                        with gr.Row(equal_height=True):
                            copy_kind = gr.Dropdown(list(COPY_FORMATS), value="Staffing summary", label="Copy as", scale=1)
                            copy_box = gr.Textbox(label="Ready to paste", lines=6, buttons=["copy"], interactive=False, scale=3)
                        with gr.Row(equal_height=True):
                            scenario_name = gr.Textbox(label="Save as a scenario", placeholder="Brazil Q3 campaign, draft 2", scale=3)
                            save_scenario_button = gr.Button("Save scenario", size="sm", scale=1)

                with gr.Tab("Saved scenarios", id="scenarios"):
                    s_table = gr.Dataframe(scenarios_frame(None), interactive=False, wrap=True)
                    gr.Markdown("<span class='crew-quiet'>Select a row, then reopen, rename or copy it. A copy runs afresh when reopened; the original stays.</span>")
                    with gr.Row(equal_height=True):
                        s_name = gr.Textbox(label="Name", scale=3)
                        s_reopen = gr.Button("Reopen", variant="primary", size="sm", scale=1)
                        s_rename = gr.Button("Rename", size="sm", scale=1)
                        s_copy = gr.Button("Duplicate", size="sm", scale=1)

                with gr.Tab("Monthly check", id="monthly"):
                    gr.Markdown("Enter the real sign ups so far. Proposals for that country made before this check are marked out of date. If sign ups are more than 10% away from the forecast, or the team changed, StaffPilot also proposes next month's hires or notices on **Workspace**.")
                    with gr.Row():
                        m_country = gr.Dropdown(countries, value=countries[0], label="Country")
                        m_month = gr.Dropdown(list(MONTHS[1:]), value="March", label="Month of the check")
                        m_headcount = gr.Number(0, label="Change in agents on staff since the plan", info="For example -2 if two agents left.", precision=0)
                    m_table = gr.Dataframe(monthly_table(countries[0], "March"), label="Sign ups so far (edit the last column)", interactive=True)
                    m_button = gr.Button("Check this month", variant="primary")
                    m_result = gr.Markdown()

                with gr.Tab("Decisions", id="decisions"):
                    log = gr.Dataframe(log_frame(), label="Approved decisions, newest first", interactive=False, wrap=True)
                    gr.Button("Refresh", size="sm").click(log_frame, outputs=log)

                if planner_pages:
                    with gr.Tab("Full planner", id="planner"):
                        gr.Markdown("The detailed planner (the year plan, policies, scenarios, country memos and the data) opens at [/planner/](/planner/).")

                with gr.Tab("Answer quality", id="quality", visible=False) as quality_tab:
                    gr.Markdown("How often StaffPilot reads a request right and picks the right option, and what each model setup costs.")
                    e_note = gr.Markdown(elem_classes="crew-quiet")
                    e_table = gr.Dataframe(label="Each setup on the labelled requests", interactive=False)
                    e_plot = gr.Plot(label="Accuracy against cost")
                    e_button = gr.Button("Run the check without models", size="sm")

                with gr.Tab("Users", id="users", visible=False) as users_tab:
                    gr.Markdown("Every account. Passwords are kept only as salted hashes, which are not shown.", elem_classes="crew-quiet")
                    users_table = gr.Dataframe(users_frame(None), interactive=False)

        # Wiring.
        today_parts = [action_table, alerts, deadlines, today_summary]
        detail = [p_header, p_cards, p_timeline, p_export, p_copy, *action_buttons.values()]
        personal = [
            current, login_panel, main_panel, sidebar, bar, banner, audience, password_button, sign_out_button,
            users_tab, users_table, quality_tab, trace_box, settings_box, request, m_country, s_table,
            *today_parts, p_table, tabs, proposal_id, *detail, *starters,
        ]  # fmt: skip
        sign_in_button.click(sign_in, [username, password], [*personal, login_message, password])
        password.submit(sign_in, [username, password], [*personal, login_message, password])
        guest_button.click(as_guest, outputs=[*personal, login_message, password])
        sign_out_button.click(sign_out, outputs=[*personal, login_message, password]).then(
            lambda: (None, [], None, "", "", "", gr.update(visible=False), gr.update(visible=False)),
            outputs=[thread, threads, proposal_id, answer, cards, status, decide_box, password_panel],
        )
        demo.load(start, current, [*personal, login_message, password])
        password_button.click(lambda: gr.update(visible=True), outputs=password_panel)
        cancel_button.click(lambda: (gr.update(visible=False), "", "", ""), outputs=[password_panel, old_password, new_password, confirm_password])
        save_button.click(calm(save_password, 7), [old_password, new_password, confirm_password, current], [current, banner, password_message, password_panel, old_password, new_password, confirm_password])

        shown = [answer, trace, status, cards, option, decide_box, propose_box, view_only, cost, copy_box]
        run_targets = [*shown, thread, threads, tabs]
        ask_inputs = [request, mode, audience, threads, current]
        ask.click(calm(send, len(run_targets)), ask_inputs, run_targets)
        request.submit(calm(send, len(run_targets)), ask_inputs, run_targets)
        for button in starters:
            button.click(lambda text: text, button, request).then(calm(send, len(run_targets)), ask_inputs, run_targets)
        change_button.click(calm(change_limits, len(run_targets)), [thread, change_text, threads, current], run_targets)
        audience.change(switch_view, [thread, audience], answer)
        copy_kind.change(show_copy, [thread, copy_kind, current], copy_box)

        proposal_targets = [proposal_id, w_note, tabs, *detail, *today_parts]
        draft_button.click(calm(lambda t, o, n, c: make_proposal(t, o, n, False, c), len(proposal_targets)), [thread, option, w_note, current], proposal_targets)
        submit_button.click(calm(lambda t, o, n, c: make_proposal(t, o, n, True, c), len(proposal_targets)), [thread, option, w_note, current], proposal_targets)
        for name, button in action_buttons.items():
            targets = [p_note, *detail, *today_parts, p_table, log]
            button.click(calm(act(name), len(targets)), [proposal_id, p_note, current, p_scope, p_states], targets)
        p_note.change(lambda pid, note, c: proposal_view(pid, note, who(c))[5:], [proposal_id, p_note, current], list(action_buttons.values()))
        revise_targets = [proposal_id, p_note, tabs, *detail]
        revise_button.click(calm(revise, len(revise_targets)), [proposal_id, thread, p_note, current], revise_targets)
        open_targets = [proposal_id, tabs, p_note, *detail]
        action_table.select(open_selected, [action_table, current], open_targets)
        p_table.select(open_selected, [p_table, current], open_targets)
        for control in (p_scope, p_states):
            control.change(lambda c, s, st: proposals_frame(who(c), s, st), [current, p_scope, p_states], p_table)
        for control in (scope, state_filter):
            control.change(today_refresh, [current, scope, state_filter], today_parts)
        today_button.click(today_refresh, [current, scope, state_filter], today_parts)
        look_button.click(calm(look_into, 1), [look_country, current], request).then(calm(send, len(run_targets)), ask_inputs, run_targets)

        save_scenario_button.click(calm(save_scenario, 2), [thread, scenario_name, mode, audience, current], [scenario_name, s_table])
        s_table.select(pick_scenario, s_table, [scenario_id, s_name])
        s_rename.click(calm(rename_scenario, 1), [scenario_id, s_name, current], s_table)
        s_copy.click(calm(duplicate_scenario, 1), [scenario_id, s_name, current], s_table)
        s_reopen.click(calm(reopen_scenario, len(run_targets)), [scenario_id, threads, current], run_targets)

        m_country.change(monthly_table, [m_country, m_month], m_table)
        m_month.change(monthly_table, [m_country, m_month], m_table)
        m_button.click(calm(review, 3), [m_country, m_month, m_table, m_headcount, mode, threads, current], [m_result, thread, threads]).then(
            load_run, [thread, threads, current], run_targets
        ).then(today_refresh, [current, scope, state_filter], today_parts)
        e_button.click(run_offline_eval, outputs=[e_table, e_plot, e_note])
        demo.load(eval_view, outputs=[e_table, e_plot, e_note])
    return demo


def serve(services: CrewServices, runner: CrewRunner, data_note: str, examples: list[str], port: int = 7861, planner_pages: bool = True) -> None:
    from fastapi import FastAPI
    from staffpilot import app as hackathon
    import uvicorn

    mode_name = auth_mode()
    store = None
    if mode_name != "disabled":
        store = AuthStore(services.crew.paths.app_db)
        created = store.seed_demo_accounts()
        if created:
            print(f"Created {created} demo accounts in {services.crew.paths.app_db}; they are listed in the README.")  # noqa: T201
    server = FastAPI()
    theme = hackathon.theme()
    if planner_pages:
        server = gr.mount_gradio_app(server, hackathon.build_app(services.planner), path="/planner", theme=theme, css=hackathon.CSS)
    crew = build_crew_app(services, runner, data_note, examples, planner_pages, store, mode_name)
    server = gr.mount_gradio_app(server, crew, path="/", theme=theme, css=CSS, footer_links=[])
    uvicorn.run(server, host="127.0.0.1", port=port)
