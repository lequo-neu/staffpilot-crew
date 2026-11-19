"""The LangGraph graph of the five agents, and a small runner for the app, the CLI and tests.

    START -> planner -> analyst (one branch per country, Send) -> dispatch
          -> optimizer (one branch per country, Send) -> critic
          -> (back to optimizer, at most `optimizer_retries` times) -> writer -> number_check
          -> (back to writer, at most `writer_retries` times) -> approval (interrupt)
          -> END on Approve, or optimizer again on Ask for changes

The planner ends the run early when it has to ask back. Every run also has a hard
`recursion_limit`, so no loop can run on even if a counter were wrong.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
import sqlite3
from typing import Any
import uuid

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send, interrupt

from staffpilot_crew import agents
from staffpilot_crew.auth import GUEST, User, approval_note, can_approve
from staffpilot_crew.constraints import Constraints, read_constraints
from staffpilot_crew.services import CrewServices
from staffpilot_crew.state import CountryTask, CrewState


def country_task(state: CrewState, country: str, version: int, attempt: int = 0, feedback: list[str] | None = None, limits: dict | None = None) -> CountryTask:
    return CountryTask(
        country=country,
        request=state.get("request", ""),
        trigger=state.get("trigger", "request"),
        monthly=state.get("monthly") or {},
        mode=state.get("mode", "best_quality"),
        plan=state["plan"],
        user_constraints=limits if limits is not None else (state.get("user_constraints") or {}),
        feedback=feedback or [],
        version=version,
        attempt=attempt,
    )


def next_version(state: CrewState) -> int:
    return 1 + max((r.get("version", 0) for r in (state.get("country_results") or {}).values()), default=-1)


def build_graph(services: CrewServices, checkpointer: Any = None):
    def after_planner(state: CrewState):
        if state["plan"].get("clarification"):
            return END
        return [Send("analyst", country_task(state, c, 0)) for c in state["plan"]["countries"]]

    def to_optimizer(state: CrewState):
        version = next_version(state)
        return [Send("optimizer", country_task(state, c, version)) for c in state["plan"]["countries"]]

    def after_critic(state: CrewState):
        feedback = state.get("critic_feedback") or {}
        if not feedback:
            return "writer"
        version, attempt = next_version(state), state["retry_counts"]["optimizer"]
        return [Send("optimizer", country_task(state, c, version, attempt, issues)) for c, issues in feedback.items()]

    def after_check(state: CrewState):
        return "approval" if state.get("approval_status") == "Waiting for approval" else "writer"

    def approval(state: CrewState) -> Command:
        decision = interrupt(
            {
                "status": "Waiting for approval",
                "options": state.get("options", []),
                "chosen": state.get("chosen"),
                "answer": state.get("answer", ""),
                "open_issues": state.get("open_issues", []),
            }
        )
        decision = decision or {}
        user = User.from_dict(decision.get("user")) or GUEST
        if decision.get("action", "approve") == "approve":
            # Checked again here, so no caller can log an approval the role does not allow.
            if not can_approve(user, state):
                raise PermissionError(approval_note(user, state))
            name = decision.get("option") or state["chosen"]
            card = next(c for c in state["options"] if c["name"] == name)
            limits = Constraints.model_validate(state.get("user_constraints") or {})
            row = services.log.write(
                {
                    "username": user.username,
                    "full_name": user.full_name,
                    "role": user.role,
                    "thread": decision.get("thread", ""),
                    "request": state.get("request", ""),
                    "countries": ", ".join(state["plan"]["countries"]),
                    "option": card["label"],
                    "net_value": round(card["totals"]["net_value"], 2),
                    "total_cost": round(card["totals"]["total_cost"], 2),
                    "service_rate": round(card["totals"]["service_rate"], 4),
                    "hires": card["totals"]["hires"],
                    "notices": card["totals"]["notices"],
                    "limits": "; ".join(limits.describe()),
                    "open_issues": "; ".join(card.get("issues", [])),
                    "status": "approved",
                }
            )
            trace = [agents.event(user.role_label, "approved", f"{card['label']} by {user.full_name}; logged, nothing applied elsewhere")]
            return Command(update={"approval_status": "Approved", "decision": row, "trace": trace}, goto=END)
        new = read_constraints(decision.get("text", "")).merged(Constraints.model_validate(decision.get("constraints") or {}))
        limits = Constraints.model_validate(state.get("user_constraints") or {}).merged(new).model_dump()
        version = next_version(state)
        sends = [Send("optimizer", country_task(state, c, version, 0, limits=limits)) for c in state["plan"]["countries"]]
        detail = "; ".join(new.describe()) or decision.get("text", "") or "no new limit"
        trace = [agents.event(user.role_label, "asked for changes", f"{user.full_name}: {detail}")]
        return Command(
            update={
                "user_constraints": limits,
                "retry_counts": {"optimizer": 0, "writer": 0},
                "approval_status": "Re-optimizing",
                "critic_feedback": {},
                "unverified": [],
                "writer_model": {},
                "trace": trace,
            },
            goto=sends,
        )

    graph = StateGraph(CrewState)
    graph.add_node("planner", lambda state: agents.planner_node(services, state))
    graph.add_node("analyst", lambda task: agents.analyst_node(services, task))
    graph.add_node("dispatch", lambda state: {})
    graph.add_node("optimizer", lambda task: agents.optimizer_node(services, task))
    graph.add_node("critic", lambda state: agents.critic_node(services, state))
    graph.add_node("writer", lambda state: agents.writer_node(services, state))
    graph.add_node("number_check", lambda state: agents.number_check_node(services, state))
    graph.add_node("approval", approval, destinations=("optimizer", END))
    graph.add_edge(START, "planner")
    graph.add_conditional_edges("planner", after_planner, ["analyst", END])
    graph.add_edge("analyst", "dispatch")
    graph.add_conditional_edges("dispatch", to_optimizer, ["optimizer"])
    graph.add_edge("optimizer", "critic")
    graph.add_conditional_edges("critic", after_critic, ["optimizer", "writer"])
    graph.add_edge("writer", "number_check")
    graph.add_conditional_edges("number_check", after_check, ["writer", "approval"])
    return graph.compile(checkpointer=checkpointer)


def sqlite_checkpointer(path: str | Path):
    """A checkpointer that keeps paused runs across restarts; the app shares it between threads."""
    from langgraph.checkpoint.sqlite import SqliteSaver

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(sqlite3.connect(str(path), check_same_thread=False))


class CrewRunner:
    """Start a run, read its state, approve it or ask for changes. One thread per conversation."""

    def __init__(self, services: CrewServices, checkpointer: Any = None) -> None:
        self.services = services
        self.checkpointer = checkpointer or InMemorySaver()
        self.graph = build_graph(services, self.checkpointer)

    def config(self, thread: str, recursion_limit: int | None = None) -> dict[str, Any]:
        limit = recursion_limit or self.services.crew.graph.recursion_limit
        return {"configurable": {"thread_id": thread}, "recursion_limit": limit}

    @staticmethod
    def new_thread() -> str:
        return uuid.uuid4().hex[:12]

    def initial(
        self,
        request: str,
        mode: str = "best_quality",
        audience: str | None = None,
        user: str = "",
        constraints: dict | None = None,
        monthly: dict | None = None,
        home_country: str | None = None,
    ) -> CrewState:
        state = CrewState(request=request, mode=mode, user=user, user_constraints=constraints or {}, trigger="monthly_update" if monthly else "request")
        if home_country:
            state["home_country"] = home_country
        if audience:
            state["audience"] = audience
        if monthly:
            state["monthly"] = monthly
        return state

    def stream(self, inputs: Any, thread: str, recursion_limit: int | None = None) -> Iterator[dict[str, Any]]:
        """Node updates as they happen, for the live trace."""
        yield from self.graph.stream(inputs, self.config(thread, recursion_limit), stream_mode="updates")

    def run(self, request: str, thread: str | None = None, recursion_limit: int | None = None, **kwargs: Any) -> dict[str, Any]:
        thread = thread or self.new_thread()
        self.graph.invoke(self.initial(request, **kwargs), self.config(thread, recursion_limit))
        return self.values(thread)

    def approve(self, thread: str, user: User, option: str | None = None) -> dict[str, Any]:
        """Approve the paused run as `user`. Raises PermissionError when the role may not approve."""
        if not can_approve(user, self.values(thread)):
            raise PermissionError(approval_note(user, self.values(thread)))
        self.graph.invoke(Command(resume={"action": "approve", "user": user.as_dict(), "option": option, "thread": thread}), self.config(thread))
        return self.values(thread)

    def ask_for_changes(self, thread: str, text: str = "", constraints: dict | None = None, user: User | None = None) -> dict[str, Any]:
        self.graph.invoke(self.resume_inputs("change", thread, text=text, constraints=constraints or {}, user=(user or GUEST).as_dict()), self.config(thread))
        return self.values(thread)

    def resume_inputs(self, action: str, thread: str, **kwargs: Any) -> Command:
        return Command(resume={"action": action, "thread": thread, **kwargs})

    def values(self, thread: str) -> dict[str, Any]:
        snapshot = self.graph.get_state(self.config(thread))
        values = dict(snapshot.values)
        values["thread"] = thread
        values["waiting"] = bool(snapshot.next)
        values["next"] = list(snapshot.next)
        return values
