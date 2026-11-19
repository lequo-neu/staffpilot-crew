"""Compare the router with each single model on labelled multi step requests.

A setup is either the router in one mode, or one model doing every step. Each request is run
through the whole graph until it pauses for approval, and scored on what can be checked
without a judge: the countries read, the audience, the manager's limits, asking back when it
should, the option the critic picks and whether infeasible limits were spotted. On top of
that every answer is checked for numbers without a source.

Offline, every setup collapses to the deterministic backend, so an offline run measures the
graph, not the models. Online setups only run when asked for and when their keys exist, and
each one costs real money; `staffpilot-crew eval --online` prints the number of runs first.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import resources
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd

from staffpilot_crew.config import CrewSettings, ModelEntry
from staffpilot_crew.graph import CrewRunner
from staffpilot_crew.models import OFFLINE
from staffpilot_crew.router import Difficulty, Mode, RouteChoice, Router
from staffpilot_crew.services import CrewServices


@dataclass
class Setup:
    name: str
    kind: str  # "router" or "single"
    mode: Mode = "best_quality"
    model: str | None = None


def load_questions(question_set: str = "sample") -> list[dict[str, Any]]:
    data = json.loads(resources.files("staffpilot_crew").joinpath("eval_questions.json").read_text())
    return [q for q in data["questions"] if q["set"] == question_set]


class FixedRouter(Router):
    """Sends every step to one model, for the single model baselines."""

    def __init__(self, crew: CrewSettings, entry: ModelEntry) -> None:
        super().__init__(crew, [entry.name], local_ready=entry.provider == "local")
        self.entry = entry

    def choose(self, step: str, difficulty: Difficulty, mode: Mode = "best_quality") -> RouteChoice:
        return self._choice(self.entry, step, f"single model baseline: {self.entry.label} does every step")

    def escalate(self, current: RouteChoice, difficulty: Difficulty, mode: Mode = "best_quality") -> RouteChoice:
        return RouteChoice(**{**current.as_dict(), "reason": "single model baseline: no escalation"})


def setups(services: CrewServices, online: bool) -> list[Setup]:
    found = [Setup("Router: best quality", "router", "best_quality")]
    if not online:
        return [Setup("Offline backend (no model)", "router", "best_quality")]
    found += [Setup("Router: lowest cost", "router", "lowest_cost")]
    if services.registry.local_ready():
        found.append(Setup("Router: private", "router", "private"))
    for entry in services.crew.router.models:
        if entry.name in services.registry.available() or (entry.provider == "local" and services.registry.local_ready()):
            found.append(Setup(f"Only {entry.label}", "single", model=entry.name))
    return found


def monthly_payload(planner, spec: dict[str, Any]) -> dict[str, Any]:
    country = planner.resolve_country(spec["country"])
    demand = planner.forecast.demand(country)[: spec["as_of"] - 1]
    return {"country": country, "as_of": spec["as_of"], "actual_signups": [round(float(v) * spec["scale"]) for v in demand], "headcount_change": 0}


def score(question: dict[str, Any], values: dict[str, Any]) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    asked_back = values.get("approval_status") == "Needs details"
    checks["asks_back"] = asked_back == question.get("asks_back", False)
    if asked_back or question.get("asks_back"):
        return checks
    plan = values.get("plan") or {}
    checks["countries"] = sorted(plan.get("countries", [])) == sorted(question.get("countries", []))
    if "audience" in question:
        checks["audience"] = values.get("audience") == question["audience"]
    if "limits" in question:
        read = values.get("user_constraints") or {}
        checks["limits"] = all(
            read.get(name) is not None and abs(float(read[name]) - float(value)) < 1e-6 for name, value in question["limits"].items()
        ) and all(read.get(name) is None for name in ("max_hires_per_month", "budget_usd", "service_goal", "automation_share") if name not in question["limits"])
    if "chosen" in question:
        checks["chosen"] = values.get("chosen") == question["chosen"]
    if "infeasible" in question:
        results = values.get("country_results") or {}
        checks["infeasible"] = any(r.get("strict_issues") for r in results.values()) == question["infeasible"]
    return checks


def run_setup(services: CrewServices, setup: Setup, questions: list[dict[str, Any]], progress: Callable[[str], None] | None = None) -> list[dict[str, Any]]:
    if setup.kind == "single":
        entry = services.crew.router.model(setup.model)
        services.router_override = lambda: FixedRouter(services.crew, entry)
    else:
        services.router_override = None
    runner = CrewRunner(services)
    rows = []
    try:
        for question in questions:
            monthly = monthly_payload(services.planner, question["monthly"]) if question.get("monthly") else None
            started = time.perf_counter()
            values = runner.run(question["request"], mode=setup.mode, monthly=monthly)
            seconds = time.perf_counter() - started
            checks = score(question, values)
            calls = values.get("calls") or []
            costs = [c["cost_usd"] for c in calls if c.get("cost_source") == "measured"]
            unknown = sum(1 for c in calls if c.get("cost_source") in {"unknown price", "no token counts"})
            writers = [r for r in values.get("routing_trace", []) if r.get("agent") == "Writer"]
            rows.append(
                {
                    "setup": setup.name,
                    "kind": setup.kind,
                    "question": question["id"],
                    "correct": all(checks.values()),
                    "checks": json.dumps(checks),
                    "grounded": not values.get("flagged"),
                    "writer_first_try": len(writers) <= 1,
                    "reading_correct": checks.get("countries", True) and checks["asks_back"] and checks.get("limits", True),
                    "seconds": round(seconds, 3),
                    "cost_usd": round(sum(costs), 6),
                    "unpriced_calls": unknown,
                    "model_calls": sum(1 for c in calls if c.get("model") != OFFLINE),
                    "models": ", ".join(sorted({r["model"] for r in values.get("routing_trace", [])})),
                }
            )
            if progress:
                progress(f"{setup.name} {question['id']}: {'correct' if rows[-1]['correct'] else 'wrong'}")
    finally:
        services.router_override = None
    return rows


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    summary = (
        rows.groupby(["setup", "kind"], sort=False)
        .agg(
            questions=("question", "count"),
            task_accuracy=("correct", "mean"),
            grounded_rate=("grounded", "mean"),
            writer_first_try=("writer_first_try", "mean"),
            cost_per_request_usd=("cost_usd", "mean"),
            unpriced_calls=("unpriced_calls", "sum"),
            seconds_per_request=("seconds", "mean"),
            model_calls=("model_calls", "sum"),
        )
        .reset_index()
    )
    summary["cost_note"] = np.where(summary["unpriced_calls"] > 0, "some calls have no price; cost is a lower bound", "measured from token counts")
    summary.loc[summary["model_calls"] == 0, "cost_note"] = "no model calls"
    summary["on_frontier"] = pareto(summary, "cost_per_request_usd")
    return summary


def pareto(summary: pd.DataFrame, cost: str) -> list[bool]:
    """True for setups no other setup beats on both accuracy and cost (or time)."""
    flags = []
    for _, row in summary.iterrows():
        dominated = any(
            (o[cost] <= row[cost] and o["task_accuracy"] >= row["task_accuracy"])
            and (o[cost] < row[cost] or o["task_accuracy"] > row["task_accuracy"])
            for _, o in summary.iterrows()
        )
        flags.append(not dominated)
    return flags


def routing_table(rows: pd.DataFrame, services: CrewServices) -> pd.DataFrame:
    """Measured accuracy per model and step, from the single model runs only."""
    singles = rows[rows["kind"] == "single"]
    records = []
    labels = {e.label: e.name for e in services.crew.router.models}
    for setup, group in singles.groupby("setup"):
        name = labels.get(setup.removeprefix("Only "))
        if not name:
            continue
        records.append({"model": name, "step": "classify", "accuracy": group["reading_correct"].mean(), "questions": len(group)})
        records.append({"model": name, "step": "write", "accuracy": group["writer_first_try"].mean(), "questions": len(group)})
    table = pd.DataFrame(records, columns=["model", "step", "accuracy", "questions"])
    table["source"] = "staffpilot-crew eval"
    return table


def evaluate(services: CrewServices, question_set: str = "sample", online: bool = False, progress: Callable[[str], None] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    questions = load_questions(question_set)
    rows = []
    for setup in setups(services, online):
        rows += run_setup(services, setup, questions, progress)
    frame = pd.DataFrame(rows)
    return frame, summarize(frame)


def save(rows: pd.DataFrame, summary: pd.DataFrame, services: CrewServices, folder: str | Path | None = None) -> Path:
    folder = Path(folder or services.crew.paths.eval_results)
    folder.mkdir(parents=True, exist_ok=True)
    rows.to_csv(folder / "crew_eval_rows.csv", index=False)
    summary.to_csv(folder / "crew_eval_summary.csv", index=False)
    table = routing_table(rows, services)
    if not table.empty:
        table.to_csv(folder / "routing_table.csv", index=False)
    return folder


def pareto_chart(summary: pd.DataFrame, cost: str = "cost_per_request_usd"):
    """Accuracy against cost (or seconds) per setup, with the frontier joined by a line."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.2))
    flags = pareto(summary, cost)
    for (_, row), front in zip(summary.iterrows(), flags, strict=True):
        color = "#1f5f8b" if row["kind"] == "router" else "#9a9a9a"
        ax.scatter(row[cost], row["task_accuracy"], s=70, color=color, edgecolor="black" if front else "none", zorder=3)
        ax.annotate(row["setup"], (row[cost], row["task_accuracy"]), textcoords="offset points", xytext=(6, 4), fontsize=8)
    front = summary[flags].sort_values(cost)
    if len(front) > 1:
        ax.plot(front[cost], front["task_accuracy"], color="#1f5f8b", linewidth=1, zorder=2)
    ax.set_xlabel("Cost per request (US dollars)" if cost.startswith("cost") else "Seconds per request")
    ax.set_ylabel("Task accuracy")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.set_title("Router and single models (outlined: on the Pareto frontier)", fontsize=10)
    fig.tight_layout()
    return fig
