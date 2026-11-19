"""The graph state, with a reducer on every field that parallel branches write.

Country branches started with ``Send`` run in the same step. LangGraph raises
``InvalidUpdateError`` when two of them write a field that has no reducer, so every field a
branch writes (analyses, results, facts, the trace and the model calls) merges instead of
overwriting. Only plain dicts, lists, strings and numbers are stored, so the state can be
saved by any checkpointer.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


def merge_dicts(old: dict[str, Any] | None, new: dict[str, Any] | None) -> dict[str, Any]:
    """Keys from `new` replace the same keys in `old`; other keys stay."""
    return {**(old or {}), **(new or {})}


def latest_by_country(old: dict[str, Any] | None, new: dict[str, Any] | None) -> dict[str, Any]:
    """One result per country; a later version replaces an earlier one, never the other way round."""
    merged = dict(old or {})
    for country, result in (new or {}).items():
        if country not in merged or result.get("version", 0) >= merged[country].get("version", 0):
            merged[country] = result
    return merged


class CrewState(TypedDict, total=False):
    # The request and how to answer it.
    request: str
    trigger: str  # "request" or "monthly_update"
    monthly: dict[str, Any]  # country, as_of, actual sign ups and headcount change for a monthly update
    mode: str  # best_quality, lowest_cost or private
    audience: str  # manager, country_manager or finance
    user: str
    home_country: str  # a country manager's own country, used when a request names none

    # Written by the Planner.
    plan: dict[str, Any]
    user_constraints: dict[str, Any]

    # Written by the parallel country branches (Analyst and Optimizer).
    analyses: Annotated[dict[str, dict[str, Any]], merge_dicts]
    country_results: Annotated[dict[str, dict[str, Any]], latest_by_country]
    fact_ledger: Annotated[dict[str, float], merge_dicts]

    # Written by the Decision critic.
    retry_counts: dict[str, int]
    critic_feedback: dict[str, list[str]]
    options: list[dict[str, Any]]
    chosen: str
    open_issues: list[str]

    # Written by the Writer and checked by the Decision critic.
    draft: str
    answer: str
    unverified: list[str]
    flagged: list[str]
    writer_model: dict[str, Any]

    # Human in the loop.
    approval_status: str  # Planning, Waiting for approval, Re-optimizing, Approved, Needs details
    decision: dict[str, Any]

    # Shown in the live trace; every agent appends.
    trace: Annotated[list[dict[str, Any]], operator.add]
    routing_trace: Annotated[list[dict[str, Any]], operator.add]
    calls: Annotated[list[dict[str, Any]], operator.add]


class CountryTask(TypedDict, total=False):
    """What a country branch receives through ``Send``."""

    country: str
    request: str
    trigger: str
    monthly: dict[str, Any]
    mode: str
    plan: dict[str, Any]
    user_constraints: dict[str, Any]
    feedback: list[str]
    version: int  # grows with every Optimizer run of the thread, so the newest result wins
    attempt: int  # 0 on a first run, then the critic's retry number
