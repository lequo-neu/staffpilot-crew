"""Reading the manager's limits and audience from plain words."""

from __future__ import annotations

import pytest

from staffpilot_crew.constraints import Constraints, read_audience, read_constraints


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Keep hiring in Brazil to at most 5 people", {"max_hires_per_month": 5}),
        ("No more than 2 hires per month in Alpha", {"max_hires_per_month": 2}),
        ("Stay within a budget of $1.5M", {"budget_usd": 1_500_000.0}),
        ("a cost under $200k", {"budget_usd": 200_000.0}),
        ("Serve at least 98% of eligible advertisers", {"service_goal": 0.98}),
        ("What if an AI assistant onboards 30% of small advertisers?", {"automation_share": 0.3}),
        (
            "40% of advertisers handled by an AI assistant, and 95% served",
            {"automation_share": 0.4, "service_goal": 0.95},
        ),
        ("Sign ups in the USA came in 20% above forecast in March", {}),
    ],
)
def test_read_constraints(text, expected):
    assert read_constraints(text).model_dump(exclude_none=True) == expected


def test_conflicts_and_merge():
    limits = Constraints(service_goal=0.98, max_hires_per_month=1)
    assert limits.count() == 2
    assert limits.conflicts() == ["a service goal with a cap on hires"]
    merged = limits.merged(Constraints(max_hires_per_month=5))
    assert merged.max_hires_per_month == 5
    assert merged.service_goal == 0.98


@pytest.mark.parametrize(
    ("text", "audience"),
    [
        ("Write a memo for the Vietnam country manager", "country_manager"),
        ("A note for finance on the cost", "finance"),
        ("What should we do in Brazil?", "manager"),
    ],
)
def test_read_audience(text, audience):
    assert read_audience(text) == audience
