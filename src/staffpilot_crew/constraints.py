"""The limits a manager writes into a request, read without any model.

The hackathon router already reads the intent, the countries and the scenario of a question.
What it does not read are the manager's own limits, which the crew needs for the Optimizer
and the Decision critic: a cap on hires, a budget, a service goal, an automation share and
who the answer is for. They are read here with plain patterns, so the result is the same
online and offline and can be tested.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field


Audience = Literal["manager", "country_manager", "finance", "people"]
AUDIENCE_LABELS = {"manager": "Staffing manager", "country_manager": "Country manager", "finance": "Finance", "people": "People team"}


class Constraints(BaseModel):
    """Limits set by the manager. None means no limit."""

    max_hires_per_month: int | None = Field(None, ge=0, le=10_000)
    budget_usd: float | None = Field(None, gt=0, description="Cap on the total staffing cost of the year")
    service_goal: float | None = Field(None, ge=0.0, le=1.0, description="Share of eligible advertisers served")
    automation_share: float | None = Field(None, ge=0.0, le=0.9, description="Share handled by an AI assistant")

    def count(self) -> int:
        return sum(v is not None for v in (self.max_hires_per_month, self.budget_usd, self.service_goal))

    def conflicts(self) -> list[str]:
        """Limits that pull against each other: serving more needs agents a cap or budget may forbid."""
        found = []
        if self.service_goal is not None and self.max_hires_per_month is not None:
            found.append("a service goal with a cap on hires")
        if self.service_goal is not None and self.budget_usd is not None:
            found.append("a service goal with a budget")
        return found

    def merged(self, other: Constraints) -> Constraints:
        """These limits with every limit `other` sets written over them."""
        return Constraints.model_validate({**self.model_dump(), **other.model_dump(exclude_none=True)})

    def describe(self) -> list[str]:
        lines = []
        if self.max_hires_per_month is not None:
            cap = self.max_hires_per_month
            lines.append(f"at most {cap} {'hire' if cap == 1 else 'hires'} a month")
        if self.budget_usd is not None:
            lines.append(f"a staffing budget of ${self.budget_usd:,.0f} for the year")
        if self.service_goal is not None:
            lines.append(f"at least {self.service_goal:.0%} of eligible advertisers served")
        if self.automation_share is not None:
            lines.append(f"{self.automation_share:.0%} of advertisers handled by an AI assistant")
        return lines


_NUMBER = r"(\d[\d,]*(?:\.\d+)?)"
_SCALE = {"k": 1e3, "thousand": 1e3, "m": 1e6, "mn": 1e6, "million": 1e6, "b": 1e9, "billion": 1e9}

_HIRE_CAP = [
    rf"(?:at most|no more than|max(?:imum)?(?: of)?|up to|cap(?:ped)? (?:at|of)|limit(?:ed)? to)\s+{_NUMBER}\s+(?:new\s+)?(?:hires?|people|agents|new agents)",
    rf"(?:hir(?:e|ing)|recruit(?:ing)?)\s+(?:\w+\s+){{0,4}}?(?:to\s+)?(?:at most|no more than|up to|max(?:imum)?)\s+{_NUMBER}",
    rf"{_NUMBER}\s+(?:hires?|new agents)\s+(?:a|per|each)\s+month\s+(?:at most|max(?:imum)?)",
]
_BUDGET = rf"(?:budget|spend(?:ing)?|cost)\s+(?:\w+\s+){{0,3}}?(?:of|at|below|under|to|within|is)?\s*\$\s?{_NUMBER}\s*(k|thousand|m|mn|million|b|billion)?\b"
_SERVICE = [
    rf"(?:at least|serve|served|serving|service goal of|service of)\s+(?:at least\s+)?{_NUMBER}\s?%",
    rf"{_NUMBER}\s?%\s+(?:of (?:eligible )?advertisers )?(?:served|service)",
]
# The share written before "handled by an AI" is the more specific reading, so it is tried first.
_AUTOMATION = [
    rf"{_NUMBER}\s?%\s+(?:of\s+)?(?:\w+\s+){{0,4}}?(?:handled|onboarded|served|done)\s+by\s+(?:an?\s+)?(?:ai|bot|assistant|automation)\b",
    rf"\b(?:ai|automat\w*|bot|assistant)\b[^%.]{{0,40}}?\b(?:handles?|onboards?|serves?|takes?|covers?)\s+(?:\w+\s+){{0,2}}?{_NUMBER}\s?%",
]


def _number(text: str) -> float:
    return float(text.replace(",", ""))


def read_constraints(text: str) -> Constraints:
    lowered = text.lower()
    values: dict[str, float | int] = {}
    for pattern in _HIRE_CAP:
        if match := re.search(pattern, lowered):
            values["max_hires_per_month"] = int(_number(match.group(1)))
            break
    if match := re.search(_BUDGET, lowered):
        values["budget_usd"] = _number(match.group(1)) * _SCALE.get(match.group(2) or "", 1.0)
    automation_span = None
    for pattern in _AUTOMATION:
        match = re.search(pattern, lowered)
        if match and 0 < (share := _number(match.group(1)) / 100) <= 0.9:
            values["automation_share"] = share
            automation_span = match.span(1)
            break
    for pattern in _SERVICE:
        for found in re.finditer(pattern, lowered):
            if automation_span and found.start(1) == automation_span[0]:
                continue
            share = _number(found.group(1)) / 100
            if 0 < share <= 1:
                values["service_goal"] = share
                break
        if "service_goal" in values:
            break
    return Constraints.model_validate(values)


def read_audience(text: str) -> Audience:
    lowered = text.lower()
    if re.search(r"\b(finance|cfo|budget note|for finance)\b", lowered):
        return "finance"
    if re.search(r"\b(people team|hiring notice|notice plan|hr|recruit(?:ing|ment) plan)\b", lowered):
        return "people"
    if re.search(r"\b(memo|country manager|note to the \w+ manager|brief the)\b", lowered):
        return "country_manager"
    return "manager"
