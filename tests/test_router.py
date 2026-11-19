"""The router in its three modes, with and without keys, and the model call records."""

from __future__ import annotations

from staffpilot.agent.backends import OfflineBackend

from staffpilot_crew.config import load_crew_settings
from staffpilot_crew.models import OFFLINE, ModelRegistry
from staffpilot_crew.router import Router, assess


SIMPLE = assess(1, 0, [])
HARD = assess(3, 2, ["a service goal with a cap on hires"])
ALL_HOSTED = ["claude", "gpt_nano", "gemini_flash"]


def router(available, local_ready=False):
    return Router(load_crew_settings(), available, local_ready)


def test_difficulty_levels():
    assert SIMPLE.level == "simple"
    assert assess(2, 0, []).level == "moderate"
    assert assess(1, 2, []).level == "moderate"
    assert HARD.level == "hard"
    assert "pull against each other" in HARD.describe()


def test_no_key_falls_back_to_the_deterministic_backend_in_every_mode():
    for mode in ("best_quality", "lowest_cost", "private"):
        choice = router([]).choose("write", SIMPLE, mode)
        assert choice.model == OFFLINE
        assert "deterministic" in choice.reason


def test_best_quality_uses_the_models_assigned_in_the_proposal():
    r = router(ALL_HOSTED)
    assert r.choose("reason", HARD).model == "claude"
    assert r.choose("decide", HARD).model == "claude"
    assert r.choose("write", HARD).model == "gemini_flash"
    assert r.choose("classify", SIMPLE).model == "gpt_nano"


def test_lowest_cost_picks_the_cheapest_model_above_the_bar():
    r = router(ALL_HOSTED)
    simple = r.choose("classify", SIMPLE, "lowest_cost")
    assert simple.model == "gpt_nano"
    assert simple.accuracy_source == "measured"
    # A hard request raises the bar to 95%, which GPT 5 nano's measured 84% does not reach.
    hard = r.choose("classify", HARD, "lowest_cost")
    assert hard.model == "claude"
    assert "95%" in hard.reason


def test_prior_values_are_marked_as_prior():
    choice = router(ALL_HOSTED).choose("reason", SIMPLE, "lowest_cost")
    assert choice.accuracy_source == "prior"
    assert "prior" in choice.reason


def test_private_mode_never_uses_a_hosted_model():
    assert router(ALL_HOSTED).choose("reason", HARD, "private").model == OFFLINE
    local = router(ALL_HOSTED, local_ready=True).choose("reason", HARD, "private")
    assert local.model == "gemma_local"


def test_local_model_is_the_fallback_when_no_hosted_key_exists():
    assert router([], local_ready=True).choose("write", SIMPLE, "lowest_cost").model == "gemma_local"


def test_escalation_moves_to_a_more_accurate_model():
    r = router(ALL_HOSTED)
    nano = r.choose("write", SIMPLE, "lowest_cost")
    assert nano.model == "gpt_nano"
    stronger = r.escalate(nano, SIMPLE, "lowest_cost")
    assert stronger.escalated
    assert stronger.model == "gemini_flash"
    offline = router([]).choose("write", SIMPLE)
    assert not router([]).escalate(offline, SIMPLE).escalated


def test_registry_is_empty_offline_and_records_every_call(settings):
    registry = ModelRegistry(load_crew_settings(), settings)
    assert registry.available() == []
    client = registry.client(OFFLINE)
    route, record = client.call("route", "What is the plan for Alpha?", ["Alpha"], agent="Planner", step="reason")
    assert route.countries == ["Alpha"]
    assert record.cost_usd == 0.0
    assert record.cost_source == "free"
    assert registry.records() == [record]


class CountingBackend(OfflineBackend):
    """Stands in for an online model: same answers, but it reports token counts."""

    def __init__(self):
        self.usage = {"input": 0, "output": 0}

    def route(self, question, countries, history=""):
        self.usage["input"] += 1_000
        self.usage["output"] += 100
        return super().route(question, countries, history)


def test_measured_cost_comes_from_token_counts_and_prices(settings):
    registry = ModelRegistry(load_crew_settings(), settings, ["anthropic"], lambda entry, s: CountingBackend())
    _, record = registry.client("claude").call("route", "Plan for Alpha?", ["Alpha"])
    assert (record.tokens_in, record.tokens_out) == (1_000, 100)
    # Claude Sonnet 5.5 at $2 and $10 per million tokens.
    assert record.cost_usd == (1_000 * 2.0 + 100 * 10.0) / 1_000_000
    assert record.cost_source == "measured"


def test_a_model_without_a_price_reports_an_unknown_cost(settings):
    registry = ModelRegistry(load_crew_settings(), settings, ["google"], lambda entry, s: CountingBackend())
    _, record = registry.client("gemini_flash").call("route", "Plan for Alpha?", ["Alpha"])
    assert record.cost_usd is None
    assert record.cost_source == "unknown price"


def test_a_failing_call_is_recorded_not_raised(settings):
    class Broken(OfflineBackend):
        def route(self, *args, **kwargs):
            raise TimeoutError("no answer")

    registry = ModelRegistry(load_crew_settings(), settings, ["openai"], lambda entry, s: Broken())
    result, record = registry.client("gpt_nano").call("route", "Plan?", [])
    assert result is None
    assert not record.ok
    assert "TimeoutError" in record.error
