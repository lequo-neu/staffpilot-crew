"""One interface for every model the crew can call, and a record of each call.

The hackathon already built the backends: ``OfflineBackend`` (a keyword router and template
writer, deterministic and free) and ``LangChainBackend`` (any provider through LangChain).
This module does not rebuild them. It wraps them so that every call made by an agent is
timed, its token counts are read, and its cost is worked out from the prices in the crew
config, with a clear note when the cost could not be measured.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
import time
from typing import Any

from staffpilot.agent.backends import OfflineBackend, available_providers, local_server_running, make_backend
from staffpilot.config import Settings

from staffpilot_crew.config import CrewSettings, ModelEntry


OFFLINE = "offline"


@dataclass
class CallRecord:
    """What one model call cost and how long it took."""

    agent: str
    step: str
    model: str
    spec: str
    method: str
    seconds: float
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float | None = 0.0
    cost_source: str = "free"  # "measured", "unknown price", "no token counts" or "free"
    ok: bool = True
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrackedClient:
    """A backend plus the crew's entry for it. Every call goes through `call`."""

    name: str
    spec: str
    backend: Any
    entry: ModelEntry | None = None
    records: list[CallRecord] = field(default_factory=list)

    @property
    def deterministic(self) -> bool:
        return self.entry is None

    def call(self, method: str, *args: Any, agent: str = "", step: str = "", **kwargs: Any) -> tuple[Any, CallRecord]:
        before = dict(getattr(self.backend, "usage", None) or {"input": 0, "output": 0})
        started = time.perf_counter()
        error, result = "", None
        try:
            result = getattr(self.backend, method)(*args, **kwargs)
        except Exception as exc:  # the caller decides how to fall back; the record keeps the reason
            error = f"{type(exc).__name__}: {str(exc)[:200]}"
        seconds = time.perf_counter() - started
        after = getattr(self.backend, "usage", None) or before
        tokens_in = int(after.get("input", 0)) - int(before.get("input", 0))
        tokens_out = int(after.get("output", 0)) - int(before.get("output", 0))
        record = CallRecord(agent, step, self.name, self.spec, method, round(seconds, 3), tokens_in, tokens_out)
        if self.deterministic:
            record.cost_usd, record.cost_source = 0.0, "free"
        elif tokens_in == 0 and tokens_out == 0:
            record.cost_usd, record.cost_source = None, "no token counts"
        else:
            cost = self.entry.cost(tokens_in, tokens_out)
            record.cost_usd, record.cost_source = (cost, "measured") if cost is not None else (None, "unknown price")
        if error:
            record.ok, record.error = False, error
        self.records.append(record)
        return result, record


BackendFactory = Callable[[ModelEntry, Settings], Any]


def default_factory(entry: ModelEntry, settings: Settings) -> Any:
    return make_backend(entry.spec, settings)


class ModelRegistry:
    """The models this computer can reach right now, built only when an agent needs them.

    `providers` defaults to the providers with a key (and "local" when LM Studio answers);
    STAFFPILOT_OFFLINE=1 makes it empty, so tests and CI never call a paid model.
    Tests pass their own `factory` to stand in for an online model without any key.
    """

    def __init__(
        self,
        crew: CrewSettings,
        settings: Settings,
        providers: list[str] | None = None,
        factory: BackendFactory | None = None,
    ) -> None:
        self.crew = crew
        self.settings = settings
        self.providers = list(available_providers() if providers is None else providers)
        self.factory = factory or default_factory
        self._clients: dict[str, TrackedClient] = {}

    def available(self) -> list[str]:
        return [m.name for m in self.crew.router.models if m.provider in self.providers]

    def local_ready(self) -> bool:
        return "local" in self.providers or local_server_running(self.settings.llm.local_url)

    def client(self, name: str) -> TrackedClient:
        if name not in self._clients:
            if name == OFFLINE:
                self._clients[name] = TrackedClient(OFFLINE, OFFLINE, OfflineBackend())
            else:
                entry = self.crew.router.model(name)
                self._clients[name] = TrackedClient(name, entry.spec, self.factory(entry, self.settings), entry)
        return self._clients[name]

    def records(self) -> list[CallRecord]:
        return [r for c in self._clients.values() for r in c.records]
