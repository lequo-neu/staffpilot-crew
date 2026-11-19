"""Everything the agents share: the planner, the models, the router, retrieval and the log.

The graph state holds only data. These services are passed to the graph when it is built, so
a saved checkpoint never contains a model client or a planner.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
import sqlite3
from typing import Any

from staffpilot.agent.rag import Retriever
from staffpilot.config import Settings, load_settings
from staffpilot.planner import Planner

from staffpilot_crew.config import CrewSettings, load_crew_settings
from staffpilot_crew.models import BackendFactory, ModelRegistry
from staffpilot_crew.router import Router


DECISION_COLUMNS = (
    "time",
    "username",
    "full_name",
    "role",
    "thread",
    "request",
    "countries",
    "option",
    "net_value",
    "total_cost",
    "service_rate",
    "hires",
    "notices",
    "limits",
    "open_issues",
    "status",
)
DECISION_SCHEMA = """
CREATE TABLE IF NOT EXISTS decision_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    time TEXT NOT NULL,
    username TEXT NOT NULL,
    full_name TEXT NOT NULL,
    role TEXT NOT NULL,
    thread TEXT,
    request TEXT,
    countries TEXT,
    option TEXT,
    net_value REAL,
    total_cost REAL,
    service_rate REAL,
    hires REAL,
    notices REAL,
    limits TEXT,
    open_issues TEXT,
    status TEXT NOT NULL
)
"""

# Built once from the fixed column names above; no user input ever reaches these statements.
INSERT_DECISION = f"INSERT INTO decision_log ({', '.join(DECISION_COLUMNS)}) VALUES ({', '.join('?' * len(DECISION_COLUMNS))})"  # noqa: S608
SELECT_DECISIONS = f"SELECT {', '.join(DECISION_COLUMNS)} FROM decision_log ORDER BY id"  # noqa: S608


@dataclass
class DecisionLog:
    """Approved decisions in SQLite, one row each, with who approved them. Nothing is applied elsewhere."""

    path: Path

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute(DECISION_SCHEMA)

    def write(self, row: dict[str, Any]) -> dict[str, Any]:
        record = {"time": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC"), **row}
        for required in ("username", "full_name", "role"):
            if not record.get(required):
                raise ValueError(f"A decision needs the approver's {required}")
        values = [record.get(k, "") for k in DECISION_COLUMNS]
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute(
                INSERT_DECISION,
                values,
            )
        return record

    def rows(self) -> list[dict[str, Any]]:
        with closing(sqlite3.connect(self.path)) as db:
            db.row_factory = sqlite3.Row
            found = db.execute(SELECT_DECISIONS).fetchall()
        return [dict(row) for row in found]


@dataclass
class CrewServices:
    planner: Planner
    crew: CrewSettings
    registry: ModelRegistry
    retriever: Retriever = field(default_factory=Retriever)
    log: DecisionLog | None = None
    # Set by the evaluation to send every step to one model.
    router_override: Callable[[], Router] | None = None

    def __post_init__(self) -> None:
        if self.log is None:
            self.log = DecisionLog(Path(self.crew.paths.app_db))

    @property
    def settings(self) -> Settings:
        return self.planner.settings

    def router(self) -> Router:
        if self.router_override is not None:
            return self.router_override()
        return Router(self.crew, self.registry.available(), self.registry.local_ready())

    @classmethod
    def build(
        cls,
        planner: Planner | None = None,
        crew: CrewSettings | None = None,
        providers: list[str] | None = None,
        factory: BackendFactory | None = None,
    ) -> CrewServices:
        crew = crew or load_crew_settings()
        planner = planner or Planner(load_settings())
        registry = ModelRegistry(crew, planner.settings, providers, factory)
        return cls(planner, crew, registry)
