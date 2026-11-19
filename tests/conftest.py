"""Synthetic data and offline settings, so the tests run anywhere in seconds.

No test reads the course file, calls a paid model or needs a local model server:
STAFFPILOT_OFFLINE=1 hides every provider, and online models are stood in for by
scripted backends passed to the registry.
"""

from __future__ import annotations

import pytest
from staffpilot.config import load_settings
from staffpilot.data import Dataset

from staffpilot_crew.config import load_crew_settings
from staffpilot_crew.sample import COUNTRIES, STAFFING, make_signups


@pytest.fixture(autouse=True, scope="session")
def _offline_and_outside_the_repository(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.setenv("STAFFPILOT_OFFLINE", "1")
    # The app's sign in form is tested on its own; elsewhere it must not block the tests.
    mp.setenv("STAFFPILOT_AUTH_MODE", "disabled")
    mp.setenv("STAFFPILOT_LOG_DIR", str(tmp_path_factory.mktemp("logs")))
    mp.setenv("STAFFPILOT_CACHE_DIR", str(tmp_path_factory.mktemp("cache")))
    yield
    mp.undo()


@pytest.fixture(scope="session")
def dataset() -> Dataset:
    return Dataset.from_frames(COUNTRIES, STAFFING, make_signups())


@pytest.fixture(scope="session")
def settings():
    return load_settings().with_updates(
        planning={"monte_carlo_runs": 3, "lookahead_months": 2}, llm={"models": {}, "fast_models": {}}
    )


@pytest.fixture(scope="session")
def planner(settings, dataset):
    from staffpilot.planner import Planner

    return Planner(settings, dataset)


@pytest.fixture
def crew_settings(tmp_path):
    return load_crew_settings().with_updates(
        paths={
            "app_db": str(tmp_path / "crew_app.sqlite"),
            "checkpoints": str(tmp_path / "checkpoints.sqlite"),
            "eval_results": str(tmp_path / "eval"),
        }
    )
