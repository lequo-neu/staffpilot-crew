"""The new package imports, and the hackathon core it builds on is available."""

import staffpilot_crew


def test_package_imports():
    assert staffpilot_crew.__version__


def test_hackathon_core_is_installed():
    from staffpilot import optimize, simulate

    assert optimize is not None
    assert simulate is not None


def test_langgraph_is_installed():
    from langgraph.graph import StateGraph

    assert StateGraph is not None
