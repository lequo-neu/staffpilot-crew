"""Sign in, the 90 day password rule, roles and the identity in the decision log."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
import sqlite3

import pytest

from staffpilot_crew import app
from staffpilot_crew.auth import (
    APPROVAL_HINT,
    DEMO_ACCOUNTS,
    GUEST,
    SCHEMA,
    SEEDED_ACCOUNTS,
    AuthStore,
    PasswordError,
    User,
    approval_note,
    auth_mode,
    can_approve,
    can_propose,
    password_warning,
    sees_diagnostics,
    starter_prompts,
)
from staffpilot_crew.graph import CrewRunner
from staffpilot_crew.services import CrewServices


NOW = datetime(2026, 11, 16, 12, 0, tzinfo=UTC)


@pytest.fixture
def store(tmp_path):
    # Few hash rounds keep the tests fast; the app uses the full count.
    store = AuthStore(tmp_path / "app.sqlite", iterations=1_000, clock=lambda: NOW)
    store.seed_demo_accounts()
    return store


def test_schema_and_demo_accounts_for_every_role(store, tmp_path):
    assert store.count() == len(SEEDED_ACCOUNTS)
    with sqlite3.connect(tmp_path / "app.sqlite") as db:
        columns = [row[1] for row in db.execute("PRAGMA table_info(users)")]
        roles = {row[0] for row in db.execute("SELECT role FROM users")}
        stored = db.execute("SELECT password_hash, salt FROM users").fetchall()
    assert columns == ["id", "username", "password_hash", "salt", "full_name", "role", "assigned_country", "password_updated_at", "created_at"]
    assert roles == {"admin", "reviewer", "staffing_manager", "country_manager", "finance", "people_team"}
    assert all(len(salt) == 32 for _, salt in stored)
    assert len({salt for _, salt in stored}) == len(stored)
    assert not any(a["password"] in h for h, _ in stored for a in SEEDED_ACCOUNTS)
    # At least one demo account is past the 90 day limit, on purpose.
    assert any(store.get(a["username"]).password_expired(NOW) for a in DEMO_ACCOUNTS)


def test_demo_accounts_are_only_created_once(store):
    assert store.seed_demo_accounts() == 0
    assert store.count() == len(SEEDED_ACCOUNTS)


def test_sign_in_with_right_and_wrong_passwords(store):
    user = store.authenticate("MORGAN ", "Manager2026")
    assert user == store.get("morgan")
    assert user.role == "staffing_manager"
    assert store.authenticate("morgan", "manager2026") is None
    assert store.authenticate("nobody", "Manager2026") is None
    assert store.authenticate("", "") is None


def test_country_manager_has_a_country_and_roles_are_checked(store):
    assert store.get("linh").assigned_country == "Vietnam"
    with pytest.raises(ValueError, match="assigned country"):
        store.create_user("ana", "LongEnough1", "Ana", "country_manager")
    with pytest.raises(ValueError, match="Unknown role"):
        store.create_user("ana", "LongEnough1", "Ana", "superuser")


def test_a_password_of_90_days_or_more_raises_the_warning(store):
    rafael = store.get("rafael")
    assert rafael.password_age_days(NOW) == 120
    warning = password_warning(rafael, NOW)
    assert "120 days ago" in warning and "30 days ago" in warning
    assert password_warning(store.get("morgan"), NOW) is None
    exactly = User("x", "X", "finance", None, (NOW - timedelta(days=90)).isoformat())
    assert exactly.password_expired(NOW)
    assert "due today" in password_warning(exactly, NOW)
    assert not User("y", "Y", "finance", None, (NOW - timedelta(days=89)).isoformat()).password_expired(NOW)


def test_changing_the_password_resets_its_age_and_clears_the_warning(store):
    updated = store.change_password("rafael", "Brazil2026", "NewBrazil2027", "NewBrazil2027")
    assert updated.password_updated_at == NOW.isoformat()
    assert password_warning(updated, NOW) is None
    assert store.authenticate("rafael", "Brazil2026") is None
    assert store.authenticate("rafael", "NewBrazil2027") is not None


@pytest.mark.parametrize(
    ("old", "new", "confirm", "message"),
    [
        ("wrong", "NewBrazil2027", "NewBrazil2027", "current password"),
        ("Brazil2026", "short", "short", "at least 8"),
        ("Brazil2026", "Brazil2026", "Brazil2026", "different"),
        ("Brazil2026", "NewBrazil2027", "NewBrazil2028", "do not match"),
    ],
)
def test_password_rules(store, old, new, confirm, message):
    with pytest.raises(PasswordError, match=message):
        store.change_password("rafael", old, new, confirm)
    assert store.get("rafael").password_expired(NOW)


def test_auth_mode_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("STAFFPILOT_AUTH_MODE", "optional")
    assert auth_mode() == "optional"
    monkeypatch.setenv("STAFFPILOT_AUTH_MODE", "nonsense")
    assert auth_mode() == "required"


def test_each_role_starts_on_its_own_writer_view(store):
    views = {a["username"]: store.get(a["username"]).audience for a in DEMO_ACCOUNTS}
    assert views == {"morgan": "manager", "linh": "country_manager", "rafael": "country_manager", "priya": "finance", "jordan": "people"}


def test_starter_prompts_follow_the_role_and_the_country():
    countries = ["Brazil", "USA", "Vietnam"]
    linh = User("linh", "Linh Tran", "country_manager", "Vietnam")
    prompts = starter_prompts(linh, countries)
    assert 3 <= len(prompts) <= 4
    assert any("Vietnam" in p for p in prompts)
    assert all("Brazil" not in p for p in prompts)
    for role in ("staffing_manager", "finance", "people_team"):
        assert 3 <= len(starter_prompts(User("u", "U", role), countries)) <= 4


def test_only_a_staffing_manager_gives_the_final_approval():
    plain = {"user_constraints": {}}
    budget = {"user_constraints": {"budget_usd": 2_000_000}}
    roles = {r: User(r, r, r, "Vietnam" if r == "country_manager" else None) for r in ("staffing_manager", "country_manager", "finance", "people_team", "reviewer")}
    assert can_approve(roles["staffing_manager"], plain)
    # Finance confirms budgets on a proposal; it no longer gives the final approval.
    assert not can_approve(roles["finance"], budget)
    assert not can_approve(roles["country_manager"], budget)
    assert not can_approve(roles["people_team"], plain)
    assert not can_approve(roles["reviewer"], plain)
    assert not can_approve(GUEST, plain)
    assert approval_note(roles["country_manager"], plain) == APPROVAL_HINT
    assert "confirms the budget" in approval_note(roles["finance"], budget)


def test_who_may_propose_and_who_sees_the_technical_panels():
    assert can_propose(User("m", "M", "staffing_manager"), ["Alpha"])
    assert can_propose(User("c", "C", "country_manager", "Beta"), ["Beta"])
    assert not can_propose(User("c", "C", "country_manager", "Beta"), ["Alpha", "Beta"])
    assert not can_propose(User("f", "F", "finance"), ["Alpha"])
    assert not can_propose(User("r", "R", "reviewer"), ["Alpha"])
    assert sees_diagnostics(User("r", "R", "reviewer"))
    assert sees_diagnostics(User("a", "A", "admin"))
    assert not any(sees_diagnostics(User(r, r, r)) for r in ("staffing_manager", "finance", "people_team"))


def test_approval_is_refused_for_other_roles_and_logged_with_identity(planner, crew_settings):
    runner = CrewRunner(CrewServices.build(planner, crew_settings))
    runner.run("What is the plan for Beta?", thread="roles")
    linh = User("linh", "Linh Tran", "country_manager", "Beta")
    with pytest.raises(PermissionError, match="Staffing Manager"):
        runner.approve("roles", linh)
    assert runner.values("roles")["waiting"]
    # A country manager may still ask for changes.
    runner.ask_for_changes("roles", "no more than 10 hires a month", user=linh)
    morgan = User("morgan", "Morgan Lee", "staffing_manager")
    runner.approve("roles", morgan)
    row = runner.services.log.rows()[0]
    assert (row["username"], row["full_name"], row["role"]) == ("morgan", "Morgan Lee", "staffing_manager")
    with sqlite3.connect(crew_settings.paths.app_db) as db:
        assert db.execute("SELECT COUNT(*) FROM decision_log").fetchone()[0] == 1


def test_finance_cannot_give_the_final_approval_of_a_run(planner, crew_settings):
    runner = CrewRunner(CrewServices.build(planner, crew_settings))
    runner.run("Beta needs to stay within a budget of $300k while serving at least 98%", thread="budget")
    priya = User("priya", "Priya Shah", "finance")
    with pytest.raises(PermissionError, match="confirms the budget"):
        runner.approve("budget", priya)
    assert runner.services.log.rows() == []


def test_a_country_manager_request_without_a_country_uses_theirs(planner, crew_settings):
    runner = CrewRunner(CrewServices.build(planner, crew_settings))
    values = runner.run("Keep hiring to at most 5 people a month and serve at least 98%", thread="home", home_country="Gamma")
    assert values["plan"]["countries"] == ["Gamma"]
    assert values["plan"]["home_country_used"]
    other = runner.run("What is the plan for Alpha?", thread="named", home_country="Gamma")
    assert other["plan"]["countries"] == ["Alpha"]


def test_people_view_lists_hires_and_notices_by_month(planner, crew_settings):
    runner = CrewRunner(CrewServices.build(planner, crew_settings))
    values = runner.run("Hiring notice for Beta", thread="people")
    assert values["audience"] == "people"
    answer = values["answer"]
    assert answer.startswith("## Hiring and notice plan: Beta")
    assert "decide" in answer and "hires; they start the month after" in answer
    assert "1 month after the hiring decision" in answer
    assert values["unverified"] == []


def test_sign_in_form_is_built_without_listing_accounts(planner, crew_settings, store):
    services = CrewServices.build(planner, crew_settings)
    demo = app.build_crew_app(services, CrewRunner(services), "Data: sample.", app.EXAMPLES["sample"], False, store, "required")
    assert demo is not None
    # The sign in page lists no accounts; they are documented in the README only.
    assert not hasattr(app, "demo_accounts_markdown")
    assert "Password older than 3 months" in app.banner_html(store.get("rafael"))
    assert app.banner_html(store.get("morgan")) == ""
    assert app.user_bar(store.get("linh")) == "**Linh Tran** | Country Manager | Vietnam"


def test_sign_in_is_required_unless_disabled(planner, crew_settings):
    services = CrewServices.build(planner, crew_settings)
    with pytest.raises(ValueError, match="AuthStore"):
        app.build_crew_app(services, CrewRunner(services), "", app.EXAMPLES["sample"], False, None, "required")


def test_admin_signs_in_with_the_short_default_password_and_is_asked_to_change_it(store):
    admin = store.authenticate("Admin", "Admin")
    assert admin is not None and admin.role == "admin"
    assert admin.weak_password
    assert "shorter than 8 characters" in password_warning(admin, NOW)
    assert "Change the default password" in app.banner_html(admin)
    # A new admin password follows the usual rules.
    with pytest.raises(PasswordError, match="at least 8"):
        store.change_password("admin", "Admin", "Admin1", "Admin1")
    changed = store.change_password("admin", "Admin", "AdminLonger2026", "AdminLonger2026")
    assert password_warning(store.authenticate("admin", "AdminLonger2026"), NOW) is None
    assert changed.role == "admin"


def test_admin_has_full_rights(store, planner, crew_settings):
    admin = store.authenticate("admin", "Admin")
    assert can_approve(admin, {"user_constraints": {}})
    assert can_approve(admin, {"user_constraints": {"budget_usd": 1}})
    assert approval_note(admin, {}) == ""
    assert admin.audience == "manager"
    assert starter_prompts(admin, ["Alpha", "Beta"]) == starter_prompts(User("m", "M", "staffing_manager"), ["Alpha", "Beta"])
    runner = CrewRunner(CrewServices.build(planner, crew_settings))
    runner.run("What is the plan for Alpha?", thread="admin")
    assert runner.approve("admin", admin)["approval_status"] == "Approved"
    assert runner.services.log.rows()[0]["role"] == "admin"
    frame = app.users_frame(store)
    assert set(frame["Username"]) == {a["username"] for a in SEEDED_ACCOUNTS}
    assert "password_hash" not in " ".join(frame.columns)


def test_admin_is_added_to_a_database_made_before_the_admin_role(tmp_path):
    path = tmp_path / "old.sqlite"
    old_schema = SCHEMA.replace("'admin', ", "")
    with sqlite3.connect(path) as db:
        db.execute(old_schema)
        db.execute(
            "INSERT INTO users (username, password_hash, salt, full_name, role, assigned_country, password_updated_at, created_at) "
            "VALUES ('kevin', 'x', '00', 'Kevin', 'staffing_manager', NULL, ?, ?)",
            (NOW.isoformat(), NOW.isoformat()),
        )
    store = AuthStore(path, iterations=1_000, clock=lambda: NOW)
    assert store.seed_demo_accounts() == 2  # only the admin and the reviewer; the other users are kept
    assert {u.username for u in store.users()} == {"kevin", "admin", "reviewer"}
    assert store.authenticate("admin", "Admin").role == "admin"
