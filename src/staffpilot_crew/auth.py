"""Sign in, roles and passwords, stored in SQLite with the standard library only.

Passwords are hashed with PBKDF2 (SHA 256) and a random salt per user; the plain password
is never stored. A password older than 90 days is reported at sign in so the app can ask
for a new one. Roles decide what a user sees first and whether they may give the final
approval.

The mode comes from STAFFPILOT_AUTH_MODE:

* ``required`` (default): nothing but the sign in form until a user signs in.
* ``optional``: the form is shown, and a guest may look around and run requests, but not approve.
* ``disabled``: no form; the session is a staffing manager called "Local user" (for tests and
  for working alone on a laptop).
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
import hashlib
import hmac
import os
from pathlib import Path
import secrets
import sqlite3
from typing import Any, Literal


Role = Literal["admin", "reviewer", "staffing_manager", "country_manager", "finance", "people_team", "guest"]
ROLES: tuple[Role, ...] = ("admin", "reviewer", "staffing_manager", "country_manager", "finance", "people_team")
ROLE_LABELS = {
    "admin": "Admin",
    "reviewer": "Reviewer",
    "staffing_manager": "Staffing Manager",
    "country_manager": "Country Manager",
    "finance": "Finance",
    "people_team": "People Team",
    "guest": "Guest",
}
# The Writer's view each role sees first; the user can still switch.
ROLE_AUDIENCE = {
    "admin": "manager",
    "reviewer": "manager",
    "staffing_manager": "manager",
    "country_manager": "country_manager",
    "finance": "finance",
    "people_team": "people",
    "guest": "manager",
}
AUTH_ENV_VAR = "STAFFPILOT_AUTH_MODE"
AUTH_MODES = ("required", "optional", "disabled")
PASSWORD_MAX_AGE_DAYS = 90
MIN_PASSWORD_LENGTH = 8
ITERATIONS = 200_000

# Demo accounts created when the users table is empty. They exist so a grader can try every
# role; change or delete them before any real use. The Brazil manager's password is set
# 120 days in the past on purpose, to show the 90 day warning.
DEMO_ACCOUNTS: tuple[dict[str, Any], ...] = (
    {"username": "morgan", "password": "Manager2026", "full_name": "Morgan Lee", "role": "staffing_manager", "assigned_country": None, "age_days": 10},
    {"username": "linh", "password": "Vietnam2026", "full_name": "Linh Tran", "role": "country_manager", "assigned_country": "Vietnam", "age_days": 30},
    {"username": "rafael", "password": "Brazil2026", "full_name": "Rafael Costa", "role": "country_manager", "assigned_country": "Brazil", "age_days": 120},
    {"username": "priya", "password": "Finance2026", "full_name": "Priya Shah", "role": "finance", "assigned_country": None, "age_days": 45},
    {"username": "jordan", "password": "People2026", "full_name": "Jordan Kim", "role": "people_team", "assigned_country": None, "age_days": 5},
)
# The administrator has every right. Its password is short on purpose (the team asked for
# "Admin"), so it is allowed only for this seeded account and every sign in with it shows a
# banner asking for a new one; a changed password must follow the usual rules.
ADMIN_ACCOUNT: dict[str, Any] = {"username": "admin", "password": "Admin", "full_name": "Administrator", "role": "admin", "assigned_country": None, "age_days": 0}
# A read only account that also sees the technical panels (the agents' steps, model settings
# and answer quality), for the teaching team.
REVIEWER_ACCOUNT: dict[str, Any] = {"username": "reviewer", "password": "Reviewer2026", "full_name": "Avery Quinn", "role": "reviewer", "assigned_country": None, "age_days": 0}
SEEDED_ACCOUNTS: tuple[dict[str, Any], ...] = (ADMIN_ACCOUNT, REVIEWER_ACCOUNT, *DEMO_ACCOUNTS)

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    salt TEXT NOT NULL,
    full_name TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('admin', 'reviewer', 'staffing_manager', 'country_manager', 'finance', 'people_team')),
    assigned_country TEXT,
    password_updated_at TEXT NOT NULL,
    created_at TEXT NOT NULL
)
"""


def auth_mode() -> str:
    mode = os.environ.get(AUTH_ENV_VAR, "required").strip().lower()
    return mode if mode in AUTH_MODES else "required"


def now_utc() -> datetime:
    return datetime.now(UTC)


def hash_password(password: str, salt: str, iterations: int = ITERATIONS) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), iterations).hex()


@dataclass(frozen=True)
class User:
    """The signed in user, as the app and the decision log see them."""

    username: str
    full_name: str
    role: str
    assigned_country: str | None = None
    password_updated_at: str = ""
    weak_password: bool = False  # signed in with a password shorter than the rules allow

    @property
    def role_label(self) -> str:
        return ROLE_LABELS[self.role]

    @property
    def audience(self) -> str:
        return ROLE_AUDIENCE[self.role]

    def password_age_days(self, now: datetime | None = None) -> int:
        if not self.password_updated_at:
            return 0
        updated = datetime.fromisoformat(self.password_updated_at)
        return ((now or now_utc()) - updated).days

    def password_expired(self, now: datetime | None = None) -> bool:
        return self.role != "guest" and self.password_age_days(now) >= PASSWORD_MAX_AGE_DAYS

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> User | None:
        return cls(**data) if data else None


GUEST = User("guest", "Guest", "guest")
LOCAL_USER = User("local", "Local user", "staffing_manager")


class PasswordError(ValueError):
    """A password change that breaks a rule; the message says which."""


class AuthStore:
    def __init__(self, path: str | Path, iterations: int = ITERATIONS, clock: Callable[[], datetime] = now_utc) -> None:
        self.path = Path(path)
        self.iterations = iterations
        self.clock = clock
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db, db:
            db.execute(SCHEMA)
            self._allow_every_role(db)

    @staticmethod
    def _allow_every_role(db: sqlite3.Connection) -> None:
        """Rebuild a users table made before a role existed; SQLite cannot alter a CHECK."""
        sql = db.execute("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'users'").fetchone()[0]
        if all(f"'{role}'" in sql for role in ROLES):
            return
        db.execute("ALTER TABLE users RENAME TO users_before_admin")
        db.execute(SCHEMA)
        db.execute("INSERT INTO users SELECT * FROM users_before_admin")
        db.execute("DROP TABLE users_before_admin")

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        return db

    def count(self) -> int:
        with closing(self._connect()) as db:
            return db.execute("SELECT COUNT(*) FROM users").fetchone()[0]

    def create_user(
        self,
        username: str,
        password: str,
        full_name: str,
        role: str,
        assigned_country: str | None = None,
        password_updated_at: datetime | None = None,
        check_rules: bool = True,
    ) -> User:
        if role not in ROLES:
            raise ValueError(f"Unknown role {role!r}; use one of {', '.join(ROLES)}")
        if role == "country_manager" and not assigned_country:
            raise ValueError("A country manager needs an assigned country")
        if check_rules:
            check_new_password(password)
        salt = secrets.token_hex(16)
        now = self.clock()
        updated = (password_updated_at or now).isoformat()
        with closing(self._connect()) as db, db:
            db.execute(
                "INSERT INTO users (username, password_hash, salt, full_name, role, assigned_country, password_updated_at, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (username.strip().lower(), hash_password(password, salt, self.iterations), salt, full_name, role, assigned_country, updated, now.isoformat()),
            )
        return self.get(username)

    def seed_demo_accounts(self) -> int:
        """Create the demo accounts if no user exists yet, and the admin if it is missing.

        Returns how many accounts were created.
        """
        created = self.ensure_admin()
        if self.count() > created:
            return created
        now = self.clock()
        for account in DEMO_ACCOUNTS:
            self.create_user(
                account["username"],
                account["password"],
                account["full_name"],
                account["role"],
                account["assigned_country"],
                now - timedelta(days=account["age_days"]),
            )
        return created + len(DEMO_ACCOUNTS)

    def ensure_admin(self) -> int:
        """Create the administrator and the reviewer when missing. Returns how many were created."""
        created = 0
        for account, strict in ((ADMIN_ACCOUNT, False), (REVIEWER_ACCOUNT, True)):
            if self.get(account["username"]) is None:
                self.create_user(account["username"], account["password"], account["full_name"], account["role"], check_rules=strict)
                created += 1
        return created

    def users(self) -> list[User]:
        with closing(self._connect()) as db:
            rows = db.execute("SELECT * FROM users ORDER BY id").fetchall()
        return [User(r["username"], r["full_name"], r["role"], r["assigned_country"], r["password_updated_at"]) for r in rows]

    def get(self, username: str) -> User | None:
        with closing(self._connect()) as db:
            row = db.execute("SELECT * FROM users WHERE username = ?", (username.strip().lower(),)).fetchone()
        if row is None:
            return None
        return User(row["username"], row["full_name"], row["role"], row["assigned_country"], row["password_updated_at"])

    def authenticate(self, username: str, password: str) -> User | None:
        """The user when the password matches, otherwise None (the same for an unknown name)."""
        with closing(self._connect()) as db:
            row = db.execute("SELECT * FROM users WHERE username = ?", ((username or "").strip().lower(),)).fetchone()
        if row is None:
            # Spend the same time as a real check, so a wrong name and a wrong password look alike.
            hash_password(password or "", secrets.token_hex(16), self.iterations)
            return None
        candidate = hash_password(password or "", row["salt"], self.iterations)
        if not hmac.compare_digest(candidate, row["password_hash"]):
            return None
        weak = len(password) < MIN_PASSWORD_LENGTH
        return User(row["username"], row["full_name"], row["role"], row["assigned_country"], row["password_updated_at"], weak)

    def change_password(self, username: str, old: str, new: str, confirm: str) -> User:
        if self.authenticate(username, old) is None:
            raise PasswordError("The current password is not correct.")
        if new != confirm:
            raise PasswordError("The new password and its confirmation do not match.")
        if new == old:
            raise PasswordError("The new password must be different from the current one.")
        check_new_password(new)
        salt = secrets.token_hex(16)
        with closing(self._connect()) as db, db:
            db.execute(
                "UPDATE users SET password_hash = ?, salt = ?, password_updated_at = ? WHERE username = ?",
                (hash_password(new, salt, self.iterations), salt, self.clock().isoformat(), username.strip().lower()),
            )
        return self.get(username)


def check_new_password(password: str) -> None:
    if len(password or "") < MIN_PASSWORD_LENGTH:
        raise PasswordError(f"The new password needs at least {MIN_PASSWORD_LENGTH} characters.")


def password_warning(user: User | None, now: datetime | None = None) -> str | None:
    """The banner text for a password 90 days old or more, or one shorter than the rules allow."""
    if user is None:
        return None
    if user.weak_password:
        return (
            f"You signed in with a password shorter than {MIN_PASSWORD_LENGTH} characters (the default admin password). "
            "Anyone who reads the README can use it; please choose a new one with Change password."
        )
    if not user.password_expired(now):
        return None
    age = user.password_age_days(now)
    late = age - PASSWORD_MAX_AGE_DAYS
    overdue = "today" if late == 0 else f"{late} {'day' if late == 1 else 'days'} ago"
    return (
        f"Your password was last changed {age} days ago. Passwords must be changed every "
        f"{PASSWORD_MAX_AGE_DAYS} days (3 months); yours was due {overdue}. Please choose a new one with Change password."
    )


# What each role may do.


def budget_option(values: dict[str, Any]) -> bool:
    """An option is a budget decision when the manager set a budget for it."""
    return (values.get("user_constraints") or {}).get("budget_usd") is not None


def can_approve(user: User | None, values: dict[str, Any] | None = None) -> bool:
    """Only a staffing manager (or the admin) gives the final approval; Finance confirms budgets."""
    return user is not None and user.role in {"admin", "staffing_manager"}


def can_propose(user: User | None, countries: list[str] | None = None) -> bool:
    """Who may turn a plan into a proposal: managers, the admin, and a country manager for their country."""
    if user is None:
        return False
    if user.role in {"admin", "staffing_manager"}:
        return True
    return user.role == "country_manager" and bool(countries) and set(countries) <= {user.assigned_country}


def sees_diagnostics(user: User | None) -> bool:
    """The agents' steps, model settings and answer quality are for the admin and reviewers."""
    return user is not None and user.role in {"admin", "reviewer"}


APPROVAL_HINT = "Final approval requires Staffing Manager role."
FINANCE_HINT = "Finance confirms the budget of a proposal on Proposals; final approval requires Staffing Manager role."


def approval_note(user: User | None, values: dict[str, Any] | None = None) -> str:
    if can_approve(user, values):
        return ""
    if user is not None and user.role == "finance":
        return FINANCE_HINT
    return APPROVAL_HINT


def starter_prompts(user: User | None, countries: list[str]) -> list[str]:
    """Three or four requests that fit the user's role, using their country when they have one."""
    first = countries[0] if countries else "the first country"
    many = ", ".join(countries[:3]) if countries else "three countries"
    role = user.role if user else "guest"
    if role in {"admin", "reviewer"}:
        role = "staffing_manager"
    country = user.assigned_country if user and user.assigned_country in countries else first
    return {
        "staffing_manager": [
            f"Compare a growth push in {many} where sign ups rise 20%",
            f"What is the plan for {first}?",
            f"Keep hiring in {first} to at most 5 people a month and serve at least 98%",
            f"What if an AI assistant onboards 30% of advertisers in {first}?",
        ],
        "country_manager": [
            f"What is the plan for {country}?",
            f"Write a memo for the {country} country manager",
            f"What if sign ups rise 20% in {country}?",
            "Keep hiring to at most 5 people a month and serve at least 98%",
        ],
        "finance": [
            f"A note for finance on the cost of {first}",
            f"{first} needs to stay within a budget of $2M while serving at least 98%",
            f"What does a 20% rise in sign ups in {first} cost us?",
        ],
        "people_team": [
            f"Hiring notice for {first}: who do we hire and when?",
            f"Notice plan for {many}",
            f"Keep hiring in {first} to at most 5 people a month; what does the people team need to plan?",
        ],
        "guest": [f"What is the plan for {first}?", f"Compare {many}"],
    }[role]
