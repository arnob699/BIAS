"""
Code-Persisted User & Administrator Store (`users_store.py`).

All user accounts and the single Administrator account are stored directly inside
this Python source file (`ADMIN_ACCOUNT` and `USERS_LIST`) so that NO external
database (no SQLite, PostgreSQL, or MongoDB) is ever required.

When the Administrator creates, toggles, or deletes a user from the web dashboard,
`save_users_to_code()` rewrites this module's `USERS_LIST` definition in-place.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import pprint
import threading
from typing import Any, Dict, List, Optional

_STORE_LOCK = threading.Lock()

# ---------------------------------------------------------------------------
# Single Administrator Account (Full Access + User Creation Privileges)
# ---------------------------------------------------------------------------
ADMIN_ACCOUNT: Dict[str, Any] = {
    "username": "admin",
    "password": "admin123",
    "display_name": "Administrator (Precious Metals Desk Lead)",
    "role": "admin",
    "active": True,
    "created_at": "2026-10-28 00:00 UTC",
}

# ---------------------------------------------------------------------------
# Code-Persisted Multi-User List (Managed by Administrator via Web UI)
# ---------------------------------------------------------------------------
# BEGIN_USERS_LIST
USERS_LIST: List[Dict[str, Any]] = [
    {
        "username": 'trader1',
        "password": 'gold123',
        "display_name": 'XAUUSD Senior Execution Trader',
        "role": "user",
        "active": True,
        "created_at": '2026-10-28 12:00 UTC',
    },
    {
        "username": 'analyst1',
        "password": 'macro123',
        "display_name": 'COMEX GC=F Macro Analyst',
        "role": "user",
        "active": True,
        "created_at": '2026-10-28 12:05 UTC',
    },
]
# END_USERS_LIST


def get_admin_account() -> Dict[str, Any]:
    """Return a copy of the single Administrator account."""
    return dict(ADMIN_ACCOUNT)


def get_all_users() -> List[Dict[str, Any]]:
    """Return all non-admin user accounts stored in code."""
    with _STORE_LOCK:
        return [dict(u) for u in USERS_LIST]


def get_public_users_roster() -> List[Dict[str, Any]]:
    """Return all accounts (Admin + Users) for the website auth/admin module."""
    with _STORE_LOCK:
        return [dict(ADMIN_ACCOUNT)] + [dict(u) for u in USERS_LIST]


def authenticate_account(username: str, password: str) -> Optional[Dict[str, Any]]:
    """
    Verify username and password against the single `ADMIN_ACCOUNT` or `USERS_LIST`.
    Returns account metadata (without exposing raw password) if valid and active.
    """
    uname = (username or "").strip().lower()
    pwd = (password or "").strip()
    if not uname or not pwd:
        return None

    with _STORE_LOCK:
        if uname == ADMIN_ACCOUNT["username"].lower() and pwd == ADMIN_ACCOUNT["password"]:
            return {
                "username": ADMIN_ACCOUNT["username"],
                "display_name": ADMIN_ACCOUNT["display_name"],
                "role": "admin",
                "active": True,
                "created_at": ADMIN_ACCOUNT.get("created_at", ""),
            }

        for user in USERS_LIST:
            if user.get("username", "").lower() == uname and user.get("password") == pwd:
                if not user.get("active", True):
                    return {"error": "Account is currently disabled by the Administrator."}
                return {
                    "username": user["username"],
                    "display_name": user.get("display_name") or user["username"],
                    "role": "user",
                    "active": True,
                    "created_at": user.get("created_at", ""),
                }

    return None


def _persist_users_list_to_source_file() -> None:
    """
    Rewrite the `# BEGIN_USERS_LIST ... # END_USERS_LIST` block directly inside
    `users_store.py` so that all created/updated users are permanently saved in code.
    """
    file_path = Path(__file__).resolve()
    source_text = file_path.read_text(encoding="utf-8")

    formatted_entries: List[str] = ["USERS_LIST: List[Dict[str, Any]] = ["]
    for u in USERS_LIST:
        formatted_entries.append("    {")
        formatted_entries.append(f'        "username": {u["username"]!r},')
        formatted_entries.append(f'        "password": {u["password"]!r},')
        formatted_entries.append(f'        "display_name": {u.get("display_name", u["username"])!r},')
        formatted_entries.append(f'        "role": "user",')
        formatted_entries.append(f'        "active": {bool(u.get("active", True))!r},')
        formatted_entries.append(f'        "created_at": {u.get("created_at", "")!r},')
        formatted_entries.append("    },")
    formatted_entries.append("]")

    replacement_block = (
        "# BEGIN_USERS_LIST\n"
        + "\n".join(formatted_entries)
        + "\n# END_USERS_LIST"
    )

    marker_start = "# BEGIN_USERS_LIST"
    marker_end = "# END_USERS_LIST"
    start_idx = source_text.find(marker_start)
    end_idx = source_text.find(marker_end)

    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        updated_source = (
            source_text[:start_idx]
            + replacement_block
            + source_text[end_idx + len(marker_end) :]
        )
        file_path.write_text(updated_source, encoding="utf-8")


def admin_create_or_update_user(
    username: str,
    password: str,
    display_name: str = "",
    active: bool = True,
) -> Dict[str, Any]:
    """
    Create a new user (or update an existing user) and save directly into `users_store.py`.
    Cannot overwrite the single `admin` account username.
    """
    clean_user = (username or "").strip()
    clean_pass = (password or "").strip()
    clean_name = (display_name or "").strip() or clean_user

    if not clean_user or len(clean_user) < 2:
        raise ValueError("Username must be at least 2 characters.")
    if not clean_pass or len(clean_pass) < 3:
        raise ValueError("Password must be at least 3 characters.")
    if clean_user.lower() == ADMIN_ACCOUNT["username"].lower():
        raise ValueError("Cannot overwrite the single Administrator username ('admin').")

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    with _STORE_LOCK:
        for existing in USERS_LIST:
            if existing.get("username", "").lower() == clean_user.lower():
                existing["username"] = clean_user
                existing["password"] = clean_pass
                existing["display_name"] = clean_name
                existing["active"] = bool(active)
                _persist_users_list_to_source_file()
                return dict(existing)

        new_user: Dict[str, Any] = {
            "username": clean_user,
            "password": clean_pass,
            "display_name": clean_name,
            "role": "user",
            "active": bool(active),
            "created_at": now_str,
        }
        USERS_LIST.append(new_user)
        _persist_users_list_to_source_file()
        return dict(new_user)


def admin_toggle_user_status(username: str) -> Dict[str, Any]:
    """Toggle a user's `active` status and persist directly into `users_store.py`."""
    clean_user = (username or "").strip().lower()
    if clean_user == ADMIN_ACCOUNT["username"].lower():
        raise ValueError("The Administrator account cannot be disabled.")

    with _STORE_LOCK:
        for existing in USERS_LIST:
            if existing.get("username", "").lower() == clean_user:
                existing["active"] = not bool(existing.get("active", True))
                _persist_users_list_to_source_file()
                return dict(existing)

    raise ValueError(f"User '{username}' not found.")


def admin_delete_user(username: str) -> bool:
    """Remove a user from `USERS_LIST` and persist directly into `users_store.py`."""
    clean_user = (username or "").strip().lower()
    if clean_user == ADMIN_ACCOUNT["username"].lower():
        raise ValueError("The single Administrator account cannot be deleted.")

    with _STORE_LOCK:
        initial_len = len(USERS_LIST)
        USERS_LIST[:] = [
            u for u in USERS_LIST if u.get("username", "").lower() != clean_user
        ]
        if len(USERS_LIST) == initial_len:
            raise ValueError(f"User '{username}' not found.")
        _persist_users_list_to_source_file()
        return True
