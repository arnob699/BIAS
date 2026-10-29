"""
Standalone Multi-User Web Server for the Gold (`XAUUSD` / COMEX `GC=F`)
Intraday Sentiment & Economic Calendar Terminal (`app.py`).

Key Features:
1. Works 100% whether `GEMINI_API_KEY` / `GROQ_API_KEY` are online, rate-limited,
   or completely offline (automatically falls back to the built-in Quantitative
   Gold Correlation Engine and offline snapshot cache).
2. Single Administrator (`arnob` / `arnob69`) can create, enable/disable, and
   delete multiple User accounts directly from the Web UI.
3. Zero Database Required: All users are persisted directly inside the Python
   code file `users_store.py` (and synced into `xauusd_intraday_report.html`).
"""

from __future__ import annotations

import os
import subprocess
import sys

# Auto-re-execute inside the project .venv if invoked using a global/external python
_script_dir = os.path.dirname(os.path.abspath(__file__))
_venv_python = os.path.join(_script_dir, ".venv", "Scripts", "python.exe")
if os.path.isfile(_venv_python) and os.path.normcase(sys.executable) != os.path.normcase(_venv_python):
    try:
        _sub_proc = subprocess.run([_venv_python] + sys.argv, check=False)
        sys.exit(_sub_proc.returncode)
    except Exception:
        pass

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
from pathlib import Path
import re
import time
from typing import Any, Dict, Optional
import warnings
import webbrowser

warnings.filterwarnings("ignore", message=".*inplace keyword.*")
warnings.filterwarnings("ignore", module="yfinance.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

from analyzer import analyze_gold_market, build_deterministic_gold_analysis
from config import GEMINI_API_KEY, GEMINI_MODEL, GROQ_API_KEY, TARGET_TICKERS
from data_fetcher import fetch_market_intelligence_payload
from main import _build_demo_gold_payload_and_analysis
from report_generator import DEFAULT_REPORT_FILENAME, generate_html_report
from users_store import (
    admin_create_or_update_user,
    admin_delete_user,
    admin_toggle_user_status,
    authenticate_account,
    get_admin_account,
    get_all_users,
)

logger = logging.getLogger(__name__)

_REPORT_PATH = Path(_script_dir) / DEFAULT_REPORT_FILENAME
_LAST_PAYLOAD_CACHE: Optional[Dict[str, Any]] = None
_LAST_ANALYSIS_CACHE: Optional[Dict[str, Any]] = None
_REPORT_VERSION: int = int(time.time() * 1000)
_IS_REFRESHING: bool = False


def refresh_gold_terminal_report(force_quant_only: bool = False) -> str:
    """
    Fetch live Gold (`XAUUSD` / `GC=F`), `DXY`, `^TNX`, `^VIX`, `GLD`, 18 breaking
    headlines, and the Gold-Only Economic Calendar, and regenerate
    `xauusd_intraday_report.html`.

    Guaranteed to succeed whether API keys are online, rate-limited, or offline.
    """
    global _LAST_PAYLOAD_CACHE, _LAST_ANALYSIS_CACHE, _REPORT_VERSION, _IS_REFRESHING
    _IS_REFRESHING = True

    try:
        logger.info("Fetching live Gold market snapshot, 18 headlines & Forex Factory Gold Calendar...")
        payload = fetch_market_intelligence_payload(target_tickers=TARGET_TICKERS)
    except Exception as exc:
        logger.warning("Live market fetch encountered error (%s). Using fallback dataset...", exc)
        if _LAST_PAYLOAD_CACHE is not None and _LAST_ANALYSIS_CACHE is not None:
            return generate_html_report(_LAST_PAYLOAD_CACHE, _LAST_ANALYSIS_CACHE, str(_REPORT_PATH))
        payload, demo_analysis = _build_demo_gold_payload_and_analysis()
        _LAST_PAYLOAD_CACHE = payload
        _LAST_ANALYSIS_CACHE = demo_analysis
        return generate_html_report(payload, demo_analysis, str(_REPORT_PATH))

    has_any_api_key = bool(
        (GEMINI_API_KEY and GEMINI_API_KEY != "your_gemini_api_key_here")
        or (GROQ_API_KEY and GROQ_API_KEY != "your_groq_api_key_here")
    )

    if force_quant_only or not has_any_api_key:
        logger.info("Running built-in Quantitative Gold Engine (API-Free Mode)...")
        analysis = build_deterministic_gold_analysis(payload)
        analysis["analysis_engine"] = "quantitative_gold_engine (API-Free Mode — Always Online)"
    else:
        try:
            analysis = analyze_gold_market(
                payload,
                model_name=GEMINI_MODEL,
                allow_fallback_on_missing_key=True,
            )
        except Exception as exc:
            logger.warning("LLM API unavailable (%s). Failing over to Quantitative Gold Engine...", exc)
            analysis = build_deterministic_gold_analysis(payload)
            analysis["analysis_engine"] = "quantitative_gold_engine (Auto-Fallback — API Offline)"

    _LAST_PAYLOAD_CACHE = payload
    _LAST_ANALYSIS_CACHE = analysis
    try:
        out_path = generate_html_report(payload, analysis, str(_REPORT_PATH))
        _REPORT_VERSION = int(time.time() * 1000)
        return out_path
    finally:
        _IS_REFRESHING = False


def _sync_users_into_existing_html() -> None:
    """
    Inject the latest `ADMIN_ACCOUNT` and `USERS_LIST` from `users_store.py`
    into `xauusd_intraday_report.html` without needing a full market re-fetch.
    """
    if _LAST_PAYLOAD_CACHE is not None and _LAST_ANALYSIS_CACHE is not None:
        generate_html_report(_LAST_PAYLOAD_CACHE, _LAST_ANALYSIS_CACHE, str(_REPORT_PATH))
        return

    if not _REPORT_PATH.is_file():
        refresh_gold_terminal_report(force_quant_only=True)
        return

    html_text = _REPORT_PATH.read_text(encoding="utf-8")
    portal_json = json.dumps(
        {
            "admin": get_admin_account(),
            "users": get_all_users(),
        }
    )
    updated_html = re.sub(
        r"const initialPortalAuth = \{.*?\};",
        lambda _: f"const initialPortalAuth = {portal_json};",
        html_text,
        count=1,
        flags=re.DOTALL,
    )
    _REPORT_PATH.write_text(updated_html, encoding="utf-8")


class GoldPortalHTTPRequestHandler(BaseHTTPRequestHandler):
    """HTTP handler serving the Gold Web Terminal and Code-Persisted Admin/User APIs."""

    def log_message(self, format: str, *args: Any) -> None:
        logger.info("%s - %s", self.address_string(), format % args)

    def _send_json(self, status_code: int, data: Dict[str, Any]) -> None:
        raw = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def _read_json_body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8", errors="replace")
        try:
            return json.loads(raw)
        except Exception:
            return {}

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]

        if path in ("/health", "/healthz", "/api/status"):
            self._send_json(
                200,
                {
                    "status": "ok",
                    "service": "xauusd-gold-portal",
                    "report_version": _REPORT_VERSION,
                    "is_refreshing": _IS_REFRESHING,
                    "last_updated_bst": (_LAST_PAYLOAD_CACHE or {}).get("generated_at_bst", ""),
                },
            )
            return

        if path == "/api/users":
            self._send_json(
                200,
                {
                    "admin": get_admin_account(),
                    "users": get_all_users(),
                    "storage_file": "users_store.py",
                },
            )
            return

        if path in ("/", "/index.html", f"/{DEFAULT_REPORT_FILENAME}"):
            if not _REPORT_PATH.is_file():
                refresh_gold_terminal_report(force_quant_only=False)
            else:
                _sync_users_into_existing_html()

            content = _REPORT_PATH.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(content)
            return

        self._send_json(404, {"error": "Not Found"})

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        body = self._read_json_body()

        if path == "/api/login":
            username = body.get("username", "")
            password = body.get("password", "")
            auth = authenticate_account(username, password)
            if auth is None:
                self._send_json(401, {"error": "Invalid username or password."})
                return
            if "error" in auth:
                self._send_json(403, {"error": auth["error"]})
                return
            self._send_json(200, {"status": "ok", "user": auth})
            return

        if path == "/api/users":
            action = (body.get("action") or "create_or_update").strip().lower()
            try:
                if action == "create_or_update":
                    user_obj = admin_create_or_update_user(
                        username=body.get("username", ""),
                        password=body.get("password", ""),
                        display_name=body.get("display_name", ""),
                        active=body.get("active", True),
                    )
                    _sync_users_into_existing_html()
                    self._send_json(
                        200,
                        {
                            "status": "ok",
                            "user": user_obj,
                            "admin": get_admin_account(),
                            "users": get_all_users(),
                        },
                    )
                    return

                if action == "toggle":
                    user_obj = admin_toggle_user_status(body.get("username", ""))
                    _sync_users_into_existing_html()
                    self._send_json(
                        200,
                        {
                            "status": "ok",
                            "user": user_obj,
                            "admin": get_admin_account(),
                            "users": get_all_users(),
                        },
                    )
                    return

                if action == "delete":
                    admin_delete_user(body.get("username", ""))
                    _sync_users_into_existing_html()
                    self._send_json(
                        200,
                        {
                            "status": "ok",
                            "admin": get_admin_account(),
                            "users": get_all_users(),
                        },
                    )
                    return

                self._send_json(400, {"error": f"Unknown action '{action}'"})
            except ValueError as exc:
                self._send_json(400, {"error": str(exc)})
            except Exception as exc:
                self._send_json(500, {"error": f"Failed to update users_store.py: {exc}"})
            return

        if path == "/api/refresh":
            force_quant = bool(body.get("fetch_only", False))
            try:
                report_file = refresh_gold_terminal_report(force_quant_only=force_quant)
                self._send_json(
                    200,
                    {
                        "status": "ok",
                        "report_path": report_file,
                    },
                )
            except Exception as exc:
                self._send_json(500, {"error": str(exc)})
            return

        self._send_json(404, {"error": "Endpoint not found"})


def run_web_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    open_browser: bool = False,
    refresh_on_start: bool = False,
) -> None:
    """Start the multi-user Gold (`XAUUSD` / `GC=F`) Web Portal server."""
    needs_ai_refresh = refresh_on_start or not _REPORT_PATH.is_file()
    if not needs_ai_refresh and _REPORT_PATH.is_file():
        try:
            existing_html = _REPORT_PATH.read_text(encoding="utf-8", errors="ignore")
            if "fetch_only_deterministic" in existing_html or "deterministic_fallback" in existing_html:
                needs_ai_refresh = True
        except Exception:
            pass

    if needs_ai_refresh:
        refresh_gold_terminal_report(force_quant_only=False)
    else:
        _sync_users_into_existing_html()
        import threading
        threading.Thread(
            target=lambda: refresh_gold_terminal_report(force_quant_only=False),
            daemon=True,
        ).start()

    admin_acc = get_admin_account()
    users_list = get_all_users()
    local_url = f"http://127.0.0.1:{port}"

    print("=" * 88)
    print(" XAUUSD / COMEX GOLD (GC=F) MULTI-USER WEB PORTAL LIVE")
    print("=" * 88)
    print(f"  • Web Portal URL        : {local_url}")
    print(f"  • Network / LAN Access  : http://{host}:{port}")
    print(f"  • API Offline Resilience: ENABLED (Auto-Fallback to Quantitative Gold Engine)")
    print(f"  • User Storage Mode     : Saved in Python Code (`users_store.py` — Zero Database)")
    print(f"  • Single Administrator  : username='{admin_acc['username']}' | password='{admin_acc['password']}'")
    print(f"  • Active Users in Code  : {len(users_list)} ({', '.join(u['username'] for u in users_list)})")
    print("=" * 88)
    print(" Press Ctrl+C to stop the web server.\n")

    if open_browser:
        try:
            webbrowser.open(local_url)
        except Exception:
            pass

    server = ThreadingHTTPServer((host, port), GoldPortalHTTPRequestHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down Gold Web Portal server...")
    finally:
        server.server_close()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    default_port = int(os.getenv("PORT", "8000"))
    parser = argparse.ArgumentParser(
        description="Gold (XAUUSD / GC=F) Multi-User Web Portal Server (No Database Required)"
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=default_port, help=f"Port number (default: {default_port})")
    parser.add_argument("--open", action="store_true", help="Open http://127.0.0.1:8000 in browser")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Fetch fresh market & news data before starting the web server",
    )
    args = parser.parse_args()

    run_web_server(
        host=args.host,
        port=args.port,
        open_browser=args.open,
        refresh_on_start=args.refresh,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

