"""
Standalone Gold (`XAUUSD` / COMEX `GC=F`) Terminal HTML Report Generator (`report_generator.py`).

Renders `xauusd_intraday_report.html` using Jinja2, Tailwind CSS CDN, and Chart.js CDN
with an institutional dark-mode Gold trading terminal aesthetic (`#0b0f19` background,
metallic gold `#f59e0b`, emerald `#10b981`, and crimson `#ef4444` accents).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from jinja2 import BaseLoader, Environment

from users_store import get_admin_account, get_all_users

DEFAULT_REPORT_FILENAME = "xauusd_intraday_report.html"


def _derive_display_bias_badge(overall_bias: str) -> Dict[str, str]:
    """
    Map `overall_bias` from `GoldAnalysisReport` to the Hero Banner badge label
    (`STRONG BUY XAUUSD`, `BUY XAUUSD`, `NEUTRAL`, `SELL XAUUSD`, `STRONG SELL XAUUSD`)
    and corresponding Tailwind color classes.
    """
    normalized = (overall_bias or "NEUTRAL / CHOPPY").upper().strip()
    if normalized == "STRONG BUY":
        return {
            "label": "STRONG BUY XAUUSD",
            "raw": normalized,
            "bg_class": "bg-emerald-500/20 border-emerald-400 text-emerald-300 shadow-emerald-500/20",
            "dot_class": "bg-emerald-400",
            "accent_hex": "#10b981",
            "bar_class": "from-emerald-600 to-emerald-400",
        }
    if normalized == "BUY":
        return {
            "label": "BUY XAUUSD",
            "raw": normalized,
            "bg_class": "bg-emerald-500/15 border-emerald-500/80 text-emerald-300 shadow-emerald-500/10",
            "dot_class": "bg-emerald-400",
            "accent_hex": "#10b981",
            "bar_class": "from-emerald-600 to-emerald-400",
        }
    if normalized == "STRONG SELL":
        return {
            "label": "STRONG SELL XAUUSD",
            "raw": normalized,
            "bg_class": "bg-red-500/20 border-red-400 text-red-300 shadow-red-500/20",
            "dot_class": "bg-red-400",
            "accent_hex": "#ef4444",
            "bar_class": "from-red-600 to-red-400",
        }
    if normalized == "SELL":
        return {
            "label": "SELL XAUUSD",
            "raw": normalized,
            "bg_class": "bg-red-500/15 border-red-500/80 text-red-300 shadow-red-500/10",
            "dot_class": "bg-red-400",
            "accent_hex": "#ef4444",
            "bar_class": "from-red-600 to-red-400",
        }
    return {
        "label": "NEUTRAL",
        "raw": normalized,
        "bg_class": "bg-amber-500/15 border-amber-400/80 text-amber-300 shadow-amber-500/10",
        "dot_class": "bg-amber-400",
        "accent_hex": "#f59e0b",
        "bar_class": "from-amber-600 to-amber-400",
    }


def _derive_confluence_status(
    analysis: Dict[str, Any],
    payload: Dict[str, Any],
) -> Dict[str, str]:
    """
    Build the Confluence Status tag for the Hero Banner based on `confidence_score`,
    wedge alignment, and cross-asset driver alignment.
    """
    conf = int(analysis.get("confidence_score", 50))
    wedges = analysis.get("wedge_breakdown") or []
    w_scores = {w.get("weight_percentage"): float(w.get("net_sentiment_score", 0.0)) for w in wedges}
    w1_score = w_scores.get(40, 0.0)
    w3_score = w_scores.get(20, 0.0)

    tickers = ((payload.get("gold_macro_snapshot") or {}).get("tickers")) or {}
    tnx = tickers.get("US_10Y_YIELD") or {}
    dxy = tickers.get("DXY_DOLLAR_INDEX") or {}
    tnx_price = tnx.get("price") or 0.0
    tnx_chg = tnx.get("change_pct") or 0.0
    dxy_price = dxy.get("price") or 0.0
    dxy_chg = dxy.get("change_pct") or 0.0

    if conf >= 80:
        return {
            "label": "High Confluence: Yields, DXY & News Aligned",
            "badge_class": "bg-emerald-500/15 text-emerald-300 border-emerald-500/40",
            "description": "Macro rates (^TNX), US Dollar (DXY), and high-weight breaking catalysts point in the same direction for XAUUSD.",
        }
    if w1_score < -10 and w3_score > 15:
        return {
            "label": "Hawkish Yield Pressure vs. Geopolitical Support",
            "badge_class": "bg-amber-500/15 text-amber-300 border-amber-500/40",
            "description": (
                f"Elevated US 10Y yields (^TNX {tnx_price:.3f}%, {tnx_chg:+.2f}%) and Fed rate hold weigh on Gold "
                f"(DXY {dxy_price:.2f}, {dxy_chg:+.2f}%), partially buffered by geopolitical safe-haven flows."
            ),
        }
    if w1_score > 10 and w3_score < -10:
        return {
            "label": "Mixed Signals: Dovish Rates vs. Fading Geopolitical Premium",
            "badge_class": "bg-amber-500/15 text-amber-300 border-amber-500/40",
            "description": "Supportive yield/USD backdrop offset by easing safe-haven risk premia.",
        }
    if conf >= 65:
        return {
            "label": "Moderate Confluence: Directional Tilt Forming",
            "badge_class": "bg-sky-500/15 text-sky-300 border-sky-500/40",
            "description": "Primary macro wedges favor a directional bias, with secondary cross-currents requiring disciplined sizing.",
        }
    return {
        "label": "Cross-Current Regime: Two-Way Macro & News Conflict",
        "badge_class": "bg-amber-500/15 text-amber-300 border-amber-500/40",
        "description": "Conflicting signals across Fed/Yields (Wedge 1) and Safe-Haven/Flow catalysts keep XAUUSD rotational.",
    }


def _prepare_report_view_model(
    payload: Dict[str, Any],
    analysis: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Prepare enriched template variables, sorted news rows, and serialized Chart.js
    datasets for `xauusd_intraday_report.html`.
    """
    bias_meta = _derive_display_bias_badge(analysis.get("overall_bias", "NEUTRAL / CHOPPY"))
    confluence_meta = _derive_confluence_status(analysis, payload)

    # Separate primary thesis bullets from the #1 invalidation trigger bullet if marked
    raw_bullets: List[str] = list(analysis.get("why_buy_or_sell_thesis") or [])
    thesis_bullets: List[str] = []
    invalidation_bullet: Optional[str] = None

    for b in raw_bullets:
        if "invalidation" in b.lower() or "trigger" in b.lower():
            if invalidation_bullet is None:
                invalidation_bullet = b
            else:
                thesis_bullets.append(b)
        else:
            thesis_bullets.append(b)

    if invalidation_bullet is None and thesis_bullets:
        invalidation_bullet = thesis_bullets.pop(-1)

    # Map URLs from raw `intraday_breaking_news` onto analyzed `news_items` by headline match
    raw_url_map: Dict[str, str] = {}
    for raw_item in payload.get("intraday_breaking_news") or []:
        hl = (raw_item.get("headline") or "").strip().lower()
        if hl and raw_item.get("url"):
            raw_url_map[hl] = raw_item["url"]

    # Sort news_items strictly by `impact_weight` descending (10 down to 1), breaking ties by |sentiment_score|
    analyzed_news: List[Dict[str, Any]] = []
    for item in analysis.get("news_items") or []:
        item_copy = dict(item)
        hl_key = (item_copy.get("headline") or "").strip().lower()
        item_copy["url"] = raw_url_map.get(hl_key, "")
        analyzed_news.append(item_copy)

    analyzed_news.sort(
        key=lambda x: (
            int(x.get("impact_weight", 1)),
            abs(int(x.get("sentiment_score", 0))),
        ),
        reverse=True,
    )

    # Build Chart 1 Data: Horizontal Bar Chart of the 4 Gold Impact Wedges (-100 to +100)
    wedges: List[Dict[str, Any]] = list(analysis.get("wedge_breakdown") or [])
    wedge_labels: List[str] = []
    wedge_scores: List[float] = []
    wedge_colors: List[str] = []
    wedge_borders: List[str] = []

    for idx, w in enumerate(wedges, start=1):
        pct = w.get("weight_percentage", 0)
        name = w.get("wedge_name", f"Wedge {idx}")
        # Create concise chart label
        short_label = name.split("(")[0].strip()
        if len(short_label) > 42:
            short_label = short_label[:40] + "…"
        wedge_labels.append(f"{short_label} ({pct}%)")

        score = round(float(w.get("net_sentiment_score", 0.0)), 1)
        wedge_scores.append(score)
        if score > 5.0:
            wedge_colors.append("rgba(16, 185, 129, 0.75)")
            wedge_borders.append("#10b981")
        elif score < -5.0:
            wedge_colors.append("rgba(239, 68, 68, 0.75)")
            wedge_borders.append("#ef4444")
        else:
            wedge_colors.append("rgba(245, 158, 11, 0.75)")
            wedge_borders.append("#f59e0b")

    # Build Chart 2 Data: Doughnut Chart of Weighted Bullish Gold vs. Bearish Gold vs. Neutral news distribution
    weighted_dist = {"BULLISH GOLD": 0, "BEARISH GOLD": 0, "NEUTRAL": 0}
    count_dist = {"BULLISH GOLD": 0, "BEARISH GOLD": 0, "NEUTRAL": 0}

    for item in analyzed_news:
        b_key = (item.get("gold_bias") or "NEUTRAL").upper()
        wt = int(item.get("impact_weight", 1))
        if b_key not in weighted_dist:
            b_key = "NEUTRAL"
        weighted_dist[b_key] += wt
        count_dist[b_key] += 1

    total_wt = max(1, sum(weighted_dist.values()))
    doughnut_pcts = {
        k: round(v / total_wt * 100.0, 1) for k, v in weighted_dist.items()
    }

    # Compute sentiment gauge position (0% to 100% across [-100, +100] scale)
    weighted_score = float(analysis.get("weighted_sentiment_score", 0.0))
    sentiment_gauge_pct = max(2.0, min(98.0, round((weighted_score + 100.0) / 2.0, 1)))

    gold_calendar = payload.get("gold_economic_calendar") or {
        "source_status": "unavailable",
        "target_pair": "XAUUSD / COMEX GC=F (USD-Denominated Gold Only)",
        "total_gold_events": 0,
        "high_impact_count": 0,
        "medium_impact_count": 0,
        "next_major_gold_catalyst": "N/A",
        "calendar_macro_tilt": "NEUTRAL",
        "events": [],
    }

    admin_acc = get_admin_account()
    code_users = get_all_users()

    return {
        "bias_meta": bias_meta,
        "confluence_meta": confluence_meta,
        "thesis_bullets": thesis_bullets,
        "invalidation_bullet": invalidation_bullet,
        "sorted_news_items": analyzed_news,
        "gold_calendar": gold_calendar,
        "admin_account": admin_acc,
        "code_users": code_users,
        "portal_auth_json": json.dumps(
            {
                "admin": admin_acc,
                "users": code_users,
            }
        ),
        "sentiment_gauge_pct": sentiment_gauge_pct,
        "weighted_dist": weighted_dist,
        "count_dist": count_dist,
        "doughnut_pcts": doughnut_pcts,
        "chart_wedge_json": json.dumps(
            {
                "labels": wedge_labels,
                "scores": wedge_scores,
                "bgColors": wedge_colors,
                "borderColors": wedge_borders,
            }
        ),
        "chart_doughnut_json": json.dumps(
            {
                "labels": [
                    f"Bullish Gold ({count_dist['BULLISH GOLD']} items | {doughnut_pcts['BULLISH GOLD']}%)",
                    f"Bearish Gold ({count_dist['BEARISH GOLD']} items | {doughnut_pcts['BEARISH GOLD']}%)",
                    f"Neutral ({count_dist['NEUTRAL']} items | {doughnut_pcts['NEUTRAL']}%)",
                ],
                "weights": [
                    weighted_dist["BULLISH GOLD"],
                    weighted_dist["BEARISH GOLD"],
                    weighted_dist["NEUTRAL"],
                ],
                "counts": [
                    count_dist["BULLISH GOLD"],
                    count_dist["BEARISH GOLD"],
                    count_dist["NEUTRAL"],
                ],
            }
        ),
    }


_TERMINAL_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en" class="dark">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>XAUUSD / GC=F Institutional Gold Intraday Sentiment Terminal</title>
  <!-- Tailwind CSS CDN -->
  <script src="https://cdn.tailwindcss.com"></script>
  <!-- Chart.js CDN -->
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <!-- Google Fonts: Inter & JetBrains Mono -->
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
  <script>
    tailwind.config = {
      darkMode: 'class',
      theme: {
        extend: {
          colors: {
            terminal: {
              bg: '#0b0f19',
              card: '#111827',
              panel: '#161f33',
              border: '#1f2d47',
              gold: '#f59e0b',
              bull: '#10b981',
              bear: '#ef4444'
            }
          },
          fontFamily: {
            sans: ['Inter', '-apple-system', 'BlinkMacSystemFont', 'sans-serif'],
            mono: ['JetBrains Mono', 'monospace']
          }
        }
      }
    }
  </script>
  <style>
    body {
      background-color: #0b0f19;
      color: #f1f5f9;
      font-family: 'Inter', sans-serif;
      background-image:
        radial-gradient(circle at 12% 0%, rgba(245, 158, 11, 0.08), transparent 32%),
        radial-gradient(circle at 88% 4%, rgba(16, 185, 129, 0.05), transparent 28%);
    }
    .terminal-card {
      background: linear-gradient(180deg, rgba(22, 31, 51, 0.88) 0%, rgba(17, 24, 39, 0.94) 100%);
      border: 1px solid #1f2d47;
      box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.65);
    }
    .gold-glow {
      box-shadow: 0 0 28px -6px rgba(245, 158, 11, 0.25);
    }
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: #0b0f19; }
    ::-webkit-scrollbar-thumb { background: #1f2d47; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #334155; }
  </style>
</head>
<body class="min-h-screen antialiased selection:bg-amber-500/30 selection:text-amber-200">

  <!-- ===================================================================== -->
  <!-- MULTI-USER & ADMIN WEB PORTAL TOP BAR (SAVED IN CODE • NO DATABASE)   -->
  <!-- ===================================================================== -->
  <nav id="portalTopNav" class="sticky top-0 z-40 bg-slate-950/95 backdrop-blur-md border-b border-amber-500/30 px-4 sm:px-6 lg:px-8 py-2.5">
    <div class="max-w-[1520px] mx-auto flex flex-wrap items-center justify-between gap-3">
      <div class="flex items-center gap-3 flex-wrap">
        <span class="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-mono font-bold bg-amber-500/15 text-amber-300 border border-amber-500/40">
          <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
          WEB PORTAL LIVE (API-ONLINE &amp; API-OFFLINE READY)
        </span>
        <span id="currentUserBadge" class="px-2.5 py-1 rounded-md text-xs font-mono bg-slate-900 text-slate-200 border border-slate-700">
          Not Signed In
        </span>
        <span class="hidden md:inline-block text-[11px] font-mono text-slate-400">
          Storage: <strong class="text-amber-300">Code-Persisted (`users_store.py`) • No Database Needed</strong>
        </span>
      </div>

      <div class="flex items-center gap-2.5 flex-wrap font-mono text-xs">
        <!-- Refresh Live Gold Data (Works with or without API key) -->
        <button id="btnRefreshLive" type="button" onclick="window.triggerGoldPortalRefresh(false)"
                class="px-3 py-1.5 rounded-lg font-semibold bg-emerald-500/20 hover:bg-emerald-500/30 text-emerald-300 border border-emerald-500/50 transition flex items-center gap-1.5">
          <span id="refreshSpinner" class="hidden animate-spin">&#9696;</span>
          <span>&#8635; Refresh Live Gold (AI / Auto-Fallback)</span>
        </button>

        <button id="btnRefreshQuant" type="button" onclick="window.triggerGoldPortalRefresh(true)"
                title="Refresh live Gold prices, 18 breaking headlines &amp; Forex Factory Calendar using the built-in Quantitative Engine (Zero API Key Needed)"
                class="px-3 py-1.5 rounded-lg font-semibold bg-sky-500/15 hover:bg-sky-500/25 text-sky-300 border border-sky-500/40 transition">
          &#9889; Instant API-Free Refresh
        </button>

        <!-- Admin-Only Button (Hidden for normal users, shown ONLY for the single Administrator) -->
        <button id="btnOpenAdminModal" type="button" onclick="window.openAdminModal()"
                class="hidden px-3.5 py-1.5 rounded-lg font-bold bg-amber-500 hover:bg-amber-400 text-slate-950 shadow-md shadow-amber-500/20 transition">
          &#9881; Admin User Manager (<span id="navUserCount">0</span> Users)
        </button>

        <button id="btnLogout" type="button" onclick="window.logoutGoldPortal()"
                class="px-3 py-1.5 rounded-lg font-semibold bg-slate-800 hover:bg-red-500/20 text-slate-300 hover:text-red-300 border border-slate-700 hover:border-red-500/40 transition">
          Sign Out
        </button>
      </div>
    </div>
  </nav>

  <!-- ===================================================================== -->
  <!-- LOGIN GATE OVERLAY (SHOWN UNTIL ADMIN OR CREATED USER LOGS IN)        -->
  <!-- ===================================================================== -->
  <div id="loginOverlay" class="fixed inset-0 z-50 bg-slate-950/95 backdrop-blur-lg flex items-center justify-center p-4">
    <div class="terminal-card rounded-2xl max-w-md w-full p-6 sm:p-8 border-t-4 border-t-amber-500 shadow-2xl space-y-5">
      <div class="flex items-center gap-3.5">
        <div class="w-12 h-12 rounded-xl bg-amber-500/20 border border-amber-500/50 flex items-center justify-center text-amber-400 font-mono font-extrabold text-xl">
          Au
        </div>
        <div>
          <span class="px-2 py-0.5 text-[10px] font-mono font-bold uppercase tracking-wider bg-amber-500/20 text-amber-300 border border-amber-500/40 rounded">
            XAUUSD / GC=F WEB TERMINAL
          </span>
          <h2 class="text-xl font-extrabold text-white mt-0.5">
            Institutional Portal Sign-In
          </h2>
        </div>
      </div>

      <p class="text-xs text-slate-400 leading-relaxed">
        Sign in with the single <strong class="text-amber-300">Administrator</strong> account (to manage users &amp; view the terminal) or any <strong class="text-emerald-300">User</strong> account stored in <code class="text-slate-200">users_store.py</code>.
      </p>

      <div id="loginErrorBox" class="hidden rounded-lg bg-red-500/15 border border-red-500/40 p-3 text-xs font-mono text-red-300"></div>

      <form id="loginForm" onsubmit="return window.handlePortalLogin(event)" class="space-y-4">
        <div>
          <label class="block text-xs font-mono uppercase tracking-wider text-slate-400 mb-1">Username</label>
          <input id="loginUsername" type="text" required autocomplete="username" placeholder="e.g. admin or trader1"
                 class="w-full rounded-lg bg-slate-900 border border-slate-700 focus:border-amber-400 focus:outline-none px-3.5 py-2.5 text-sm font-mono text-white" />
        </div>
        <div>
          <label class="block text-xs font-mono uppercase tracking-wider text-slate-400 mb-1">Password</label>
          <input id="loginPassword" type="password" required autocomplete="current-password" placeholder="Enter password"
                 class="w-full rounded-lg bg-slate-900 border border-slate-700 focus:border-amber-400 focus:outline-none px-3.5 py-2.5 text-sm font-mono text-white" />
        </div>
        <button type="submit"
                class="w-full py-2.5 px-4 rounded-xl bg-amber-500 hover:bg-amber-400 text-slate-950 font-mono font-extrabold text-sm tracking-wide shadow-lg shadow-amber-500/20 transition">
          Access Gold Sentiment Terminal &rarr;
        </button>
      </form>

      <!-- Quick One-Click Credential Fillers for Convenience -->
      <div class="pt-3 border-t border-slate-800 space-y-2">
        <div class="text-[11px] font-mono text-slate-400 flex items-center justify-between">
          <span>Code-Stored Accounts (<code class="text-amber-300">users_store.py</code>):</span>
          <span class="text-emerald-400">Zero-DB Mode</span>
        </div>
        <div class="grid grid-cols-2 gap-2 text-xs font-mono">
          <button type="button" onclick="window.fillLogin('admin', 'admin123')"
                  class="px-2.5 py-1.5 rounded-lg bg-amber-500/10 hover:bg-amber-500/20 border border-amber-500/30 text-amber-300 text-left truncate">
            &#128081; Admin: <strong>admin</strong> / admin123
          </button>
          <button type="button" onclick="window.fillLogin('trader1', 'gold123')"
                  class="px-2.5 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 border border-slate-700 text-slate-300 text-left truncate">
            &#128100; User: <strong>trader1</strong> / gold123
          </button>
        </div>
      </div>
    </div>
  </div>

  <!-- ===================================================================== -->
  <!-- ADMINISTRATOR USER MANAGEMENT MODAL (ONLY FOR SINGLE ADMIN)           -->
  <!-- ===================================================================== -->
  <div id="adminModal" class="hidden fixed inset-0 z-50 bg-slate-950/85 backdrop-blur-md flex items-center justify-center p-4 overflow-y-auto">
    <div class="terminal-card rounded-2xl max-w-3xl w-full p-6 border-t-4 border-t-amber-500 shadow-2xl space-y-5 my-8">
      <div class="flex items-start justify-between gap-4 border-b border-slate-800 pb-4">
        <div>
          <div class="flex items-center gap-2">
            <span class="px-2 py-0.5 text-[10px] font-mono font-bold uppercase bg-amber-500/20 text-amber-300 border border-amber-500/40 rounded">
              SINGLE ADMINISTRATOR CONTROL PANEL
            </span>
            <span class="px-2 py-0.5 text-[10px] font-mono bg-emerald-500/15 text-emerald-300 border border-emerald-500/30 rounded">
              Persisted in Code: users_store.py
            </span>
          </div>
          <h3 class="text-lg font-extrabold text-white mt-1">
            Create &amp; Manage Portal Users (No Database Required)
          </h3>
          <p class="text-xs text-slate-400 mt-0.5">
            Every user created or updated below is written directly into <code class="text-amber-300">users_store.py</code> so all users can sign in and view the Gold Terminal page.
          </p>
        </div>
        <button type="button" onclick="window.closeAdminModal()"
                class="px-3 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 font-mono text-xs">
          &#10005; Close
        </button>
      </div>

      <div id="adminStatusBanner" class="hidden rounded-lg p-3 text-xs font-mono border"></div>

      <!-- Create / Update User Form -->
      <form id="adminCreateUserForm" onsubmit="return window.handleAdminCreateUser(event)"
            class="bg-slate-900/80 border border-slate-800 rounded-xl p-4 space-y-3">
        <div class="text-xs font-mono font-bold uppercase tracking-wider text-amber-400">
          + Create New User Account (or Update Existing Username)
        </div>
        <div class="grid grid-cols-1 sm:grid-cols-3 gap-3">
          <div>
            <label class="block text-[11px] font-mono text-slate-400 mb-1">Username *</label>
            <input id="newUsername" type="text" required placeholder="e.g. trader2"
                   class="w-full rounded-lg bg-slate-950 border border-slate-700 focus:border-amber-400 focus:outline-none px-3 py-2 text-xs font-mono text-white" />
          </div>
          <div>
            <label class="block text-[11px] font-mono text-slate-400 mb-1">Password *</label>
            <input id="newPassword" type="text" required placeholder="e.g. xauusd99"
                   class="w-full rounded-lg bg-slate-950 border border-slate-700 focus:border-amber-400 focus:outline-none px-3 py-2 text-xs font-mono text-white" />
          </div>
          <div>
            <label class="block text-[11px] font-mono text-slate-400 mb-1">Display Name / Desk Role</label>
            <input id="newDisplayName" type="text" placeholder="e.g. London Bullion Desk"
                   class="w-full rounded-lg bg-slate-950 border border-slate-700 focus:border-amber-400 focus:outline-none px-3 py-2 text-xs font-mono text-white" />
          </div>
        </div>
        <div class="flex flex-wrap items-center justify-between gap-2 pt-1">
          <span class="text-[11px] font-mono text-slate-400">
            Single Admin Account (<code class="text-amber-300">admin</code>) is permanently locked as the sole Administrator.
          </span>
          <div class="flex items-center gap-2">
            <button type="button" onclick="window.downloadUsersStorePy()"
                    title="Download updated users_store.py source file directly"
                    class="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 font-mono text-xs">
              &#128190; Export `users_store.py`
            </button>
            <button type="submit"
                    class="px-4 py-1.5 rounded-lg bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-mono font-bold text-xs shadow">
              + Save User to Code
            </button>
          </div>
        </div>
      </form>

      <!-- Existing Users Table -->
      <div class="border border-slate-800 rounded-xl overflow-hidden">
        <div class="bg-slate-900/90 px-4 py-2.5 border-b border-slate-800 flex items-center justify-between text-xs font-mono">
          <span class="font-bold text-slate-200">Code-Persisted User Directory (<code class="text-amber-300">USERS_LIST</code>)</span>
          <span class="text-slate-400">1 Admin + <strong id="modalUserCount" class="text-amber-400">0</strong> Users</span>
        </div>
        <div class="overflow-x-auto max-h-72 overflow-y-auto">
          <table class="w-full text-left border-collapse text-xs font-mono">
            <thead>
              <tr class="bg-slate-950 text-slate-400 border-b border-slate-800 uppercase text-[10px]">
                <th class="py-2.5 px-3">Username</th>
                <th class="py-2.5 px-3">Password</th>
                <th class="py-2.5 px-3">Display Name</th>
                <th class="py-2.5 px-3">Role</th>
                <th class="py-2.5 px-3 text-center">Status</th>
                <th class="py-2.5 px-3 text-right">Admin Actions</th>
              </tr>
            </thead>
            <tbody id="adminUsersTableBody" class="divide-y divide-slate-800/80">
            </tbody>
          </table>
        </div>
      </div>
    </div>
  </div>

  <div class="max-w-[1520px] mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">

    <!-- =================================================================== -->
    <!-- TOP INSTITUTIONAL TERMINAL HEADER                                   -->
    <!-- =================================================================== -->
    <header class="terminal-card rounded-xl px-6 py-4 flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4 border-t-2 border-t-amber-500">
      <div class="flex items-center gap-4">
        <div class="w-12 h-12 rounded-xl bg-amber-500/15 border border-amber-500/40 flex items-center justify-center text-amber-400 font-mono font-bold text-lg shadow-inner">
          Au
        </div>
        <div>
          <div class="flex items-center gap-2.5 flex-wrap">
            <span class="px-2 py-0.5 text-[11px] font-mono font-semibold uppercase tracking-wider bg-amber-500/20 text-amber-300 border border-amber-500/40 rounded">
              XAUUSD / COMEX GC=F
            </span>
            <span class="text-xs font-mono text-slate-400">
              PRECIOUS METALS QUANTITATIVE DESK
            </span>
          </div>
          <h1 class="text-xl sm:text-2xl font-extrabold tracking-tight text-white mt-0.5">
            Gold Intraday News &amp; Macro Correlation Terminal
          </h1>
        </div>
      </div>

      <div class="flex flex-wrap items-center gap-4 text-xs font-mono">
        <div class="bg-slate-900/90 border border-slate-800 rounded-lg px-3.5 py-2">
          <div class="text-slate-400 text-[10px] uppercase tracking-wider">Snapshot Timestamp (UTC)</div>
          <div class="text-slate-200 font-semibold mt-0.5">{{ payload.generated_at_utc }}</div>
        </div>
        <div class="bg-slate-900/90 border border-slate-800 rounded-lg px-3.5 py-2">
          <div class="text-slate-400 text-[10px] uppercase tracking-wider">Analysis Engine</div>
          <div class="text-amber-400 font-semibold mt-0.5">{{ analysis.analysis_engine or "gemini-3.5-flash" }}</div>
        </div>
        <div class="bg-slate-900/90 border border-slate-800 rounded-lg px-3.5 py-2">
          <div class="text-slate-400 text-[10px] uppercase tracking-wider">Feed Coverage</div>
          <div class="text-emerald-400 font-semibold mt-0.5">
            {{ payload.ingestion_diagnostics.active_feeds_returning_data }}/{{ payload.ingestion_diagnostics.total_feeds_queried }} Active ({{ payload.fetch_latency_seconds }}s)
          </div>
        </div>
      </div>
    </header>

    <!-- =================================================================== -->
    <!-- 1. TOP LIVE MARKET RIBBON (GC=F, DXY, ^TNX, ^VIX, GLD)              -->
    <!-- =================================================================== -->
    <section aria-label="Live Gold and Macro Correlation Ribbon">
      <div class="flex items-center justify-between mb-2.5 px-1">
        <h2 class="text-xs font-mono uppercase tracking-widest text-amber-400/90 font-semibold flex items-center gap-2">
          <span class="w-2 h-2 rounded-full bg-amber-400 animate-pulse"></span>
          Live Cross-Asset Gold Correlation Ribbon
        </h2>
        <span class="text-xs font-mono text-slate-400">
          Inverse Drivers: <strong class="text-slate-300">DXY</strong> &amp; <strong class="text-slate-300">^TNX (10Y Yield)</strong> |
          Safe-Haven &amp; Flow Proxies: <strong class="text-slate-300">^VIX</strong> &amp; <strong class="text-slate-300">GLD</strong>
        </span>
      </div>

      <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3.5">
        {% for key, item in payload.gold_macro_snapshot.tickers.items() %}
        {% set is_primary = (key == 'GOLD_FUTURES') %}
        {% set price_up = (item.change is not none and item.change >= 0) %}
        {% set gold_eff = item.gold_directional_effect %}
        <div class="terminal-card rounded-xl p-4 relative overflow-hidden transition duration-200 hover:border-amber-500/50 {{ 'border-amber-500/60 gold-glow' if is_primary else '' }}">
          <div class="flex items-start justify-between gap-2">
            <div>
              <div class="text-[11px] font-mono font-bold uppercase tracking-wider {{ 'text-amber-400' if is_primary else 'text-slate-400' }}">
                {{ item.display_symbol }}
              </div>
              <div class="text-xs text-slate-300 font-medium truncate max-w-[180px]" title="{{ item.label }}">
                {{ item.label }}
              </div>
            </div>
            <span class="px-2 py-0.5 rounded text-[10px] font-mono font-semibold border
              {% if gold_eff == 'BULLISH GOLD' %}
                bg-emerald-500/15 text-emerald-300 border-emerald-500/40
              {% elif gold_eff == 'BEARISH GOLD' %}
                bg-red-500/15 text-red-300 border-red-500/40
              {% else %}
                bg-amber-500/15 text-amber-300 border-amber-500/40
              {% endif %}">
              {{ gold_eff }}
            </span>
          </div>

          {% if item.price is not none %}
          <div class="mt-3 flex items-baseline justify-between">
            <div class="text-2xl font-mono font-extrabold tracking-tight text-white">
              {% if item.unit == 'USD/oz' or item.unit == 'USD' %}${% endif %}{{ "{:,.2f}".format(item.price) if item.symbol != '^TNX' else "{:.3f}".format(item.price) }}<span class="text-xs font-normal text-slate-400 ml-1">{{ item.unit if item.unit not in ('USD', 'USD/oz') else ('/oz' if item.unit == 'USD/oz' else '') }}</span>
            </div>
            <div class="flex items-center gap-1 font-mono text-xs font-bold
              {% if is_primary or key == 'GLD_ETF' %}
                {{ 'text-emerald-400' if price_up else 'text-red-400' }}
              {% elif key in ('DXY_DOLLAR_INDEX', 'US_10Y_YIELD') %}
                {{ 'text-red-400' if price_up else 'text-emerald-400' }}
              {% else %}
                {{ 'text-amber-300' if price_up else 'text-slate-300' }}
              {% endif %}">
              <span>{{ '▲' if price_up else '▼' }}</span>
              <span>{{ "%+.2f"|format(item.change_pct) }}%</span>
            </div>
          </div>

          <div class="mt-2.5 pt-2.5 border-t border-slate-800/80 flex items-center justify-between text-[11px] font-mono text-slate-400">
            <span>L: <strong class="text-slate-300">{{ "{:,.2f}".format(item.day_low) if item.symbol != '^TNX' else "{:.3f}".format(item.day_low) }}</strong></span>
            <span>H: <strong class="text-slate-300">{{ "{:,.2f}".format(item.day_high) if item.symbol != '^TNX' else "{:.3f}".format(item.day_high) }}</strong></span>
            <span>5d: <strong class="{{ 'text-emerald-400' if item.five_day_change_pct >= 0 else 'text-red-400' }}">{{ "%+.2f"|format(item.five_day_change_pct) }}%</strong></span>
          </div>
          <p class="mt-2 text-[11px] text-slate-400 leading-snug line-clamp-2" title="{{ item.regime_signal }}">
            {{ item.regime_signal }}
          </p>
          {% else %}
          <div class="mt-4 text-lg font-mono text-amber-400">Live Quote Unavailable</div>
          {% endif %}
        </div>
        {% endfor %}
      </div>
    </section>

    <!-- =================================================================== -->
    <!-- 2. HERO INTRADAY BIAS & CONFIDENCE BANNER                           -->
    <!-- =================================================================== -->
    <section aria-label="Hero Intraday Bias and Confidence Banner" class="terminal-card rounded-2xl p-6 border-l-4"
             style="border-left-color: {{ vm.bias_meta.accent_hex }};">
      <div class="grid grid-cols-1 lg:grid-cols-12 gap-6 items-center">

        <!-- Left: Large Color-Coded Bias Badge -->
        <div class="lg:col-span-4 flex flex-col justify-center space-y-3 border-b lg:border-b-0 lg:border-r border-slate-800 pb-5 lg:pb-0 lg:pr-6">
          <div class="text-xs font-mono uppercase tracking-widest text-slate-400">
            Institutional Intraday Signal
          </div>
          <div class="inline-flex items-center gap-3 self-start px-5 py-3 rounded-xl border-2 shadow-lg {{ vm.bias_meta.bg_class }}">
            <span class="w-3.5 h-3.5 rounded-full {{ vm.bias_meta.dot_class }} animate-ping"></span>
            <span class="text-2xl sm:text-3xl font-mono font-extrabold tracking-tight">
              {{ vm.bias_meta.label }}
            </span>
          </div>
          <div class="pt-1">
            <span class="inline-flex items-center gap-1.5 px-3 py-1 rounded-md text-xs font-mono font-semibold border {{ vm.confluence_meta.badge_class }}">
              {{ vm.confluence_meta.label }}
            </span>
            <p class="text-xs text-slate-400 mt-1.5 leading-relaxed">
              {{ vm.confluence_meta.description }}
            </p>
          </div>
        </div>

        <!-- Middle: Weighted Sentiment Score Meter (-100 to +100) -->
        <div class="lg:col-span-4 flex flex-col justify-center space-y-3 border-b lg:border-b-0 lg:border-r border-slate-800 pb-5 lg:pb-0 lg:px-4">
          <div class="flex items-center justify-between">
            <span class="text-xs font-mono uppercase tracking-wider text-slate-400">
              Wedge-Weighted Sentiment Score
            </span>
            <span class="text-2xl font-mono font-extrabold
              {% if analysis.weighted_sentiment_score > 10 %}text-emerald-400
              {% elif analysis.weighted_sentiment_score < -10 %}text-red-400
              {% else %}text-amber-400{% endif %}">
              {{ "%+.1f"|format(analysis.weighted_sentiment_score) }} <span class="text-xs font-normal text-slate-500">/ 100</span>
            </span>
          </div>

          <!-- Bi-Directional Sentiment Bar (-100 Bearish to +100 Bullish) -->
          <div class="relative h-4 w-full rounded-full bg-slate-900 border border-slate-700/80 overflow-hidden">
            <div class="absolute inset-0 flex">
              <div class="w-1/2 bg-gradient-to-r from-red-500/35 via-red-500/15 to-transparent border-r border-slate-600"></div>
              <div class="w-1/2 bg-gradient-to-r from-transparent via-emerald-500/15 to-emerald-500/35"></div>
            </div>
            <!-- Marker Pin -->
            <div class="absolute top-0 bottom-0 w-2.5 -ml-1.5 rounded-full bg-amber-400 border-2 border-white shadow-md"
                 style="left: {{ vm.sentiment_gauge_pct }}%;"></div>
          </div>

          <div class="flex justify-between text-[11px] font-mono text-slate-400">
            <span class="text-red-400">-100 Max Bearish (Hawkish/Strong USD)</span>
            <span class="text-slate-500">0 Neutral</span>
            <span class="text-emerald-400">+100 Max Bullish (Dovish/Safe-Haven)</span>
          </div>
        </div>

        <!-- Right: Confidence Score Gauge / Progress Bar (0% - 100%) -->
        <div class="lg:col-span-4 flex flex-col justify-center space-y-3 lg:pl-2">
          <div class="flex items-center justify-between">
            <span class="text-xs font-mono uppercase tracking-wider text-slate-400">
              Conviction &amp; Alignment Confidence
            </span>
            <span class="text-2xl font-mono font-extrabold text-amber-400">
              {{ analysis.confidence_score }}%
            </span>
          </div>

          <div class="w-full h-4 bg-slate-900 rounded-full border border-slate-700/80 overflow-hidden p-0.5">
            <div class="h-full rounded-full bg-gradient-to-r {{ vm.bias_meta.bar_class }} transition-all duration-500"
                 style="width: {{ analysis.confidence_score }}%;"></div>
          </div>

          <div class="flex justify-between text-[11px] font-mono text-slate-400">
            <span>0% Low</span>
            <span>30–65% Cross-Current Conflict</span>
            <span class="text-amber-300">80–100% Full Macro Alignment</span>
          </div>
        </div>

      </div>
    </section>

    <!-- =================================================================== -->
    <!-- 3. EXECUTIVE THESIS PANEL & MACROECONOMIC CONDITION PANEL           -->
    <!-- =================================================================== -->
    <section class="grid grid-cols-1 lg:grid-cols-12 gap-6">

      <!-- Left (7 cols): "Why Buy or Sell XAUUSD / GC" Executive Thesis Panel -->
      <div class="lg:col-span-7 terminal-card rounded-2xl p-6 flex flex-col justify-between">
        <div>
          <div class="flex items-center justify-between border-b border-slate-800 pb-3.5 mb-4">
            <h2 class="text-base font-bold text-white flex items-center gap-2.5">
              <span class="w-2.5 h-2.5 rounded-sm bg-amber-400"></span>
              Why Buy or Sell XAUUSD / GC — Executive Institutional Thesis
            </h2>
            <span class="text-xs font-mono text-amber-400 bg-amber-500/10 border border-amber-500/30 px-2.5 py-0.5 rounded">
              4-Wedge Synthesis
            </span>
          </div>

          <ul class="space-y-3 text-sm text-slate-200 leading-relaxed">
            {% for bullet in vm.thesis_bullets %}
            <li class="flex items-start gap-3 bg-slate-900/60 border border-slate-800/80 rounded-xl p-3.5">
              <span class="mt-1 w-5 h-5 rounded-full bg-amber-500/20 border border-amber-500/50 text-amber-300 font-mono text-xs flex items-center justify-center shrink-0">
                {{ loop.index }}
              </span>
              <span>{{ bullet }}</span>
            </li>
            {% endfor %}
          </ul>
        </div>

        {% if vm.invalidation_bullet %}
        <div class="mt-4 bg-red-950/35 border border-red-500/40 rounded-xl p-4 flex items-start gap-3.5">
          <div class="px-2 py-1 rounded bg-red-500/20 border border-red-500/40 text-red-300 font-mono text-xs font-bold uppercase shrink-0 mt-0.5">
            Invalidation Trigger
          </div>
          <p class="text-xs sm:text-sm text-red-200/90 leading-relaxed">
            {{ vm.invalidation_bullet }}
          </p>
        </div>
        {% endif %}
      </div>

      <!-- Right (5 cols): Current Macroeconomic Condition Panel -->
      <div class="lg:col-span-5 terminal-card rounded-2xl p-6 flex flex-col justify-between space-y-4">
        <div>
          <div class="flex items-center justify-between border-b border-slate-800 pb-3.5 mb-4">
            <h2 class="text-base font-bold text-white flex items-center gap-2.5">
              <span class="w-2.5 h-2.5 rounded-sm bg-emerald-400"></span>
              Current Macroeconomic Condition Panel
            </h2>
            <span class="text-xs font-mono text-slate-400">Real Yields • DXY • Safe-Haven</span>
          </div>

          <div class="bg-slate-900/75 border border-slate-800 rounded-xl p-4 text-sm text-slate-200 leading-relaxed">
            {{ analysis.macro_condition_summary }}
          </div>
        </div>

        <!-- 4 Gold Impact Wedges Mini-Cards -->
        <div class="space-y-2.5">
          <div class="text-xs font-mono uppercase tracking-wider text-slate-400">
            Quantitative Wedge Attribution (40% / 30% / 20% / 10%)
          </div>
          {% for w in analysis.wedge_breakdown %}
          <div class="bg-slate-900/60 border border-slate-800/90 rounded-lg px-3.5 py-2.5">
            <div class="flex items-center justify-between text-xs font-mono">
              <span class="font-semibold text-slate-200 truncate pr-2">{{ w.wedge_name }}</span>
              <div class="flex items-center gap-2 shrink-0">
                <span class="px-1.5 py-0.5 rounded bg-slate-800 text-slate-300">{{ w.weight_percentage }}% Wt</span>
                <span class="font-bold px-2 py-0.5 rounded
                  {% if w.net_sentiment_score > 5 %}bg-emerald-500/15 text-emerald-300 border border-emerald-500/30
                  {% elif w.net_sentiment_score < -5 %}bg-red-500/15 text-red-300 border border-red-500/30
                  {% else %}bg-amber-500/15 text-amber-300 border border-amber-500/30{% endif %}">
                  {{ "%+.1f"|format(w.net_sentiment_score) }}
                </span>
              </div>
            </div>
            <p class="text-xs text-slate-400 mt-1 leading-snug">{{ w.summary }}</p>
          </div>
          {% endfor %}
        </div>
      </div>

    </section>

    <!-- =================================================================== -->
    <!-- 4. CHART.JS VISUALIZATIONS (SIDE-BY-SIDE)                           -->
    <!-- =================================================================== -->
    <section class="grid grid-cols-1 lg:grid-cols-12 gap-6">

      <!-- Chart 1: Horizontal Bar Chart of the 4 Gold Impact Wedges (-100 to +100) -->
      <div class="lg:col-span-7 terminal-card rounded-2xl p-6">
        <div class="flex items-center justify-between mb-4">
          <div>
            <h2 class="text-base font-bold text-white">
              Gold Impact Wedges — Net Bullish vs. Bearish Score (-100 to +100)
            </h2>
            <p class="text-xs text-slate-400 mt-0.5">
              Directional contribution of each macro wedge to XAUUSD / COMEX GC=F
            </p>
          </div>
          <span class="text-xs font-mono px-2.5 py-1 rounded bg-slate-900 border border-slate-800 text-slate-300">
            Horizontal Bar
          </span>
        </div>
        <div class="relative h-[290px] w-full">
          <canvas id="wedgeBarChart"></canvas>
        </div>
      </div>

      <!-- Chart 2: Doughnut Chart of Weighted Bullish vs. Bearish vs. Neutral News -->
      <div class="lg:col-span-5 terminal-card rounded-2xl p-6">
        <div class="flex items-center justify-between mb-4">
          <div>
            <h2 class="text-base font-bold text-white">
              Impact-Weighted News Polarity Distribution
            </h2>
            <p class="text-xs text-slate-400 mt-0.5">
              Share of total catalyst impact weight across {{ vm.sorted_news_items | length }} breaking Gold headlines
            </p>
          </div>
          <span class="text-xs font-mono px-2.5 py-1 rounded bg-slate-900 border border-slate-800 text-slate-300">
            Weighted Doughnut
          </span>
        </div>
        <div class="relative h-[240px] w-full flex items-center justify-center">
          <canvas id="newsDoughnutChart"></canvas>
        </div>
        <div class="grid grid-cols-3 gap-2 mt-4 pt-3 border-t border-slate-800 text-center font-mono text-xs">
          <div class="bg-emerald-500/10 border border-emerald-500/30 rounded-lg py-1.5">
            <div class="text-emerald-300 font-bold">{{ vm.doughnut_pcts['BULLISH GOLD'] }}%</div>
            <div class="text-[10px] text-slate-400">Bullish ({{ vm.count_dist['BULLISH GOLD'] }} items)</div>
          </div>
          <div class="bg-red-500/10 border border-red-500/30 rounded-lg py-1.5">
            <div class="text-red-300 font-bold">{{ vm.doughnut_pcts['BEARISH GOLD'] }}%</div>
            <div class="text-[10px] text-slate-400">Bearish ({{ vm.count_dist['BEARISH GOLD'] }} items)</div>
          </div>
          <div class="bg-amber-500/10 border border-amber-500/30 rounded-lg py-1.5">
            <div class="text-amber-300 font-bold">{{ vm.doughnut_pcts['NEUTRAL'] }}%</div>
            <div class="text-[10px] text-slate-400">Neutral ({{ vm.count_dist['NEUTRAL'] }} items)</div>
          </div>
        </div>
      </div>

    </section>

    <!-- =================================================================== -->
    <!-- 5. INTRADAY BREAKING NEWS TABLE (SORTED BY IMPACT WEIGHT 10 -> 1)   -->
    <!-- =================================================================== -->
    <section class="terminal-card rounded-2xl overflow-hidden">
      <div class="px-6 py-4 border-b border-slate-800 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
        <div>
          <h2 class="text-base font-bold text-white flex items-center gap-2">
            <span class="w-2.5 h-2.5 rounded-full bg-amber-400"></span>
            Intraday Gold &amp; Macro Breaking News Matrix (Sorted by Impact Weight 10 &rarr; 1)
          </h2>
          <p class="text-xs text-slate-400 mt-0.5">
            Deduplicated headlines scored for XAUUSD / GC=F transmission mechanism and wedge attribution
          </p>
        </div>
        <div class="text-xs font-mono text-slate-400">
          Showing <strong class="text-amber-400">{{ vm.sorted_news_items | length }}</strong> Verified Catalysts
        </div>
      </div>

      <div class="overflow-x-auto">
        <table class="w-full text-left border-collapse">
          <thead>
            <tr class="bg-slate-900/90 border-b border-slate-800 text-[11px] font-mono uppercase tracking-wider text-slate-400">
              <th class="py-3.5 px-4 text-center w-28">Impact Wt</th>
              <th class="py-3.5 px-4 w-56">Gold Wedge Category</th>
              <th class="py-3.5 px-4">Breaking Headline &amp; Timestamp</th>
              <th class="py-3.5 px-4 text-center w-40">Individual Bias</th>
              <th class="py-3.5 px-5 w-[34%]">Gold Transmission Mechanism</th>
            </tr>
          </thead>
          <tbody class="divide-y divide-slate-800/80 text-sm">
            {% for item in vm.sorted_news_items %}
            {% set wt = item.impact_weight | int %}
            {% set bias = item.gold_bias %}
            <tr class="hover:bg-slate-800/40 transition-colors">
              <!-- Column 1: Impact Weight Badge (1-10) -->
              <td class="py-4 px-4 text-center align-top">
                <div class="inline-flex flex-col items-center justify-center px-3 py-1.5 rounded-lg font-mono font-bold border
                  {% if wt >= 8 %}
                    bg-amber-500/20 text-amber-300 border-amber-500/50 shadow-sm shadow-amber-500/10
                  {% elif wt >= 5 %}
                    bg-sky-500/15 text-sky-300 border-sky-500/40
                  {% else %}
                    bg-slate-800 text-slate-300 border-slate-700
                  {% endif %}">
                  <span class="text-base leading-none">{{ wt }}<span class="text-[10px] font-normal opacity-75">/10</span></span>
                  <span class="text-[9px] uppercase tracking-wider mt-0.5 opacity-80">
                    {{ 'High' if wt >= 8 else ('Med' if wt >= 5 else 'Low') }}
                  </span>
                </div>
              </td>

              <!-- Column 2: Gold Wedge Category -->
              <td class="py-4 px-4 align-top">
                <div class="text-xs font-mono font-semibold text-amber-300/95 leading-snug">
                  {{ item.gold_wedge_category }}
                </div>
              </td>

              <!-- Column 3: Headline & Timestamp -->
              <td class="py-4 px-4 align-top">
                {% if item.url %}
                <a href="{{ item.url }}" target="_blank" rel="noopener noreferrer"
                   class="font-semibold text-white hover:text-amber-300 transition-colors leading-snug block">
                  {{ item.headline }}
                </a>
                {% else %}
                <div class="font-semibold text-white leading-snug">
                  {{ item.headline }}
                </div>
                {% endif %}
                <div class="text-xs font-mono text-slate-400 mt-1.5">
                  {{ item.source_time }}
                </div>
              </td>

              <!-- Column 4: Individual Bias Badge -->
              <td class="py-4 px-4 text-center align-top">
                <span class="inline-block px-2.5 py-1 rounded-md text-xs font-mono font-bold border
                  {% if bias == 'BULLISH GOLD' %}
                    bg-emerald-500/15 text-emerald-300 border-emerald-500/40
                  {% elif bias == 'BEARISH GOLD' %}
                    bg-red-500/15 text-red-300 border-red-500/40
                  {% else %}
                    bg-amber-500/15 text-amber-300 border-amber-500/40
                  {% endif %}">
                  {{ bias }}
                </span>
                <div class="mt-1 font-mono text-xs font-semibold
                  {% if item.sentiment_score > 0 %}text-emerald-400
                  {% elif item.sentiment_score < 0 %}text-red-400
                  {% else %}text-amber-400{% endif %}">
                  Score: {{ "%+d"|format(item.sentiment_score) }}
                </div>
              </td>

              <!-- Column 5: 1-Sentence Gold Transmission Mechanism -->
              <td class="py-4 px-5 align-top text-xs sm:text-sm text-slate-300 leading-relaxed">
                {{ item.gold_transmission_mechanism }}
              </td>
            </tr>
            {% endfor %}
          </tbody>
        </table>
      </div>
    </section>

    <!-- =================================================================== -->
    <!-- 6. GOLD-ONLY ECONOMIC CALENDAR & SMART EVENT ANALYSIS               -->
    <!--    (FOREX FACTORY JSON/XML • MQL5 / JBLANKED • ML 1M/30M/1H • GPT)  -->
    <!-- =================================================================== -->
    {% if vm.gold_calendar and vm.gold_calendar.events %}
    <section aria-label="Gold-Only Economic Calendar and Smart Event Analysis" class="terminal-card rounded-2xl overflow-hidden border-t-2 border-t-amber-500/80">
      <!-- Section Header -->
      <div class="px-6 py-5 border-b border-slate-800 bg-slate-900/60">
        <div class="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
          <div>
            <div class="flex items-center gap-2.5 flex-wrap mb-1">
              <span class="px-2.5 py-0.5 text-[11px] font-mono font-bold uppercase tracking-wider bg-amber-500/20 text-amber-300 border border-amber-500/40 rounded">
                GOLD-ONLY FUNDAMENTAL CALENDAR (USD &rarr; XAUUSD / GC=F)
              </span>
              <span class="px-2 py-0.5 text-[11px] font-mono text-emerald-300 bg-emerald-500/10 border border-emerald-500/30 rounded">
                Forex Factory Weekly JSON/XML + MQL5/JBlanked Event IDs
              </span>
              <span class="px-2 py-0.5 text-[11px] font-mono text-sky-300 bg-sky-500/10 border border-sky-500/30 rounded">
                Smart Analysis • ML (1m/30m/1h) • NewsGPT Gold Engine
              </span>
            </div>
            <h2 class="text-base sm:text-lg font-bold text-white flex items-center gap-2">
              <span class="w-2.5 h-2.5 rounded-full bg-amber-400"></span>
              XAUUSD Economic Calendar, Smart Outcome Playbook &amp; ML Price Action Projections
            </h2>
            <p class="text-xs text-slate-400 mt-0.5">
              Strictly filtered for USD &amp; Federal Reserve events that directly drive Gold (`XAUUSD` / `GC=F`) via real yields (`^TNX`) and `DXY` (all non-Gold FX pairs &amp; energy noise excluded)
            </p>
          </div>

          <div class="flex items-center gap-3 text-xs font-mono shrink-0">
            <div class="bg-slate-950/90 border border-slate-800 rounded-lg px-3.5 py-2 text-center">
              <div class="text-[10px] uppercase text-slate-400">Tier-1 High Impact</div>
              <div class="text-amber-400 font-extrabold text-sm mt-0.5">{{ vm.gold_calendar.high_impact_count }} Events</div>
            </div>
            <div class="bg-slate-950/90 border border-slate-800 rounded-lg px-3.5 py-2 text-center">
              <div class="text-[10px] uppercase text-slate-400">Total Gold Events</div>
              <div class="text-slate-200 font-extrabold text-sm mt-0.5">{{ vm.gold_calendar.total_gold_events }} Filtered</div>
            </div>
          </div>
        </div>

        <!-- 2 Summary KPI Callout Strips: Next Tier-1 Gold Catalyst & Weekly Calendar Gold Tilt -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-3.5 mt-4 pt-4 border-t border-slate-800/80 text-xs font-mono">
          <div class="bg-slate-950/75 border border-amber-500/30 rounded-xl px-4 py-3 flex items-start gap-3">
            <span class="px-2 py-0.5 rounded bg-amber-500/20 text-amber-300 border border-amber-500/40 font-bold uppercase text-[10px] shrink-0 mt-0.5">
              Next Tier-1 Gold Catalyst
            </span>
            <span class="text-slate-200 font-semibold leading-snug">
              {{ vm.gold_calendar.next_major_gold_catalyst }}
            </span>
          </div>
          <div class="bg-slate-950/75 border border-slate-800 rounded-xl px-4 py-3 flex items-start gap-3">
            <span class="px-2 py-0.5 rounded bg-sky-500/20 text-sky-300 border border-sky-500/40 font-bold uppercase text-[10px] shrink-0 mt-0.5">
              Calendar Macro Gold Tilt
            </span>
            <span class="text-slate-200 font-semibold leading-snug">
              {{ vm.gold_calendar.calendar_macro_tilt }}
            </span>
          </div>
        </div>
      </div>

      <!-- Economic Calendar Table -->
      <div class="overflow-x-auto">
        <table class="w-full text-left border-collapse">
          <thead>
            <tr class="bg-slate-900/95 border-b border-slate-800 text-[11px] font-mono uppercase tracking-wider text-slate-400">
              <th class="py-3.5 px-4 w-44">Date / Time (UTC)</th>
              <th class="py-3.5 px-4 w-64">Gold Event, ID &amp; Wedge</th>
              <th class="py-3.5 px-3 text-center w-36">Strength &amp; Quality</th>
              <th class="py-3.5 px-4 text-center w-52">Actual / Forecast / Prev</th>
              <th class="py-3.5 px-4 w-[28%]">Smart Analysis &amp; ML (1m / 30m / 1h) Gold Reaction</th>
              <th class="py-3.5 px-4 w-[24%]">NewsGPT Gold Bias &amp; Projection</th>
            </tr>
          </thead>
          <tbody class="divide-y divide-slate-800/80 text-xs sm:text-sm">
            {% for ev in vm.gold_calendar.events %}
            <tr class="hover:bg-slate-800/40 transition-colors {{ 'bg-amber-500/[0.03]' if ev.impact_level == 'High' else '' }}">
              <!-- Column 1: Date/Time & Release Status -->
              <td class="py-4 px-4 align-top font-mono">
                <div class="text-slate-200 font-semibold text-xs">{{ ev.date_utc }}</div>
                <span class="inline-block mt-1.5 px-2 py-0.5 rounded text-[10px] font-bold border
                  {% if 'IMMINENT' in ev.release_status %}
                    bg-amber-500/20 text-amber-300 border-amber-500/50 animate-pulse
                  {% elif 'RELEASED TODAY' in ev.release_status %}
                    bg-emerald-500/15 text-emerald-300 border-emerald-500/40
                  {% elif 'UPCOMING' in ev.release_status %}
                    bg-sky-500/15 text-sky-300 border-sky-500/40
                  {% else %}
                    bg-slate-800 text-slate-400 border-slate-700
                  {% endif %}">
                  {{ ev.release_status }}
                </span>
              </td>

              <!-- Column 2: Event Name, MQL5/JBlanked ID & Gold Wedge Category -->
              <td class="py-4 px-4 align-top">
                <div class="flex items-center gap-1.5 flex-wrap mb-1">
                  <span class="px-1.5 py-0.5 rounded bg-amber-500/15 border border-amber-500/30 text-amber-300 font-mono text-[10px] font-bold">
                    {{ ev.currency }}
                  </span>
                  <span class="px-1.5 py-0.5 rounded bg-slate-800 border border-slate-700 text-slate-400 font-mono text-[10px]">
                    ID: {{ ev.event_id }}
                  </span>
                </div>
                <a href="{{ ev.url }}" target="_blank" rel="noopener noreferrer"
                   class="font-bold text-white hover:text-amber-300 transition-colors leading-snug block">
                  {{ ev.event_name }}
                </a>
                <div class="text-[11px] font-mono text-amber-300/90 mt-1">
                  {{ ev.gold_category }}
                </div>
              </td>

              <!-- Column 3: Strength & Quality -->
              <td class="py-4 px-3 text-center align-top font-mono">
                <span class="inline-block px-2.5 py-1 rounded-md text-[11px] font-bold border
                  {% if ev.impact_level == 'High' %}
                    bg-red-500/20 text-red-300 border-red-500/50 shadow-sm shadow-red-500/10
                  {% elif ev.impact_level == 'Medium' %}
                    bg-amber-500/15 text-amber-300 border-amber-500/40
                  {% else %}
                    bg-slate-800 text-slate-300 border-slate-700
                  {% endif %}">
                  {{ ev.strength }}
                </span>
                <div class="text-[10px] text-slate-400 mt-1.5">
                  {{ ev.quality }}
                </div>
              </td>

              <!-- Column 4: Actual / Forecast / Previous & Outcome -->
              <td class="py-4 px-4 align-top font-mono">
                <div class="grid grid-cols-3 gap-1.5 text-center bg-slate-900/90 border border-slate-800 rounded-lg p-2 text-xs">
                  <div>
                    <div class="text-[9px] uppercase text-slate-500">Actual</div>
                    <div class="font-bold text-amber-300 mt-0.5">{{ ev.actual }}</div>
                  </div>
                  <div class="border-x border-slate-800 px-1">
                    <div class="text-[9px] uppercase text-slate-500">Forecast</div>
                    <div class="font-semibold text-slate-200 mt-0.5">{{ ev.forecast }}</div>
                  </div>
                  <div>
                    <div class="text-[9px] uppercase text-slate-500">Prev</div>
                    <div class="text-slate-400 mt-0.5">{{ ev.previous }}</div>
                  </div>
                </div>
                <div class="text-[10px] text-center text-slate-400 mt-1.5 truncate" title="{{ ev.outcome_vs_projection }}">
                  {{ ev.outcome_vs_projection }}
                </div>
              </td>

              <!-- Column 5: Smart Analysis Rules & ML (1m / 30m / 1h) XAUUSD Price Action -->
              <td class="py-4 px-4 align-top text-xs space-y-1.5">
                <div class="bg-slate-900/70 border border-slate-800/90 rounded-lg p-2 space-y-1">
                  <div class="text-[10px] font-mono uppercase text-slate-400 font-semibold">Smart Analysis Gold Playbook:</div>
                  <div class="text-slate-300 leading-snug">
                    <strong class="text-red-300">If Hot/Higher:</strong> {{ ev.smart_rule_hot }}
                  </div>
                  <div class="text-slate-300 leading-snug">
                    <strong class="text-emerald-300">If Cool/Lower:</strong> {{ ev.smart_rule_cool }}
                  </div>
                </div>
                <div class="flex flex-wrap items-center gap-1.5 text-[10px] font-mono text-slate-400 pt-0.5">
                  <span class="px-1.5 py-0.5 rounded bg-slate-900 border border-slate-800 text-slate-300">
                    <strong>1m:</strong> {{ ev.ml_1m }}
                  </span>
                  <span class="px-1.5 py-0.5 rounded bg-slate-900 border border-slate-800 text-slate-300">
                    <strong>30m:</strong> {{ ev.ml_30m }}
                  </span>
                  <span class="px-1.5 py-0.5 rounded bg-slate-900 border border-slate-800 text-slate-300">
                    <strong>1h:</strong> {{ ev.ml_1h }}
                  </span>
                </div>
              </td>

              <!-- Column 6: NewsGPT Gold Bias & Commentary -->
              <td class="py-4 px-4 align-top">
                <span class="inline-block px-2.5 py-1 rounded-md text-xs font-mono font-bold border mb-1.5
                  {% if ev.smart_gold_bias == 'BULLISH GOLD' %}
                    bg-emerald-500/15 text-emerald-300 border-emerald-500/40
                  {% elif ev.smart_gold_bias == 'BEARISH GOLD' %}
                    bg-red-500/15 text-red-300 border-red-500/40
                  {% else %}
                    bg-amber-500/15 text-amber-300 border-amber-500/40
                  {% endif %}">
                  {{ ev.smart_gold_bias }}
                </span>
                <p class="text-xs text-slate-300 leading-relaxed">
                  {{ ev.newsgpt_gold_commentary }}
                </p>
              </td>
            </tr>
            {% endfor %}
          </tbody>
        </table>
      </div>
    </section>
    {% endif %}

    <footer class="text-center text-xs font-mono text-slate-500 py-4 border-t border-slate-800/80">
      Gold (XAUUSD / COMEX GC=F) Intraday News &amp; Macro Sentiment Trading Agent • Quantitative Decision Support Terminal
    </footer>

  </div>

  <!-- ===================================================================== -->
  <!-- CHART.JS INITIALIZATION SCRIPT                                        -->
  <!-- ===================================================================== -->
  <script>
    (function() {
      const wedgeData = {{ vm.chart_wedge_json | safe }};
      const doughnutData = {{ vm.chart_doughnut_json | safe }};
      const initialPortalAuth = {{ vm.portal_auth_json | safe }};

      // ===================================================================
      // MULTI-USER & SINGLE-ADMIN PORTAL CONTROLLER (CODE-PERSISTED)
      // ===================================================================
      const STORAGE_USERS_KEY = 'xauusd_portal_code_users_v1';
      const STORAGE_SESSION_KEY = 'xauusd_portal_active_session_v1';

      let portalAdmin = initialPortalAuth.admin;
      let portalUsers = Array.isArray(initialPortalAuth.users) ? initialPortalAuth.users.slice() : [];

      // Merge any locally cached users if running in standalone file:// mode
      try {
        const cachedRaw = localStorage.getItem(STORAGE_USERS_KEY);
        if (cachedRaw && window.location.protocol === 'file:') {
          const parsedUsers = JSON.parse(cachedRaw);
          if (Array.isArray(parsedUsers) && parsedUsers.length > 0) {
            portalUsers = parsedUsers;
          }
        }
      } catch (e) {}

      function syncUsersFromServerIfAvailable() {
        if (window.location.protocol === 'file:') return;
        fetch('/api/users', { method: 'GET', headers: { 'Accept': 'application/json' } })
          .then(function(r) { return r.ok ? r.json() : null; })
          .then(function(data) {
            if (data && Array.isArray(data.users)) {
              portalUsers = data.users;
              if (data.admin) portalAdmin = data.admin;
              localStorage.setItem(STORAGE_USERS_KEY, JSON.stringify(portalUsers));
              updatePortalUI();
            }
          })
          .catch(function() {});
      }

      function getActiveSession() {
        try {
          const raw = sessionStorage.getItem(STORAGE_SESSION_KEY) || localStorage.getItem(STORAGE_SESSION_KEY);
          return raw ? JSON.parse(raw) : null;
        } catch (e) {
          return null;
        }
      }

      function setActiveSession(sessionObj) {
        try {
          if (sessionObj) {
            const s = JSON.stringify(sessionObj);
            sessionStorage.setItem(STORAGE_SESSION_KEY, s);
            localStorage.setItem(STORAGE_SESSION_KEY, s);
          } else {
            sessionStorage.removeItem(STORAGE_SESSION_KEY);
            localStorage.removeItem(STORAGE_SESSION_KEY);
          }
        } catch (e) {}
      }

      function showAdminBanner(msg, isError) {
        const banner = document.getElementById('adminStatusBanner');
        if (!banner) return;
        banner.textContent = msg;
        banner.className = 'rounded-lg p-3 text-xs font-mono border ' + (
          isError
            ? 'bg-red-500/15 border-red-500/40 text-red-300'
            : 'bg-emerald-500/15 border-emerald-500/40 text-emerald-300'
        );
        banner.classList.remove('hidden');
      }

      function renderAdminUsersTable() {
        const tbody = document.getElementById('adminUsersTableBody');
        const navCount = document.getElementById('navUserCount');
        const modalCount = document.getElementById('modalUserCount');
        if (navCount) navCount.textContent = String(portalUsers.length);
        if (modalCount) modalCount.textContent = String(portalUsers.length);
        if (!tbody) return;

        let html = '';
        // Row 0: Locked Single Administrator Account
        html += '<tr class="bg-amber-500/10">' +
          '<td class="py-2.5 px-3 font-bold text-amber-300">' + portalAdmin.username + '</td>' +
          '<td class="py-2.5 px-3 text-slate-300">' + portalAdmin.password + '</td>' +
          '<td class="py-2.5 px-3 text-slate-200">' + (portalAdmin.display_name || 'Administrator') + '</td>' +
          '<td class="py-2.5 px-3"><span class="px-2 py-0.5 rounded bg-amber-500/20 text-amber-300 border border-amber-500/40 text-[10px] font-bold">SINGLE ADMIN</span></td>' +
          '<td class="py-2.5 px-3 text-center"><span class="px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 text-[10px] font-bold">ACTIVE</span></td>' +
          '<td class="py-2.5 px-3 text-right text-slate-500 text-[10px]">Locked Owner</td>' +
        '</tr>';

        portalUsers.forEach(function(u) {
          const isActive = u.active !== false;
          const safeUser = String(u.username || '').replace(/'/g, "\\'");
          html += '<tr class="hover:bg-slate-900/60">' +
            '<td class="py-2.5 px-3 font-bold text-white">' + u.username + '</td>' +
            '<td class="py-2.5 px-3 text-amber-300">' + u.password + '</td>' +
            '<td class="py-2.5 px-3 text-slate-300">' + (u.display_name || u.username) + '</td>' +
            '<td class="py-2.5 px-3"><span class="px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700 text-[10px]">USER</span></td>' +
            '<td class="py-2.5 px-3 text-center">' +
              (isActive
                ? '<span class="px-2 py-0.5 rounded bg-emerald-500/15 text-emerald-300 border border-emerald-500/30 text-[10px] font-bold">ACTIVE</span>'
                : '<span class="px-2 py-0.5 rounded bg-red-500/15 text-red-300 border border-red-500/30 text-[10px] font-bold">DISABLED</span>') +
            '</td>' +
            '<td class="py-2.5 px-3 text-right space-x-1.5">' +
              '<button type="button" onclick="window.adminToggleUser(\'' + safeUser + '\')" class="px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 text-[10px]">' +
                (isActive ? 'Disable' : 'Enable') +
              '</button>' +
              '<button type="button" onclick="window.adminDeleteUser(\'' + safeUser + '\')" class="px-2 py-1 rounded bg-red-500/15 hover:bg-red-500/30 text-red-300 border border-red-500/30 text-[10px]">' +
                'Delete' +
              '</button>' +
            '</td>' +
          '</tr>';
        });

        tbody.innerHTML = html;
      }

      function updatePortalUI() {
        const session = getActiveSession();
        const overlay = document.getElementById('loginOverlay');
        const userBadge = document.getElementById('currentUserBadge');
        const adminBtn = document.getElementById('btnOpenAdminModal');

        renderAdminUsersTable();

        if (!session || !session.username) {
          if (overlay) overlay.classList.remove('hidden');
          if (userBadge) userBadge.textContent = 'Not Signed In';
          if (adminBtn) adminBtn.classList.add('hidden');
          return;
        }

        if (overlay) overlay.classList.add('hidden');
        if (session.role === 'admin') {
          if (userBadge) {
            userBadge.innerHTML = '&#128081; <strong>ADMIN:</strong> ' + session.username + ' (' + (session.display_name || 'Administrator') + ')';
            userBadge.className = 'px-2.5 py-1 rounded-md text-xs font-mono bg-amber-500/20 text-amber-300 border border-amber-500/40';
          }
          if (adminBtn) adminBtn.classList.remove('hidden');
        } else {
          if (userBadge) {
            userBadge.innerHTML = '&#128100; <strong>USER:</strong> ' + session.username + ' (' + (session.display_name || session.username) + ')';
            userBadge.className = 'px-2.5 py-1 rounded-md text-xs font-mono bg-emerald-500/15 text-emerald-300 border border-emerald-500/30';
          }
          if (adminBtn) adminBtn.classList.add('hidden');
        }
      }

      window.fillLogin = function(u, p) {
        const uEl = document.getElementById('loginUsername');
        const pEl = document.getElementById('loginPassword');
        if (uEl) uEl.value = u;
        if (pEl) pEl.value = p;
      };

      window.handlePortalLogin = function(ev) {
        if (ev) ev.preventDefault();
        const u = (document.getElementById('loginUsername').value || '').trim();
        const p = (document.getElementById('loginPassword').value || '').trim();
        const errBox = document.getElementById('loginErrorBox');
        if (errBox) errBox.classList.add('hidden');

        // 1. Check against local/embedded code store immediately
        if (u.toLowerCase() === portalAdmin.username.toLowerCase() && p === portalAdmin.password) {
          setActiveSession({
            username: portalAdmin.username,
            display_name: portalAdmin.display_name,
            role: 'admin'
          });
          updatePortalUI();
          return false;
        }

        const matchedUser = portalUsers.find(function(item) {
          return String(item.username || '').toLowerCase() === u.toLowerCase() && item.password === p;
        });

        if (matchedUser) {
          if (matchedUser.active === false) {
            if (errBox) {
              errBox.textContent = 'This user account has been disabled by the Administrator.';
              errBox.classList.remove('hidden');
            }
            return false;
          }
          setActiveSession({
            username: matchedUser.username,
            display_name: matchedUser.display_name || matchedUser.username,
            role: 'user'
          });
          updatePortalUI();
          return false;
        }

        // 2. If running on web server, also verify against live /api/login
        if (window.location.protocol !== 'file:') {
          fetch('/api/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username: u, password: p })
          })
            .then(function(r) { return r.json().then(function(d) { return { status: r.status, body: d }; }); })
            .then(function(res) {
              if (res.status === 200 && res.body && res.body.user) {
                setActiveSession(res.body.user);
                syncUsersFromServerIfAvailable();
                updatePortalUI();
              } else if (errBox) {
                errBox.textContent = (res.body && res.body.error) || 'Invalid username or password.';
                errBox.classList.remove('hidden');
              }
            })
            .catch(function() {
              if (errBox) {
                errBox.textContent = 'Invalid username or password.';
                errBox.classList.remove('hidden');
              }
            });
          return false;
        }

        if (errBox) {
          errBox.textContent = 'Invalid username or password. Check users_store.py credentials.';
          errBox.classList.remove('hidden');
        }
        return false;
      };

      window.logoutGoldPortal = function() {
        setActiveSession(null);
        window.closeAdminModal();
        updatePortalUI();
      };

      window.openAdminModal = function() {
        const session = getActiveSession();
        if (!session || session.role !== 'admin') return;
        renderAdminUsersTable();
        const modal = document.getElementById('adminModal');
        if (modal) modal.classList.remove('hidden');
      };

      window.closeAdminModal = function() {
        const modal = document.getElementById('adminModal');
        if (modal) modal.classList.add('hidden');
      };

      window.handleAdminCreateUser = function(ev) {
        if (ev) ev.preventDefault();
        const session = getActiveSession();
        if (!session || session.role !== 'admin') return false;

        const uname = (document.getElementById('newUsername').value || '').trim();
        const pwd = (document.getElementById('newPassword').value || '').trim();
        const dname = (document.getElementById('newDisplayName').value || '').trim() || uname;

        if (!uname || !pwd) return false;
        if (uname.toLowerCase() === portalAdmin.username.toLowerCase()) {
          showAdminBanner("Cannot overwrite the single Administrator username ('admin').", true);
          return false;
        }

        if (window.location.protocol !== 'file:') {
          fetch('/api/users', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              action: 'create_or_update',
              username: uname,
              password: pwd,
              display_name: dname
            })
          })
            .then(function(r) { return r.json().then(function(d) { return { ok: r.ok, body: d }; }); })
            .then(function(res) {
              if (res.ok && res.body && Array.isArray(res.body.users)) {
                portalUsers = res.body.users;
                localStorage.setItem(STORAGE_USERS_KEY, JSON.stringify(portalUsers));
                renderAdminUsersTable();
                document.getElementById('newUsername').value = '';
                document.getElementById('newPassword').value = '';
                document.getElementById('newDisplayName').value = '';
                showAdminBanner("Saved user '" + uname + "' directly into users_store.py code!", false);
              } else {
                showAdminBanner((res.body && res.body.error) || 'Failed to save user.', true);
              }
            })
            .catch(function(err) {
              showAdminBanner('Server error: ' + err, true);
            });
          return false;
        }

        // Standalone file:// mode fallback
        const nowStr = new Date().toISOString().slice(0, 16).replace('T', ' ') + ' UTC';
        const existingIdx = portalUsers.findIndex(function(u) {
          return String(u.username || '').toLowerCase() === uname.toLowerCase();
        });
        if (existingIdx >= 0) {
          portalUsers[existingIdx].username = uname;
          portalUsers[existingIdx].password = pwd;
          portalUsers[existingIdx].display_name = dname;
          portalUsers[existingIdx].active = true;
        } else {
          portalUsers.push({
            username: uname,
            password: pwd,
            display_name: dname,
            role: 'user',
            active: true,
            created_at: nowStr
          });
        }
        localStorage.setItem(STORAGE_USERS_KEY, JSON.stringify(portalUsers));
        renderAdminUsersTable();
        document.getElementById('newUsername').value = '';
        document.getElementById('newPassword').value = '';
        document.getElementById('newDisplayName').value = '';
        showAdminBanner("Added user '" + uname + "' (Run `python app.py` or click Export `users_store.py` to save to disk).", false);
        return false;
      };

      window.adminToggleUser = function(username) {
        if (window.location.protocol !== 'file:') {
          fetch('/api/users', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action: 'toggle', username: username })
          })
            .then(function(r) { return r.json(); })
            .then(function(data) {
              if (data && Array.isArray(data.users)) {
                portalUsers = data.users;
                localStorage.setItem(STORAGE_USERS_KEY, JSON.stringify(portalUsers));
                renderAdminUsersTable();
                showAdminBanner("Updated status for '" + username + "' in users_store.py.", false);
              }
            });
          return;
        }
        portalUsers.forEach(function(u) {
          if (String(u.username || '').toLowerCase() === String(username).toLowerCase()) {
            u.active = !(u.active !== false);
          }
        });
        localStorage.setItem(STORAGE_USERS_KEY, JSON.stringify(portalUsers));
        renderAdminUsersTable();
      };

      window.adminDeleteUser = function(username) {
        if (window.location.protocol !== 'file:') {
          fetch('/api/users', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action: 'delete', username: username })
          })
            .then(function(r) { return r.json(); })
            .then(function(data) {
              if (data && Array.isArray(data.users)) {
                portalUsers = data.users;
                localStorage.setItem(STORAGE_USERS_KEY, JSON.stringify(portalUsers));
                renderAdminUsersTable();
                showAdminBanner("Deleted user '" + username + "' from users_store.py.", false);
              }
            });
          return;
        }
        portalUsers = portalUsers.filter(function(u) {
          return String(u.username || '').toLowerCase() !== String(username).toLowerCase();
        });
        localStorage.setItem(STORAGE_USERS_KEY, JSON.stringify(portalUsers));
        renderAdminUsersTable();
      };

      window.downloadUsersStorePy = function() {
        let pyLines = [
          '# Code-Persisted User & Administrator Store (users_store.py)',
          '',
          'ADMIN_ACCOUNT = ' + JSON.stringify(portalAdmin, null, 4).replace(/true/g, 'True').replace(/false/g, 'False'),
          '',
          '# BEGIN_USERS_LIST',
          'USERS_LIST = ' + JSON.stringify(portalUsers, null, 4).replace(/true/g, 'True').replace(/false/g, 'False'),
          '# END_USERS_LIST',
          ''
        ];
        const blob = new Blob([pyLines.join('\n')], { type: 'text/x-python' });
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = 'users_store.py';
        a.click();
      };

      window.triggerGoldPortalRefresh = function(forceQuantOnly) {
        if (window.location.protocol === 'file:') {
          window.location.reload();
          return;
        }
        const btnLive = document.getElementById('btnRefreshLive');
        const btnQuant = document.getElementById('btnRefreshQuant');
        const spinner = document.getElementById('refreshSpinner');
        if (spinner) spinner.classList.remove('hidden');
        if (btnLive) btnLive.disabled = true;
        if (btnQuant) btnQuant.disabled = true;

        fetch('/api/refresh', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ fetch_only: Boolean(forceQuantOnly) })
        })
          .then(function(r) { return r.json(); })
          .then(function(res) {
            window.location.reload();
          })
          .catch(function() {
            window.location.reload();
          });
      };

      updatePortalUI();
      syncUsersFromServerIfAvailable();

      Chart.defaults.color = '#94a3b8';
      Chart.defaults.font.family = "'JetBrains Mono', 'Inter', sans-serif";

      // Chart 1: Horizontal Bar Chart of the 4 Gold Impact Wedges (-100 to +100)
      const barCtx = document.getElementById('wedgeBarChart');
      if (barCtx) {
        new Chart(barCtx, {
          type: 'bar',
          data: {
            labels: wedgeData.labels,
            datasets: [{
              label: 'Net Gold Sentiment Score (-100 to +100)',
              data: wedgeData.scores,
              backgroundColor: wedgeData.bgColors,
              borderColor: wedgeData.borderColors,
              borderWidth: 1.5,
              borderRadius: 6,
              barThickness: 28
            }]
          },
          options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            scales: {
              x: {
                min: -100,
                max: 100,
                grid: {
                  color: 'rgba(51, 65, 85, 0.45)',
                  zeroLineColor: '#f59e0b'
                },
                ticks: {
                  callback: function(value) { return (value > 0 ? '+' : '') + value; }
                }
              },
              y: {
                grid: { display: false },
                ticks: { color: '#e2e8f0', font: { size: 11, weight: '600' } }
              }
            },
            plugins: {
              legend: { display: false },
              tooltip: {
                backgroundColor: '#0f172a',
                borderColor: '#334155',
                borderWidth: 1,
                callbacks: {
                  label: function(context) {
                    const v = context.parsed.x;
                    return ' Net Gold Wedge Score: ' + (v > 0 ? '+' : '') + v.toFixed(1) + ' / 100';
                  }
                }
              }
            }
          }
        });
      }

      // Chart 2: Doughnut Chart of Weighted Bullish vs. Bearish vs. Neutral News
      const doughnutCtx = document.getElementById('newsDoughnutChart');
      if (doughnutCtx) {
        new Chart(doughnutCtx, {
          type: 'doughnut',
          data: {
            labels: doughnutData.labels,
            datasets: [{
              data: doughnutData.weights,
              backgroundColor: [
                'rgba(16, 185, 129, 0.80)',
                'rgba(239, 68, 68, 0.80)',
                'rgba(245, 158, 11, 0.80)'
              ],
              borderColor: [
                '#10b981',
                '#ef4444',
                '#f59e0b'
              ],
              borderWidth: 2,
              hoverOffset: 6
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '64%',
            plugins: {
              legend: {
                position: 'bottom',
                labels: {
                  color: '#e2e8f0',
                  padding: 14,
                  font: { size: 11 }
                }
              },
              tooltip: {
                backgroundColor: '#0f172a',
                borderColor: '#334155',
                borderWidth: 1,
                callbacks: {
                  label: function(context) {
                    return ' Total Impact Weight: ' + context.parsed;
                  }
                }
              }
            }
          }
        });
      }
    })();
  </script>
</body>
</html>
"""


def generate_html_report(
    payload: Dict[str, Any],
    analysis: Optional[Dict[str, Any]] = None,
    output_path: str = DEFAULT_REPORT_FILENAME,
) -> str:
    """
    Render the standalone Gold (`XAUUSD` / `GC=F`) trading terminal HTML report
    to `xauusd_intraday_report.html` and return the resolved absolute file path.
    """
    if analysis is None:
        from analyzer import build_deterministic_gold_analysis

        analysis = build_deterministic_gold_analysis(payload)
        analysis["analysis_engine"] = "deterministic_fallback"

    view_model = _prepare_report_view_model(payload, analysis)

    env = Environment(loader=BaseLoader(), autoescape=True)
    template = env.from_string(_TERMINAL_HTML_TEMPLATE)
    rendered_html = template.render(
        payload=payload,
        analysis=analysis,
        vm=view_model,
    )

    out_file = Path(output_path).resolve()
    out_file.write_text(rendered_html, encoding="utf-8")
    return str(out_file)
