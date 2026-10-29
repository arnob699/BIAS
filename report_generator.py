"""
Gold (`XAUUSD` / COMEX `GC=F`) Desk Research & Intraday Portal Generator (`report_generator.py`).

Renders `xauusd_intraday_report.html` using a warm, creamy, human-crafted editorial layout
(inspired by classic financial journals and bullion desk morning briefs) with clear
hierarchy, natural paper/ink contrast, and intuitive member management.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

from jinja2 import BaseLoader, Environment

from users_store import get_admin_account, get_all_users

DEFAULT_REPORT_FILENAME = "xauusd_intraday_report.html"
BST_TZ = timezone(timedelta(hours=6), name="BST")


def _convert_any_utc_to_bst(text: str) -> str:
    """
    Convert any ISO-8601 UTC timestamp (`2026-10-29T11:08:50Z`) or `YYYY-MM-DD HH:MM UTC`
    found inside `text` into Bangladesh Standard Time (`DD Mon YYYY, hh:mm AM/PM BST`).
    """
    if not text:
        return ""
    s = str(text).strip()

    # Match ISO UTC: 2026-10-29T11:08:50Z
    def _repl_iso(m: re.Match) -> str:
        raw_iso = m.group(0)
        try:
            dt = datetime.fromisoformat(raw_iso.replace("Z", "+00:00"))
            dt_bst = dt.astimezone(BST_TZ)
            if dt.second:
                return dt_bst.strftime("%d %b %Y, %I:%M:%S %p BST")
            return dt_bst.strftime("%d %b %Y, %I:%M %p BST")
        except Exception:
            return raw_iso

    s = re.sub(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?Z", _repl_iso, s)

    # Match YYYY-MM-DD HH:MM UTC
    def _repl_utc(m: re.Match) -> str:
        ymd_hm = m.group(1)
        try:
            dt = datetime.strptime(ymd_hm, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
            return dt.astimezone(BST_TZ).strftime("%d %b %Y, %I:%M %p BST")
        except Exception:
            return m.group(0)

    s = re.sub(r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})\s*UTC", _repl_utc, s)
    return s



def _derive_display_bias_badge(overall_bias: str) -> Dict[str, str]:
    """
    Map `overall_bias` to clean, human-friendly badge labels and warm editorial colors.
    """
    normalized = (overall_bias or "NEUTRAL / CHOPPY").upper().strip()
    if normalized == "STRONG BUY":
        return {
            "label": "STRONG BUY • XAUUSD",
            "raw": normalized,
            "bg_class": "bg-[#E6F2EC] border-[#1B6B45] text-[#145235]",
            "dot_class": "bg-[#1B6B45]",
            "accent_hex": "#1B6B45",
            "bar_class": "bg-[#1B6B45]",
        }
    if normalized == "BUY":
        return {
            "label": "BUY • XAUUSD",
            "raw": normalized,
            "bg_class": "bg-[#EBF5F0] border-[#287D54] text-[#1B6B45]",
            "dot_class": "bg-[#287D54]",
            "accent_hex": "#287D54",
            "bar_class": "bg-[#287D54]",
        }
    if normalized == "STRONG SELL":
        return {
            "label": "STRONG SELL • XAUUSD",
            "raw": normalized,
            "bg_class": "bg-[#FAEDED] border-[#A82828] text-[#851D1D]",
            "dot_class": "bg-[#A82828]",
            "accent_hex": "#A82828",
            "bar_class": "bg-[#A82828]",
        }
    if normalized == "SELL":
        return {
            "label": "SELL • XAUUSD",
            "raw": normalized,
            "bg_class": "bg-[#FDF2F2] border-[#B83232] text-[#A82828]",
            "dot_class": "bg-[#B83232]",
            "accent_hex": "#B83232",
            "bar_class": "bg-[#B83232]",
        }
    return {
        "label": "NEUTRAL / RANGE • XAUUSD",
        "raw": normalized,
        "bg_class": "bg-[#FBF3E0] border-[#B47B1C] text-[#8A5A0E]",
        "dot_class": "bg-[#B47B1C]",
        "accent_hex": "#B47B1C",
        "bar_class": "bg-[#B47B1C]",
    }


def _derive_confluence_status(
    analysis: Dict[str, Any],
    payload: Dict[str, Any],
) -> Dict[str, str]:
    """
    Build a plain-English desk summary tag based on conviction and cross-asset alignment.
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
            "label": "Strong Alignment Across Yields, Dollar & News",
            "badge_class": "bg-[#E6F2EC] text-[#1B6B45] border-[#B8D9C8]",
            "description": "Treasury yields (^TNX), the US Dollar (DXY), and breaking macro news are all pointing in the same direction for Gold.",
        }
    if w1_score < -10 and w3_score > 15:
        return {
            "label": "Firm Yields Weighing Against Safe-Haven Demand",
            "badge_class": "bg-[#FBF3E0] text-[#8A5A0E] border-[#E6CFA0]",
            "description": (
                f"Higher US 10Y yields ({tnx_price:.3f}%, {tnx_chg:+.2f}%) and a steady Dollar "
                f"(DXY {dxy_price:.2f}, {dxy_chg:+.2f}%) cap upside, while geopolitical buying offers support on dips."
            ),
        }
    if w1_score > 10 and w3_score < -10:
        return {
            "label": "Supportive Yields Offset by Calmer Geopolitics",
            "badge_class": "bg-[#FBF3E0] text-[#8A5A0E] border-[#E6CFA0]",
            "description": "Softer yields and Dollar support Gold, though fading safe-haven urgency slows breakout momentum.",
        }
    if conf >= 65:
        return {
            "label": "Constructive Setup with Selective Cross-Currents",
            "badge_class": "bg-[#EDF3F8] text-[#2B547E] border-[#C2D6E8]",
            "description": "Primary macro drivers lean in favor of the desk bias, though traders should respect intraday support and resistance.",
        }
    return {
        "label": "Two-Way Market: Mixed Rate & Safe-Haven Signals",
        "badge_class": "bg-[#FBF3E0] text-[#8A5A0E] border-[#E6CFA0]",
        "description": "Competing signals between Treasury yields and physical/safe-haven flows keep Gold in a range-trading environment.",
    }


def _prepare_report_view_model(
    payload: Dict[str, Any],
    analysis: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Prepare template variables, sorted news rows, and Chart.js datasets.
    """
    bias_meta = _derive_display_bias_badge(analysis.get("overall_bias", "NEUTRAL / CHOPPY"))
    confluence_meta = _derive_confluence_status(analysis, payload)

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

    raw_news_list: List[Dict[str, Any]] = list(payload.get("intraday_breaking_news") or [])
    raw_meta_map: Dict[str, Dict[str, Any]] = {}
    for raw_item in raw_news_list:
        hl = (raw_item.get("headline") or "").strip().lower()
        if hl:
            raw_meta_map[hl] = raw_item

    fallback_bst_now = _convert_any_utc_to_bst(payload.get("generated_at_utc", "")) or datetime.now(BST_TZ).strftime("%d %b %Y, %I:%M %p BST")

    analyzed_news: List[Dict[str, Any]] = []
    for idx_item, item in enumerate(analysis.get("news_items") or []):
        item_copy = dict(item)
        hl_key = (item_copy.get("headline") or "").strip().lower()
        matched_raw = raw_meta_map.get(hl_key)

        # Fuzzy fallback match by word overlap if LLM slightly edited headline
        if matched_raw is None and hl_key:
            hl_words = set(re.findall(r"[a-z0-9]{3,}", hl_key))
            best_score = 0.0
            for r_item in raw_news_list:
                r_words = set(re.findall(r"[a-z0-9]{3,}", (r_item.get("headline") or "").lower()))
                if hl_words and r_words:
                    overlap = len(hl_words & r_words) / max(1, min(len(hl_words), len(r_words)))
                    if overlap > best_score and overlap >= 0.5:
                        best_score = overlap
                        matched_raw = r_item

        if matched_raw is None and idx_item < len(raw_news_list):
            matched_raw = raw_news_list[idx_item]

        item_copy["url"] = (matched_raw or {}).get("url", "")
        raw_source = (matched_raw or {}).get("source") or str(item_copy.get("source_time") or "Financial Desk").split("|")[0].strip()
        raw_pub_bst = (matched_raw or {}).get("published_bst") or _convert_any_utc_to_bst((matched_raw or {}).get("published_utc", ""))
        if not raw_pub_bst:
            converted_st = _convert_any_utc_to_bst(str(item_copy.get("source_time") or ""))
            if "BST" in converted_st:
                raw_pub_bst = converted_st.split("|")[-1].strip()
            else:
                raw_pub_bst = fallback_bst_now

        raw_age = (matched_raw or {}).get("age_human", "")
        if "," in raw_pub_bst:
            bst_date_part, bst_time_part = [p.strip() for p in raw_pub_bst.split(",", 1)]
        else:
            bst_date_part, bst_time_part = raw_pub_bst, ""

        item_copy["source_name"] = raw_source
        item_copy["bst_datetime"] = raw_pub_bst
        item_copy["bst_date"] = bst_date_part
        item_copy["bst_time"] = bst_time_part
        item_copy["age_human"] = raw_age
        item_copy["source_time"] = f"{raw_source} • {raw_pub_bst}" + (f" ({raw_age})" if raw_age and raw_age not in raw_pub_bst else "")
        analyzed_news.append(item_copy)

    analyzed_news.sort(
        key=lambda x: (
            int(x.get("impact_weight", 1)),
            abs(int(x.get("sentiment_score", 0))),
        ),
        reverse=True,
    )

    wedges: List[Dict[str, Any]] = list(analysis.get("wedge_breakdown") or [])
    wedge_labels: List[str] = []
    wedge_scores: List[float] = []
    wedge_colors: List[str] = []
    wedge_borders: List[str] = []

    for idx, w in enumerate(wedges, start=1):
        pct = w.get("weight_percentage", 0)
        name = w.get("wedge_name", f"Factor {idx}")
        short_label = name.split("(")[0].strip()
        if len(short_label) > 42:
            short_label = short_label[:40] + "…"
        wedge_labels.append(f"{short_label} ({pct}%)")

        score = round(float(w.get("net_sentiment_score", 0.0)), 1)
        wedge_scores.append(score)
        if score > 5.0:
            wedge_colors.append("rgba(27, 107, 69, 0.78)")
            wedge_borders.append("#1B6B45")
        elif score < -5.0:
            wedge_colors.append("rgba(168, 40, 40, 0.78)")
            wedge_borders.append("#A82828")
        else:
            wedge_colors.append("rgba(180, 123, 28, 0.78)")
            wedge_borders.append("#B47B1C")

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

    weighted_score = float(analysis.get("weighted_sentiment_score", 0.0))
    sentiment_gauge_pct = max(2.0, min(98.0, round((weighted_score + 100.0) / 2.0, 1)))

    raw_cal = dict(payload.get("gold_economic_calendar") or {
        "source_status": "unavailable",
        "target_pair": "XAUUSD / COMEX GC=F",
        "total_gold_events": 0,
        "high_impact_count": 0,
        "medium_impact_count": 0,
        "next_major_gold_catalyst": "N/A",
        "calendar_macro_tilt": "NEUTRAL",
        "events": [],
    })
    raw_cal["next_major_gold_catalyst"] = _convert_any_utc_to_bst(raw_cal.get("next_major_gold_catalyst", "N/A"))
    cal_events_bst: List[Dict[str, Any]] = []
    for ev in raw_cal.get("events") or []:
        ev_copy = dict(ev)
        ev_copy["date_utc"] = _convert_any_utc_to_bst(ev_copy.get("date_utc", "")) or _convert_any_utc_to_bst(ev_copy.get("date_iso", ""))
        cal_events_bst.append(ev_copy)
    raw_cal["events"] = cal_events_bst
    gold_calendar = raw_cal
    if False:
        gold_calendar = {
        "source_status": "unavailable",
        "target_pair": "XAUUSD / COMEX GC=F",
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
        "generated_at_bst": _convert_any_utc_to_bst(payload.get("generated_at_bst") or payload.get("generated_at_utc") or ""),
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
                    f"Bullish ({count_dist['BULLISH GOLD']} stories | {doughnut_pcts['BULLISH GOLD']}%)",
                    f"Bearish ({count_dist['BEARISH GOLD']} stories | {doughnut_pcts['BEARISH GOLD']}%)",
                    f"Neutral ({count_dist['NEUTRAL']} stories | {doughnut_pcts['NEUTRAL']}%)",
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
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>XAUUSD • Gold Desk Morning &amp; Intraday Research Brief</title>
  <!-- Tailwind CSS CDN -->
  <script src="https://cdn.tailwindcss.com"></script>
  <!-- Chart.js CDN -->
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <!-- Google Fonts: Newsreader (Editorial Serif), Plus Jakarta Sans (Clean UI), IBM Plex Mono (Numbers) -->
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Newsreader:ital,opsz,wght@0,6..72,500;0,6..72,600;0,6..72,700;1,6..72,400&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
  <script>
    tailwind.config = {
      theme: {
        extend: {
          colors: {
            cream: {
              bg: '#F7F3EB',
              card: '#FFFDF9',
              sub: '#F0EAE0',
              border: '#DED6C6',
              ink: '#1F1C18',
              muted: '#655D52',
              brass: '#9A6B16',
              bull: '#1B6B45',
              bear: '#A82828'
            }
          },
          fontFamily: {
            serif: ['Newsreader', 'Georgia', 'Cambria', 'serif'],
            sans: ['Plus Jakarta Sans', '-apple-system', 'BlinkMacSystemFont', 'sans-serif'],
            mono: ['IBM Plex Mono', 'monospace']
          }
        }
      }
    }
  </script>
  <style>
    body {
      background-color: #F7F3EB;
      color: #1F1C18;
      font-family: 'Plus Jakarta Sans', sans-serif;
    }
    .paper-card {
      background-color: #FFFDF9;
      border: 1px solid #DED6C6;
      box-shadow: 0 2px 6px rgba(31, 28, 24, 0.04);
    }
    .paper-inset {
      background-color: #F3EDE2;
      border: 1px solid #E2D9C8;
    }
    .editorial-heading {
      font-family: 'Newsreader', Georgia, serif;
      letter-spacing: -0.015em;
    }
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: #F7F3EB; }
    ::-webkit-scrollbar-thumb { background: #CFC5B3; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #B5A892; }
  </style>
</head>
<body class="min-h-screen antialiased selection:bg-[#E6CFA0] selection:text-[#1F1C18]">

  <!-- ===================================================================== -->
  <!-- TOP NAVIGATION BAR (WARM CREAM EDITORIAL HEADER)                      -->
  <!-- ===================================================================== -->
  <nav id="portalTopNav" class="sticky top-0 z-40 bg-[#FAF6EE]/95 backdrop-blur-sm border-b border-[#DED6C6] px-4 sm:px-6 lg:px-8 py-3">
    <div class="max-w-[1440px] mx-auto flex flex-wrap items-center justify-between gap-3">
      <div class="flex items-center gap-3 flex-wrap">
        <div class="flex items-center gap-2">
          <span class="w-7 h-7 rounded bg-[#1F1C18] text-[#FAF6EE] font-serif font-bold text-sm flex items-center justify-center">
            Au
          </span>
          <div>
            <div class="font-serif font-bold text-base sm:text-lg text-[#1F1C18] leading-tight">
              XAUUSD Gold Desk
            </div>
            <div class="text-[11px] font-medium text-[#9A6B16] leading-tight">
              Created by Arnob &bull; Computer Science Department, BRAC University
            </div>
          </div>
        </div>
        <span class="text-[#CFC5B3]">|</span>
        <span id="currentUserBadge" class="px-2.5 py-1 rounded text-xs font-medium bg-[#F0EAE0] text-[#4A433B] border border-[#DED6C6]">
          Not Signed In
        </span>
      </div>

      <div class="flex items-center gap-2.5 flex-wrap text-xs font-medium">
        <!-- Refresh Market Data -->
        <button id="btnRefreshLive" type="button" onclick="window.triggerGoldPortalRefresh(false)"
                class="px-3.5 py-1.5 rounded-md font-semibold bg-[#1F1C18] hover:bg-[#36312A] text-[#FAF6EE] transition flex items-center gap-1.5 shadow-sm">
          <span id="refreshSpinner" class="hidden animate-spin">&#8635;</span>
          <span>Update Market Data</span>
        </button>

        <button id="btnRefreshQuant" type="button" onclick="window.triggerGoldPortalRefresh(true)"
                title="Quickly refresh live prices and economic calendar"
                class="px-3 py-1.5 rounded-md font-medium bg-[#F0EAE0] hover:bg-[#E5DEC9] text-[#1F1C18] border border-[#D5CBB8] transition">
          Quick Quote Refresh
        </button>

        <!-- Admin-Only Button (Shown ONLY when signed in as arnob) -->
        <button id="btnOpenAdminModal" type="button" onclick="window.openAdminModal()"
                class="hidden px-3.5 py-1.5 rounded-md font-semibold bg-[#9A6B16] hover:bg-[#825910] text-white transition shadow-sm">
          Manage Members (<span id="navUserCount">0</span>)
        </button>

        <button id="btnLogout" type="button" onclick="window.logoutGoldPortal()"
                class="px-3 py-1.5 rounded-md font-medium bg-[#FFFDF9] hover:bg-[#FAEDED] text-[#5C5346] hover:text-[#A82828] border border-[#DED6C6] transition">
          Sign Out
        </button>
      </div>
    </div>
  </nav>

  <!-- ===================================================================== -->
  <!-- SIGN-IN OVERLAY (CLEAN CREAM CARD, MANUAL ACCESS ONLY)                -->
  <!-- ===================================================================== -->
  <div id="loginOverlay" class="fixed inset-0 z-50 bg-[#F7F3EB]/95 backdrop-blur-sm flex items-center justify-center p-4">
    <div class="paper-card rounded-xl max-w-md w-full p-7 sm:p-8 border-t-4 border-t-[#9A6B16] shadow-xl space-y-5">
      <div class="flex items-center gap-3.5 border-b border-[#EAE3D5] pb-4">
        <div class="w-11 h-11 rounded-lg bg-[#1F1C18] text-[#FAF6EE] flex items-center justify-center font-serif font-bold text-xl">
          Au
        </div>
        <div>
          <div class="text-[11px] font-semibold uppercase tracking-widest text-[#9A6B16]">
            Precious Metals Research Desk
          </div>
          <h2 class="editorial-heading text-2xl font-bold text-[#1F1C18] mt-0.5">
            Member Sign In
          </h2>
        </div>
      </div>

      <p class="text-xs sm:text-sm text-[#5C5346] leading-relaxed">
        Please enter your desk credentials to view today&rsquo;s Gold (<strong class="text-[#1F1C18]">XAUUSD / GC=F</strong>) research brief, live cross-asset quotes, and economic calendar.
      </p>

      <div id="loginErrorBox" class="hidden rounded-md bg-[#FAEDED] border border-[#E8C3C3] p-3 text-xs font-medium text-[#A82828]"></div>

      <form id="loginForm" onsubmit="return window.handlePortalLogin(event)" class="space-y-4">
        <div>
          <label class="block text-xs font-semibold uppercase tracking-wider text-[#4A433B] mb-1.5">Username</label>
          <input id="loginUsername" type="text" required autocomplete="username" placeholder="Enter your username"
                 class="w-full rounded-md bg-[#FFFDF9] border border-[#CFC5B3] focus:border-[#9A6B16] focus:outline-none px-3.5 py-2.5 text-sm text-[#1F1C18]" />
        </div>
        <div>
          <label class="block text-xs font-semibold uppercase tracking-wider text-[#4A433B] mb-1.5">Password</label>
          <input id="loginPassword" type="password" required autocomplete="current-password" placeholder="Enter your password"
                 class="w-full rounded-md bg-[#FFFDF9] border border-[#CFC5B3] focus:border-[#9A6B16] focus:outline-none px-3.5 py-2.5 text-sm text-[#1F1C18]" />
        </div>
        <button type="submit"
                class="w-full py-2.5 px-4 rounded-md bg-[#1F1C18] hover:bg-[#36312A] text-[#FAF6EE] font-semibold text-sm tracking-wide shadow-sm transition">
          Sign In to Gold Desk &rarr;
        </button>
      </form>

      <div class="pt-3 border-t border-[#EAE3D5] space-y-1 text-xs text-[#786E60]">
        <div class="font-semibold text-[#1F1C18]">
          Created by Arnob &bull; Computer Science Department, BRAC University
        </div>
        <div class="flex items-center justify-between text-[11px]">
          <span>Private Desk Access</span>
          <span>Contact Administrator for Registration</span>
        </div>
      </div>
    </div>
  </div>

  <!-- ===================================================================== -->
  <!-- MEMBER DIRECTORY MODAL (VISIBLE ONLY TO ADMIN: arnob)                 -->
  <!-- ===================================================================== -->
  <div id="adminModal" class="hidden fixed inset-0 z-50 bg-[#1F1C18]/50 backdrop-blur-xs flex items-center justify-center p-4 overflow-y-auto">
    <div class="paper-card rounded-xl max-w-3xl w-full p-6 sm:p-7 border-t-4 border-t-[#9A6B16] shadow-2xl space-y-5 my-8">
      <div class="flex items-start justify-between gap-4 border-b border-[#EAE3D5] pb-4">
        <div>
          <span class="text-[11px] font-semibold uppercase tracking-wider text-[#9A6B16]">
            Administrator Control Panel
          </span>
          <h3 class="editorial-heading text-xl font-bold text-[#1F1C18] mt-0.5">
            Manual Member Registration &amp; Access List
          </h3>
          <p class="text-xs text-[#655D52] mt-0.5">
            Register members manually below so they can sign in and read the Gold research report.
          </p>
        </div>
        <button type="button" onclick="window.closeAdminModal()"
                class="px-3 py-1.5 rounded-md bg-[#F0EAE0] hover:bg-[#E2D9C8] text-[#1F1C18] font-medium text-xs">
          &#10005; Close
        </button>
      </div>

      <div id="adminStatusBanner" class="hidden rounded-md p-3 text-xs font-medium border"></div>

      <!-- Manual Member Registration Form -->
      <form id="adminCreateUserForm" onsubmit="return window.handleAdminCreateUser(event)"
            class="paper-inset rounded-lg p-4 space-y-3.5">
        <div class="text-xs font-bold uppercase tracking-wider text-[#4A433B]">
          + Register New Member (or Update Password)
        </div>
        <div class="grid grid-cols-1 sm:grid-cols-3 gap-3">
          <div>
            <label class="block text-xs font-medium text-[#4A433B] mb-1">Username *</label>
            <input id="newUsername" type="text" required placeholder="e.g. rahim"
                   class="w-full rounded-md bg-[#FFFDF9] border border-[#CFC5B3] focus:border-[#9A6B16] focus:outline-none px-3 py-2 text-xs text-[#1F1C18]" />
          </div>
          <div>
            <label class="block text-xs font-medium text-[#4A433B] mb-1">Password *</label>
            <input id="newPassword" type="text" required placeholder="Enter password"
                   class="w-full rounded-md bg-[#FFFDF9] border border-[#CFC5B3] focus:border-[#9A6B16] focus:outline-none px-3 py-2 text-xs text-[#1F1C18]" />
          </div>
          <div>
            <label class="block text-xs font-medium text-[#4A433B] mb-1">Full Name / Desk Note</label>
            <input id="newDisplayName" type="text" placeholder="e.g. Senior Trader"
                   class="w-full rounded-md bg-[#FFFDF9] border border-[#CFC5B3] focus:border-[#9A6B16] focus:outline-none px-3 py-2 text-xs text-[#1F1C18]" />
          </div>
        </div>
        <div class="flex flex-wrap items-center justify-between gap-2 pt-1">
          <span class="text-xs text-[#655D52]">
            Administrator (<strong class="text-[#1F1C18]">arnob</strong>) has permanent full control.
          </span>
          <button type="submit"
                  class="px-4 py-2 rounded-md bg-[#1B6B45] hover:bg-[#145235] text-white font-semibold text-xs shadow-sm transition">
            + Register Member
          </button>
        </div>
      </form>

      <!-- Registered Members Table -->
      <div class="border border-[#DED6C6] rounded-lg overflow-hidden bg-[#FFFDF9]">
        <div class="bg-[#F0EAE0] px-4 py-2.5 border-b border-[#DED6C6] flex items-center justify-between text-xs">
          <span class="font-bold text-[#1F1C18]">Authorized Desk Accounts</span>
          <span class="text-[#655D52]">1 Administrator &bull; <strong id="modalUserCount" class="text-[#1F1C18]">0</strong> Registered Members</span>
        </div>
        <div class="overflow-x-auto max-h-72 overflow-y-auto">
          <table class="w-full text-left border-collapse text-xs">
            <thead>
              <tr class="bg-[#FAF6EE] text-[#655D52] border-b border-[#DED6C6] uppercase text-[11px]">
                <th class="py-2.5 px-3.5">Username</th>
                <th class="py-2.5 px-3.5">Password</th>
                <th class="py-2.5 px-3.5">Member Name</th>
                <th class="py-2.5 px-3.5">Role</th>
                <th class="py-2.5 px-3.5 text-center">Status</th>
                <th class="py-2.5 px-3.5 text-right">Actions</th>
              </tr>
            </thead>
            <tbody id="adminUsersTableBody" class="divide-y divide-[#EAE3D5]">
            </tbody>
          </table>
        </div>
      </div>
    </div>
  </div>

  <!-- ===================================================================== -->
  <!-- MAIN EDITORIAL PAGE CONTAINER                                         -->
  <!-- ===================================================================== -->
  <div class="max-w-[1440px] mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">

    <!-- =================================================================== -->
    <!-- EDITORIAL MASTHEAD                                                  -->
    <!-- =================================================================== -->
    <header class="paper-card rounded-xl px-6 py-5 border-t-4 border-t-[#1F1C18] flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
      <div>
        <div class="flex items-center gap-2.5 flex-wrap text-xs font-semibold uppercase tracking-widest text-[#9A6B16]">
          <span>Spot Gold (XAUUSD) &amp; COMEX Futures (GC=F)</span>
          <span>&bull;</span>
          <span>Intraday Macro &amp; Fundamental Desk Note</span>
        </div>
        <h1 class="editorial-heading text-2xl sm:text-3xl font-bold text-[#1F1C18] mt-1">
          Gold Daily Bias, Macro Drivers &amp; Economic Calendar
        </h1>
        <div class="mt-2 inline-flex flex-wrap items-center gap-2 px-3 py-1 rounded-md bg-[#FBF3E0] border border-[#E6CFA0] text-xs font-medium text-[#1F1C18]">
          <span class="font-bold text-[#8A5A0E]">Created by Arnob</span>
          <span class="text-[#B5A892]">&bull;</span>
          <span>Computer Science Department</span>
          <span class="text-[#B5A892]">&bull;</span>
          <span class="font-semibold">BRAC University</span>
        </div>
      </div>

      <div class="flex flex-wrap items-center gap-3 text-xs">
        <div class="paper-inset rounded-lg px-3.5 py-2">
          <div class="text-[#786E60] text-[10px] uppercase font-semibold tracking-wider">Last Updated (BST &bull; Bangladesh Standard Time)</div>
          <div class="text-[#1F1C18] font-mono font-semibold mt-0.5">{{ vm.generated_at_bst }}</div>
        </div>
        <div class="paper-inset rounded-lg px-3.5 py-2">
          <div class="text-[#786E60] text-[10px] uppercase font-semibold tracking-wider">Analysis Engine</div>
          <div class="text-[#9A6B16] font-mono font-semibold mt-0.5">{{ analysis.analysis_engine or "gemini-3.5-flash" }}</div>
        </div>
        <div class="paper-inset rounded-lg px-3.5 py-2">
          <div class="text-[#786E60] text-[10px] uppercase font-semibold tracking-wider">News Sources Checked</div>
          <div class="text-[#1B6B45] font-mono font-semibold mt-0.5">
            {{ payload.ingestion_diagnostics.active_feeds_returning_data }}/{{ payload.ingestion_diagnostics.total_feeds_queried }} Active Feeds
          </div>
        </div>
      </div>
    </header>

    <!-- =================================================================== -->
    <!-- 1. HERO DESK VERDICT & CONVICTION SUMMARY                           -->
    <!-- =================================================================== -->
    <section aria-label="Daily Desk Verdict" class="paper-card rounded-xl p-6 border-l-4"
             style="border-left-color: {{ vm.bias_meta.accent_hex }};">
      <div class="grid grid-cols-1 lg:grid-cols-12 gap-6 items-center">

        <!-- Left: Directional Call -->
        <div class="lg:col-span-4 flex flex-col justify-center space-y-2.5 border-b lg:border-b-0 lg:border-r border-[#EAE3D5] pb-5 lg:pb-0 lg:pr-6">
          <div class="text-xs font-semibold uppercase tracking-wider text-[#655D52]">
            Current Desk Call
          </div>
          <div class="inline-flex items-center gap-2.5 self-start px-4 py-2.5 rounded-lg border-2 {{ vm.bias_meta.bg_class }}">
            <span class="w-3 h-3 rounded-full {{ vm.bias_meta.dot_class }}"></span>
            <span class="font-serif text-xl sm:text-2xl font-bold tracking-tight">
              {{ vm.bias_meta.label }}
            </span>
          </div>
          <div class="pt-1">
            <span class="inline-flex items-center px-2.5 py-1 rounded text-xs font-semibold border {{ vm.confluence_meta.badge_class }}">
              {{ vm.confluence_meta.label }}
            </span>
            <p class="text-xs text-[#5C5346] mt-1.5 leading-relaxed">
              {{ vm.confluence_meta.description }}
            </p>
          </div>
        </div>

        <!-- Middle: Net Sentiment Balance (-100 to +100) -->
        <div class="lg:col-span-4 flex flex-col justify-center space-y-3 border-b lg:border-b-0 lg:border-r border-[#EAE3D5] pb-5 lg:pb-0 lg:px-4">
          <div class="flex items-center justify-between">
            <span class="text-xs font-semibold uppercase tracking-wider text-[#655D52]">
              Net Fundamental Balance
            </span>
            <span class="text-xl font-mono font-bold
              {% if analysis.weighted_sentiment_score > 10 %}text-[#1B6B45]
              {% elif analysis.weighted_sentiment_score < -10 %}text-[#A82828]
              {% else %}text-[#9A6B16]{% endif %}">
              {{ "%+.1f"|format(analysis.weighted_sentiment_score) }} <span class="text-xs font-normal text-[#786E60]">/ 100</span>
            </span>
          </div>

          <div class="relative h-3.5 w-full rounded-full bg-[#EFE9DD] border border-[#D5CBB8] overflow-hidden">
            <div class="absolute inset-0 flex">
              <div class="w-1/2 bg-gradient-to-r from-[#A82828]/25 via-[#A82828]/10 to-transparent border-r border-[#B5A892]"></div>
              <div class="w-1/2 bg-gradient-to-r from-transparent via-[#1B6B45]/10 to-[#1B6B45]/25"></div>
            </div>
            <div class="absolute top-0 bottom-0 w-2.5 -ml-1.5 rounded-full bg-[#1F1C18] border-2 border-[#FFFDF9] shadow"
                 style="left: {{ vm.sentiment_gauge_pct }}%;"></div>
          </div>

          <div class="flex justify-between text-[11px] font-medium text-[#655D52]">
            <span class="text-[#A82828]">&larr; Bearish (Strong USD / Yields)</span>
            <span>Neutral (0)</span>
            <span class="text-[#1B6B45]">Bullish (Safe-Haven / Rate Cuts) &rarr;</span>
          </div>
        </div>

        <!-- Right: Desk Conviction Level (0% - 100%) -->
        <div class="lg:col-span-4 flex flex-col justify-center space-y-3 lg:pl-2">
          <div class="flex items-center justify-between">
            <span class="text-xs font-semibold uppercase tracking-wider text-[#655D52]">
              Desk Conviction Level
            </span>
            <span class="text-xl font-mono font-bold text-[#1F1C18]">
              {{ analysis.confidence_score }}%
            </span>
          </div>

          <div class="w-full h-3.5 bg-[#EFE9DD] rounded-full border border-[#D5CBB8] overflow-hidden p-0.5">
            <div class="h-full rounded-full {{ vm.bias_meta.bar_class }} transition-all duration-500"
                 style="width: {{ analysis.confidence_score }}%;"></div>
          </div>

          <div class="flex justify-between text-[11px] font-medium text-[#655D52]">
            <span>Low (&lt;40%)</span>
            <span>Moderate (40–75%)</span>
            <span class="text-[#1F1C18] font-semibold">High Alignment (80%+)</span>
          </div>
        </div>

      </div>
    </section>

    <!-- =================================================================== -->
    <!-- 2. LIVE MARKET QUOTES & CORRELATION STRIP                           -->
    <!-- =================================================================== -->
    <section aria-label="Live Market Quotes">
      <div class="flex flex-wrap items-center justify-between mb-2.5 px-1 gap-2">
        <h2 class="editorial-heading text-lg font-bold text-[#1F1C18]">
          Live Cross-Asset Quotes &amp; Gold Relationship
        </h2>
        <span class="text-xs text-[#655D52]">
          Note: Rising <strong class="text-[#1F1C18]">DXY</strong> and <strong class="text-[#1F1C18]">10Y Yields (^TNX)</strong> typically pressure Gold; rising <strong class="text-[#1F1C18]">VIX</strong> and <strong class="text-[#1F1C18]">GLD</strong> support Gold.
        </span>
      </div>

      <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3.5">
        {% for key, item in payload.gold_macro_snapshot.tickers.items() %}
        {% set is_primary = (key == 'GOLD_FUTURES') %}
        {% set price_up = (item.change is not none and item.change >= 0) %}
        {% set gold_eff = item.gold_directional_effect %}
        <div class="paper-card rounded-xl p-4 transition duration-150 {{ 'border-2 border-[#9A6B16] bg-[#FFFDF7]' if is_primary else '' }}">
          <div class="flex items-start justify-between gap-2">
            <div>
              <div class="text-xs font-mono font-bold uppercase {{ 'text-[#9A6B16]' if is_primary else 'text-[#655D52]' }}">
                {{ item.display_symbol }}
              </div>
              <div class="text-xs text-[#4A433B] font-medium truncate max-w-[180px]" title="{{ item.label }}">
                {{ item.label }}
              </div>
            </div>
            <span class="px-2 py-0.5 rounded text-[10px] font-semibold border
              {% if gold_eff == 'BULLISH GOLD' %}
                bg-[#E6F2EC] text-[#1B6B45] border-[#B8D9C8]
              {% elif gold_eff == 'BEARISH GOLD' %}
                bg-[#FAEDED] text-[#A82828] border-[#E8C3C3]
              {% else %}
                bg-[#FBF3E0] text-[#8A5A0E] border-[#E6CFA0]
              {% endif %}">
              {{ gold_eff }}
            </span>
          </div>

          {% if item.price is not none %}
          <div class="mt-3 flex items-baseline justify-between">
            <div class="text-2xl font-mono font-semibold tracking-tight text-[#1F1C18]">
              {% if item.unit == 'USD/oz' or item.unit == 'USD' %}${% endif %}{{ "{:,.2f}".format(item.price) if item.symbol != '^TNX' else "{:.3f}".format(item.price) }}<span class="text-xs font-normal text-[#786E60] ml-1">{{ item.unit if item.unit not in ('USD', 'USD/oz') else ('/oz' if item.unit == 'USD/oz' else '') }}</span>
            </div>
            <div class="flex items-center gap-0.5 font-mono text-xs font-semibold
              {% if is_primary or key == 'GLD_ETF' %}
                {{ 'text-[#1B6B45]' if price_up else 'text-[#A82828]' }}
              {% elif key in ('DXY_DOLLAR_INDEX', 'US_10Y_YIELD') %}
                {{ 'text-[#A82828]' if price_up else 'text-[#1B6B45]' }}
              {% else %}
                {{ 'text-[#9A6B16]' if price_up else 'text-[#655D52]' }}
              {% endif %}">
              <span>{{ '▲' if price_up else '▼' }}</span>
              <span>{{ "%+.2f"|format(item.change_pct) }}%</span>
            </div>
          </div>

          <div class="mt-2.5 pt-2 border-t border-[#EAE3D5] flex items-center justify-between text-[11px] font-mono text-[#655D52]">
            <span>Low: <strong class="text-[#1F1C18]">{{ "{:,.2f}".format(item.day_low) if item.symbol != '^TNX' else "{:.3f}".format(item.day_low) }}</strong></span>
            <span>High: <strong class="text-[#1F1C18]">{{ "{:,.2f}".format(item.day_high) if item.symbol != '^TNX' else "{:.3f}".format(item.day_high) }}</strong></span>
            <span>5d: <strong class="{{ 'text-[#1B6B45]' if item.five_day_change_pct >= 0 else 'text-[#A82828]' }}">{{ "%+.2f"|format(item.five_day_change_pct) }}%</strong></span>
          </div>
          <p class="mt-2 text-xs text-[#5C5346] leading-snug line-clamp-2" title="{{ item.regime_signal }}">
            {{ item.regime_signal }}
          </p>
          {% else %}
          <div class="mt-4 text-sm font-medium text-[#9A6B16]">Quote Unavailable</div>
          {% endif %}
        </div>
        {% endfor %}
      </div>
    </section>

    <!-- =================================================================== -->
    <!-- 3. DESK TRADING THESIS & MACRO DRIVER BREAKDOWN                     -->
    <!-- =================================================================== -->
    <section class="grid grid-cols-1 lg:grid-cols-12 gap-6">

      <!-- Left (7 cols): Why Buy or Sell Gold Today -->
      <div class="lg:col-span-7 paper-card rounded-xl p-6 flex flex-col justify-between">
        <div>
          <div class="flex items-center justify-between border-b border-[#EAE3D5] pb-3.5 mb-4">
            <div>
              <span class="text-[11px] font-semibold uppercase tracking-wider text-[#9A6B16]">Desk Rationale</span>
              <h2 class="editorial-heading text-xl font-bold text-[#1F1C18]">
                Why Buy or Sell Gold Today — Key Takeaways
              </h2>
            </div>
          </div>

          <ol class="space-y-3 text-sm text-[#2C2621] leading-relaxed">
            {% for bullet in vm.thesis_bullets %}
            <li class="flex items-start gap-3 paper-inset rounded-lg p-3.5">
              <span class="mt-0.5 w-5 h-5 rounded-full bg-[#1F1C18] text-[#FAF6EE] font-mono text-xs font-semibold flex items-center justify-center shrink-0">
                {{ loop.index }}
              </span>
              <span>{{ bullet }}</span>
            </li>
            {% endfor %}
          </ol>
        </div>

        {% if vm.invalidation_bullet %}
        <div class="mt-4 bg-[#FAEDED] border border-[#E8C3C3] rounded-lg p-4 flex items-start gap-3">
          <div class="px-2.5 py-1 rounded bg-[#A82828] text-white font-semibold text-xs uppercase shrink-0 mt-0.5">
            Key Risk / Invalidation
          </div>
          <p class="text-xs sm:text-sm text-[#6E1A1A] leading-relaxed">
            {{ vm.invalidation_bullet }}
          </p>
        </div>
        {% endif %}
      </div>

      <!-- Right (5 cols): Macroeconomic Condition & 4 Core Pillars -->
      <div class="lg:col-span-5 paper-card rounded-xl p-6 flex flex-col justify-between space-y-4">
        <div>
          <div class="border-b border-[#EAE3D5] pb-3.5 mb-4">
            <span class="text-[11px] font-semibold uppercase tracking-wider text-[#1B6B45]">Macro Overview</span>
            <h2 class="editorial-heading text-xl font-bold text-[#1F1C18]">
              Current Market Environment
            </h2>
          </div>

          <div class="paper-inset rounded-lg p-4 text-sm text-[#2C2621] leading-relaxed">
            {{ analysis.macro_condition_summary }}
          </div>
        </div>

        <!-- 4 Gold Drivers -->
        <div class="space-y-2.5">
          <div class="text-xs font-semibold uppercase tracking-wider text-[#655D52]">
            Four Core Gold Drivers (Weighted Importance)
          </div>
          {% for w in analysis.wedge_breakdown %}
          <div class="bg-[#FAF6EE] border border-[#E2D9C8] rounded-lg px-3.5 py-2.5">
            <div class="flex items-center justify-between text-xs">
              <span class="font-semibold text-[#1F1C18] truncate pr-2">{{ w.wedge_name }}</span>
              <div class="flex items-center gap-2 shrink-0 font-mono">
                <span class="px-1.5 py-0.5 rounded bg-[#EAE3D5] text-[#4A433B] text-[11px]">{{ w.weight_percentage }}% Wt</span>
                <span class="font-semibold px-2 py-0.5 rounded text-[11px]
                  {% if w.net_sentiment_score > 5 %}bg-[#E6F2EC] text-[#1B6B45] border border-[#B8D9C8]
                  {% elif w.net_sentiment_score < -5 %}bg-[#FAEDED] text-[#A82828] border border-[#E8C3C3]
                  {% else %}bg-[#FBF3E0] text-[#8A5A0E] border border-[#E6CFA0]{% endif %}">
                  {{ "%+.1f"|format(w.net_sentiment_score) }}
                </span>
              </div>
            </div>
            <p class="text-xs text-[#5C5346] mt-1 leading-snug">{{ w.summary }}</p>
          </div>
          {% endfor %}
        </div>
      </div>

    </section>

    <!-- =================================================================== -->
    <!-- 4. VISUAL SUMMARY CHARTS                                            -->
    <!-- =================================================================== -->
    <section class="grid grid-cols-1 lg:grid-cols-12 gap-6">

      <!-- Chart 1: Factor Scores -->
      <div class="lg:col-span-7 paper-card rounded-xl p-6">
        <div class="mb-4">
          <h2 class="editorial-heading text-lg font-bold text-[#1F1C18]">
            Gold Driver Scores (Bearish -100 to Bullish +100)
          </h2>
          <p class="text-xs text-[#655D52] mt-0.5">
            How each macro factor is currently pushing XAUUSD / GC=F
          </p>
        </div>
        <div class="relative h-[280px] w-full">
          <canvas id="wedgeBarChart"></canvas>
        </div>
      </div>

      <!-- Chart 2: News Balance -->
      <div class="lg:col-span-5 paper-card rounded-xl p-6">
        <div class="mb-4">
          <h2 class="editorial-heading text-lg font-bold text-[#1F1C18]">
            Today&rsquo;s News Sentiment Balance
          </h2>
          <p class="text-xs text-[#655D52] mt-0.5">
            Breakdown of {{ vm.sorted_news_items | length }} tracked stories weighted by market importance
          </p>
        </div>
        <div class="relative h-[230px] w-full flex items-center justify-center">
          <canvas id="newsDoughnutChart"></canvas>
        </div>
        <div class="grid grid-cols-3 gap-2 mt-4 pt-3 border-t border-[#EAE3D5] text-center text-xs">
          <div class="bg-[#E6F2EC] border border-[#B8D9C8] rounded-lg py-1.5">
            <div class="text-[#1B6B45] font-mono font-bold">{{ vm.doughnut_pcts['BULLISH GOLD'] }}%</div>
            <div class="text-[11px] text-[#4A433B]">Bullish ({{ vm.count_dist['BULLISH GOLD'] }})</div>
          </div>
          <div class="bg-[#FAEDED] border border-[#E8C3C3] rounded-lg py-1.5">
            <div class="text-[#A82828] font-mono font-bold">{{ vm.doughnut_pcts['BEARISH GOLD'] }}%</div>
            <div class="text-[11px] text-[#4A433B]">Bearish ({{ vm.count_dist['BEARISH GOLD'] }})</div>
          </div>
          <div class="bg-[#FBF3E0] border border-[#E6CFA0] rounded-lg py-1.5">
            <div class="text-[#8A5A0E] font-mono font-bold">{{ vm.doughnut_pcts['NEUTRAL'] }}%</div>
            <div class="text-[11px] text-[#4A433B]">Neutral ({{ vm.count_dist['NEUTRAL'] }})</div>
          </div>
        </div>
      </div>

    </section>

    <!-- =================================================================== -->
    <!-- 5. CURATED GOLD & MACRO NEWS BRIEFING (SORTED BY IMPORTANCE)        -->
    <!-- =================================================================== -->
    <section class="paper-card rounded-xl overflow-hidden">
      <div class="px-6 py-4 border-b border-[#DED6C6] bg-[#FAF6EE] flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
        <div>
          <h2 class="editorial-heading text-xl font-bold text-[#1F1C18]">
            Intraday Gold &amp; Macro News Briefing
          </h2>
          <p class="text-xs text-[#655D52] mt-0.5">
            Ranked from highest market priority (10/10) to lowest, with desk notes on how each story affects Gold
          </p>
        </div>
        <div class="text-xs font-medium text-[#655D52]">
          Showing <strong class="text-[#1F1C18]">{{ vm.sorted_news_items | length }}</strong> Stories
        </div>
      </div>

      <div class="overflow-x-auto">
        <table class="w-full text-left border-collapse">
          <thead>
            <tr class="bg-[#F0EAE0] border-b border-[#DED6C6] text-[11px] font-semibold uppercase tracking-wider text-[#5C5346]">
              <th class="py-3 px-3 text-center w-20">Priority</th>
              <th class="py-3 px-4 w-44">Date &amp; Time (BST)</th>
              <th class="py-3 px-4 w-48">Driver Category</th>
              <th class="py-3 px-4">Headline &amp; Source</th>
              <th class="py-3 px-4 text-center w-36">Gold Impact</th>
              <th class="py-3 px-5 w-[30%]">Desk Note (Why It Matters)</th>
            </tr>
          </thead>
          <tbody class="divide-y divide-[#EAE3D5] text-sm">
            {% for item in vm.sorted_news_items %}
            {% set wt = item.impact_weight | int %}
            {% set bias = item.gold_bias %}
            <tr class="hover:bg-[#FAF6EE] transition-colors">
              <td class="py-3.5 px-3 text-center align-top">
                <div class="inline-flex flex-col items-center justify-center px-2.5 py-1 rounded-md font-mono font-semibold border
                  {% if wt >= 8 %}
                    bg-[#FBF3E0] text-[#8A5A0E] border-[#E6CFA0]
                  {% elif wt >= 5 %}
                    bg-[#F0EAE0] text-[#1F1C18] border-[#D5CBB8]
                  {% else %}
                    bg-[#FAF6EE] text-[#655D52] border-[#E2D9C8]
                  {% endif %}">
                  <span class="text-sm leading-none">{{ wt }}<span class="text-[10px] font-normal">/10</span></span>
                </div>
              </td>

              <td class="py-3.5 px-4 align-top font-mono text-xs">
                <div class="font-semibold text-[#1F1C18]">{{ item.bst_date }}</div>
                <div class="text-[#9A6B16] font-medium mt-0.5">{{ item.bst_time }}</div>
                {% if item.age_human %}
                <span class="inline-block mt-1 px-1.5 py-0.5 rounded bg-[#F0EAE0] text-[#5C5346] font-sans text-[10px]">
                  {{ item.age_human }}
                </span>
                {% endif %}
              </td>

              <td class="py-3.5 px-4 align-top">
                <div class="text-xs font-semibold text-[#5C5346] leading-snug">
                  {{ item.gold_wedge_category }}
                </div>
              </td>

              <td class="py-3.5 px-4 align-top">
                {% if item.url %}
                <a href="{{ item.url }}" target="_blank" rel="noopener noreferrer"
                   class="font-semibold text-[#1F1C18] hover:text-[#9A6B16] underline decoration-[#DED6C6] hover:decoration-[#9A6B16] transition-colors leading-snug block">
                  {{ item.headline }}
                </a>
                {% else %}
                <div class="font-semibold text-[#1F1C18] leading-snug">
                  {{ item.headline }}
                </div>
                {% endif %}
                <div class="flex flex-wrap items-center gap-2 text-xs text-[#655D52] mt-1.5">
                  <span class="px-2 py-0.5 rounded bg-[#F0EAE0] border border-[#DED6C6] text-[#1F1C18] font-medium text-[11px]">
                    {{ item.source_name }}
                  </span>
                  <span class="font-mono text-[11px] text-[#786E60]">
                    {{ item.bst_datetime }}
                  </span>
                </div>
              </td>

              <td class="py-3.5 px-4 text-center align-top">
                <span class="inline-block px-2.5 py-1 rounded text-xs font-semibold border
                  {% if bias == 'BULLISH GOLD' %}
                    bg-[#E6F2EC] text-[#1B6B45] border-[#B8D9C8]
                  {% elif bias == 'BEARISH GOLD' %}
                    bg-[#FAEDED] text-[#A82828] border-[#E8C3C3]
                  {% else %}
                    bg-[#FBF3E0] text-[#8A5A0E] border-[#E6CFA0]
                  {% endif %}">
                  {{ bias }}
                </span>
              </td>

              <td class="py-3.5 px-5 align-top text-xs sm:text-sm text-[#36312A] leading-relaxed">
                {{ item.gold_transmission_mechanism }}
              </td>
            </tr>
            {% endfor %}
          </tbody>
        </table>
      </div>
    </section>

    <!-- =================================================================== -->
    <!-- 6. GOLD-ONLY ECONOMIC CALENDAR & RELEASE PLAYBOOK (USD -> XAUUSD)   -->
    <!-- =================================================================== -->
    {% if vm.gold_calendar and vm.gold_calendar.events %}
    <section aria-label="Gold Economic Calendar" class="paper-card rounded-xl overflow-hidden border-t-4 border-t-[#9A6B16]">
      <div class="px-6 py-5 border-b border-[#DED6C6] bg-[#FAF6EE]">
        <div class="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
          <div>
            <div class="text-xs font-semibold uppercase tracking-wider text-[#9A6B16] mb-0.5">
              US Economic Releases Driving Gold (USD &rarr; XAUUSD / GC=F)
            </div>
            <h2 class="editorial-heading text-xl sm:text-2xl font-bold text-[#1F1C18]">
              Gold Economic Calendar &amp; Event Reaction Playbook
            </h2>
            <p class="text-xs text-[#655D52] mt-0.5">
              Filtered strictly for US &amp; Federal Reserve releases that move Gold via Treasury yields and the Dollar
            </p>
          </div>

          <div class="flex items-center gap-3 text-xs shrink-0">
            <div class="paper-card rounded-lg px-3.5 py-2 text-center">
              <div class="text-[10px] uppercase font-semibold text-[#786E60]">High-Impact Releases</div>
              <div class="text-[#A82828] font-mono font-bold text-sm mt-0.5">{{ vm.gold_calendar.high_impact_count }} Major</div>
            </div>
            <div class="paper-card rounded-lg px-3.5 py-2 text-center">
              <div class="text-[10px] uppercase font-semibold text-[#786E60]">Tracked Releases</div>
              <div class="text-[#1F1C18] font-mono font-bold text-sm mt-0.5">{{ vm.gold_calendar.total_gold_events }} Events</div>
            </div>
          </div>
        </div>

        <!-- 2 Summary Callout Strips -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-3.5 mt-4 pt-4 border-t border-[#EAE3D5] text-xs">
          <div class="paper-card rounded-lg px-4 py-3 flex items-start gap-3">
            <span class="px-2 py-0.5 rounded bg-[#FBF3E0] text-[#8A5A0E] border border-[#E6CFA0] font-semibold uppercase text-[10px] shrink-0 mt-0.5">
              Next Major Catalyst
            </span>
            <span class="text-[#1F1C18] font-medium leading-snug">
              {{ vm.gold_calendar.next_major_gold_catalyst }}
            </span>
          </div>
          <div class="paper-card rounded-lg px-4 py-3 flex items-start gap-3">
            <span class="px-2 py-0.5 rounded bg-[#F0EAE0] text-[#4A433B] border border-[#D5CBB8] font-semibold uppercase text-[10px] shrink-0 mt-0.5">
              Calendar Macro Tilt
            </span>
            <span class="text-[#1F1C18] font-medium leading-snug">
              {{ vm.gold_calendar.calendar_macro_tilt }}
            </span>
          </div>
        </div>
      </div>

      <div class="overflow-x-auto">
        <table class="w-full text-left border-collapse">
          <thead>
            <tr class="bg-[#F0EAE0] border-b border-[#DED6C6] text-[11px] font-semibold uppercase tracking-wider text-[#5C5346]">
              <th class="py-3 px-4 w-44">Date &amp; Time (BST)</th>
              <th class="py-3 px-4 w-60">Economic Release</th>
              <th class="py-3 px-3 text-center w-32">Impact</th>
              <th class="py-3 px-4 text-center w-52">Actual / Forecast / Prev</th>
              <th class="py-3 px-4 w-[29%]">What Happens to Gold &amp; Typical Move (1m / 30m / 1h)</th>
              <th class="py-3 px-4 w-[24%]">Desk Bias &amp; Note</th>
            </tr>
          </thead>
          <tbody class="divide-y divide-[#EAE3D5] text-xs sm:text-sm">
            {% for ev in vm.gold_calendar.events %}
            <tr class="hover:bg-[#FAF6EE] transition-colors {{ 'bg-[#FFFDF4]' if ev.impact_level == 'High' else '' }}">
              <td class="py-3.5 px-4 align-top font-mono">
                <div class="text-[#1F1C18] font-semibold text-xs">{{ ev.date_utc }}</div>
                <span class="inline-block mt-1 px-2 py-0.5 rounded text-[10px] font-sans font-semibold border
                  {% if 'IMMINENT' in ev.release_status %}
                    bg-[#FBF3E0] text-[#8A5A0E] border-[#E6CFA0]
                  {% elif 'RELEASED TODAY' in ev.release_status %}
                    bg-[#E6F2EC] text-[#1B6B45] border-[#B8D9C8]
                  {% elif 'UPCOMING' in ev.release_status %}
                    bg-[#EDF3F8] text-[#2B547E] border-[#C2D6E8]
                  {% else %}
                    bg-[#F0EAE0] text-[#655D52] border-[#DED6C6]
                  {% endif %}">
                  {{ ev.release_status }}
                </span>
              </td>

              <td class="py-3.5 px-4 align-top">
                <div class="flex items-center gap-1.5 mb-1">
                  <span class="px-1.5 py-0.5 rounded bg-[#F0EAE0] border border-[#D5CBB8] text-[#4A433B] font-mono text-[10px] font-semibold">
                    {{ ev.currency }}
                  </span>
                </div>
                <a href="{{ ev.url }}" target="_blank" rel="noopener noreferrer"
                   class="font-bold text-[#1F1C18] hover:text-[#9A6B16] underline decoration-[#DED6C6] transition-colors leading-snug block">
                  {{ ev.event_name }}
                </a>
                <div class="text-[11px] text-[#786E60] mt-0.5">
                  {{ ev.gold_category }}
                </div>
              </td>

              <td class="py-3.5 px-3 text-center align-top">
                <span class="inline-block px-2.5 py-1 rounded text-[11px] font-semibold border
                  {% if ev.impact_level == 'High' %}
                    bg-[#FAEDED] text-[#A82828] border-[#E8C3C3]
                  {% elif ev.impact_level == 'Medium' %}
                    bg-[#FBF3E0] text-[#8A5A0E] border-[#E6CFA0]
                  {% else %}
                    bg-[#F0EAE0] text-[#655D52] border-[#DED6C6]
                  {% endif %}">
                  {{ ev.strength }}
                </span>
                <div class="text-[11px] text-[#786E60] mt-1">
                  {{ ev.quality }}
                </div>
              </td>

              <td class="py-3.5 px-4 align-top font-mono">
                <div class="grid grid-cols-3 gap-1.5 text-center paper-inset rounded-lg p-2 text-xs">
                  <div>
                    <div class="text-[9px] font-sans uppercase text-[#786E60]">Actual</div>
                    <div class="font-bold text-[#1F1C18] mt-0.5">{{ ev.actual }}</div>
                  </div>
                  <div class="border-x border-[#DED6C6] px-1">
                    <div class="text-[9px] font-sans uppercase text-[#786E60]">Forecast</div>
                    <div class="font-semibold text-[#4A433B] mt-0.5">{{ ev.forecast }}</div>
                  </div>
                  <div>
                    <div class="text-[9px] font-sans uppercase text-[#786E60]">Prev</div>
                    <div class="text-[#786E60] mt-0.5">{{ ev.previous }}</div>
                  </div>
                </div>
                <div class="text-[11px] font-sans text-center text-[#655D52] mt-1 truncate" title="{{ ev.outcome_vs_projection }}">
                  {{ ev.outcome_vs_projection }}
                </div>
              </td>

              <td class="py-3.5 px-4 align-top text-xs space-y-1.5">
                <div class="paper-inset rounded-lg p-2.5 space-y-1">
                  <div class="text-[#2C2621] leading-snug">
                    <strong class="text-[#A82828]">If Higher / Stronger:</strong> {{ ev.smart_rule_hot }}
                  </div>
                  <div class="text-[#2C2621] leading-snug">
                    <strong class="text-[#1B6B45]">If Lower / Softer:</strong> {{ ev.smart_rule_cool }}
                  </div>
                </div>
                <div class="flex flex-wrap items-center gap-1.5 text-[11px] font-mono text-[#5C5346] pt-0.5">
                  <span class="px-2 py-0.5 rounded bg-[#FAF6EE] border border-[#DED6C6]">
                    <strong>1m:</strong> {{ ev.ml_1m }}
                  </span>
                  <span class="px-2 py-0.5 rounded bg-[#FAF6EE] border border-[#DED6C6]">
                    <strong>30m:</strong> {{ ev.ml_30m }}
                  </span>
                  <span class="px-2 py-0.5 rounded bg-[#FAF6EE] border border-[#DED6C6]">
                    <strong>1h:</strong> {{ ev.ml_1h }}
                  </span>
                </div>
              </td>

              <td class="py-3.5 px-4 align-top">
                <span class="inline-block px-2.5 py-1 rounded text-xs font-semibold border mb-1.5
                  {% if ev.smart_gold_bias == 'BULLISH GOLD' %}
                    bg-[#E6F2EC] text-[#1B6B45] border-[#B8D9C8]
                  {% elif ev.smart_gold_bias == 'BEARISH GOLD' %}
                    bg-[#FAEDED] text-[#A82828] border-[#E8C3C3]
                  {% else %}
                    bg-[#FBF3E0] text-[#8A5A0E] border-[#E6CFA0]
                  {% endif %}">
                  {{ ev.smart_gold_bias }}
                </span>
                <p class="text-xs text-[#36312A] leading-relaxed">
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

    <footer class="text-center text-xs text-[#786E60] py-5 border-t border-[#DED6C6]">
      XAUUSD / COMEX Gold (GC=F) Desk Research &amp; Fundamental Briefing
    </footer>

  </div>

  <!-- ===================================================================== -->
  <!-- PORTAL & CHART CONTROLLER                                             -->
  <!-- ===================================================================== -->
  <script>
    (function() {
      const wedgeData = {{ vm.chart_wedge_json | safe }};
      const doughnutData = {{ vm.chart_doughnut_json | safe }};
      const initialPortalAuth = {{ vm.portal_auth_json | safe }};

      const STORAGE_USERS_KEY = 'xauusd_portal_code_users_v2';
      const STORAGE_SESSION_KEY = 'xauusd_portal_active_session_v2';
      try {
        localStorage.removeItem('xauusd_portal_code_users_v1');
        localStorage.removeItem('xauusd_portal_active_session_v1');
        sessionStorage.removeItem('xauusd_portal_active_session_v1');
      } catch (e) {}

      let portalAdmin = initialPortalAuth.admin;
      let portalUsers = Array.isArray(initialPortalAuth.users) ? initialPortalAuth.users.slice() : [];

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
        banner.className = 'rounded-md p-3 text-xs font-medium border ' + (
          isError
            ? 'bg-[#FAEDED] border-[#E8C3C3] text-[#A82828]'
            : 'bg-[#E6F2EC] border-[#B8D9C8] text-[#1B6B45]'
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
        html += '<tr class="bg-[#FBF3E0]/60">' +
          '<td class="py-2.5 px-3.5 font-bold text-[#1F1C18]">' + portalAdmin.username + '</td>' +
          '<td class="py-2.5 px-3.5 font-mono text-[#5C5346]">' + portalAdmin.password + '</td>' +
          '<td class="py-2.5 px-3.5 text-[#1F1C18]">' + (portalAdmin.display_name || 'Administrator') + '</td>' +
          '<td class="py-2.5 px-3.5"><span class="px-2 py-0.5 rounded bg-[#FBF3E0] text-[#8A5A0E] border border-[#E6CFA0] text-[10px] font-semibold">ADMIN</span></td>' +
          '<td class="py-2.5 px-3.5 text-center"><span class="px-2 py-0.5 rounded bg-[#E6F2EC] text-[#1B6B45] text-[10px] font-semibold">ACTIVE</span></td>' +
          '<td class="py-2.5 px-3.5 text-right text-[#786E60] text-[11px]">Owner</td>' +
        '</tr>';

        if (portalUsers.length === 0) {
          html += '<tr><td colspan="6" class="py-4 px-3.5 text-center text-[#786E60] italic">No members registered yet. Use the form above to register a member manually.</td></tr>';
        }

        portalUsers.forEach(function(u) {
          const isActive = u.active !== false;
          const safeUser = String(u.username || '').replace(/'/g, "\\'");
          html += '<tr class="hover:bg-[#FAF6EE]">' +
            '<td class="py-2.5 px-3.5 font-semibold text-[#1F1C18]">' + u.username + '</td>' +
            '<td class="py-2.5 px-3.5 font-mono text-[#9A6B16]">' + u.password + '</td>' +
            '<td class="py-2.5 px-3.5 text-[#4A433B]">' + (u.display_name || u.username) + '</td>' +
            '<td class="py-2.5 px-3.5"><span class="px-2 py-0.5 rounded bg-[#F0EAE0] text-[#4A433B] border border-[#DED6C6] text-[10px]">MEMBER</span></td>' +
            '<td class="py-2.5 px-3.5 text-center">' +
              (isActive
                ? '<span class="px-2 py-0.5 rounded bg-[#E6F2EC] text-[#1B6B45] border border-[#B8D9C8] text-[10px] font-semibold">ACTIVE</span>'
                : '<span class="px-2 py-0.5 rounded bg-[#FAEDED] text-[#A82828] border border-[#E8C3C3] text-[10px] font-semibold">PAUSED</span>') +
            '</td>' +
            '<td class="py-2.5 px-3.5 text-right space-x-1.5">' +
              '<button type="button" onclick="window.adminToggleUser(\'' + safeUser + '\')" class="px-2.5 py-1 rounded bg-[#F0EAE0] hover:bg-[#E2D9C8] text-[#1F1C18] border border-[#D5CBB8] text-[11px]">' +
                (isActive ? 'Pause' : 'Activate') +
              '</button>' +
              '<button type="button" onclick="window.adminDeleteUser(\'' + safeUser + '\')" class="px-2.5 py-1 rounded bg-[#FAEDED] hover:bg-[#F5D6D6] text-[#A82828] border border-[#E8C3C3] text-[11px]">' +
                'Remove' +
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
            userBadge.innerHTML = 'Signed in as <strong>' + session.username + '</strong> (Admin)';
            userBadge.className = 'px-2.5 py-1 rounded text-xs font-medium bg-[#FBF3E0] text-[#8A5A0E] border border-[#E6CFA0]';
          }
          if (adminBtn) adminBtn.classList.remove('hidden');
        } else {
          if (userBadge) {
            userBadge.innerHTML = 'Signed in as <strong>' + (session.display_name || session.username) + '</strong>';
            userBadge.className = 'px-2.5 py-1 rounded text-xs font-medium bg-[#E6F2EC] text-[#1B6B45] border border-[#B8D9C8]';
          }
          if (adminBtn) adminBtn.classList.add('hidden');
        }
      }

      window.handlePortalLogin = function(ev) {
        if (ev) ev.preventDefault();
        const u = (document.getElementById('loginUsername').value || '').trim();
        const p = (document.getElementById('loginPassword').value || '').trim();
        const errBox = document.getElementById('loginErrorBox');
        if (errBox) errBox.classList.add('hidden');

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
              errBox.textContent = 'This account is currently paused by the Administrator.';
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
          errBox.textContent = 'Invalid username or password.';
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
          showAdminBanner("Cannot overwrite the Administrator username ('" + portalAdmin.username + "').", true);
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
                showAdminBanner("Member '" + uname + "' registered successfully.", false);
              } else {
                showAdminBanner((res.body && res.body.error) || 'Failed to register member.', true);
              }
            })
            .catch(function(err) {
              showAdminBanner('Error: ' + err, true);
            });
          return false;
        }

        const nowStr = new Date(Date.now() + 6 * 3600 * 1000).toISOString().slice(0, 16).replace('T', ' ') + ' BST';
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
        showAdminBanner("Member '" + uname + "' registered.", false);
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
                showAdminBanner("Updated access status for '" + username + "'.", false);
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
                showAdminBanner("Removed member '" + username + "'.", false);
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
          .then(function() {
            window.location.reload();
          })
          .catch(function() {
            window.location.reload();
          });
      };

      updatePortalUI();
      syncUsersFromServerIfAvailable();

      // Warm Editorial Chart.js Theme
      Chart.defaults.color = '#5C5346';
      Chart.defaults.font.family = "'Plus Jakarta Sans', -apple-system, sans-serif";

      const barCtx = document.getElementById('wedgeBarChart');
      if (barCtx) {
        new Chart(barCtx, {
          type: 'bar',
          data: {
            labels: wedgeData.labels,
            datasets: [{
              label: 'Net Gold Driver Score (-100 to +100)',
              data: wedgeData.scores,
              backgroundColor: wedgeData.bgColors,
              borderColor: wedgeData.borderColors,
              borderWidth: 1.5,
              borderRadius: 5,
              barThickness: 26
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
                  color: 'rgba(213, 203, 184, 0.55)',
                  zeroLineColor: '#9A6B16'
                },
                ticks: {
                  color: '#5C5346',
                  callback: function(value) { return (value > 0 ? '+' : '') + value; }
                }
              },
              y: {
                grid: { display: false },
                ticks: { color: '#1F1C18', font: { size: 11, weight: '600' } }
              }
            },
            plugins: {
              legend: { display: false },
              tooltip: {
                backgroundColor: '#1F1C18',
                titleColor: '#FAF6EE',
                bodyColor: '#FAF6EE',
                borderColor: '#9A6B16',
                borderWidth: 1,
                callbacks: {
                  label: function(context) {
                    const v = context.parsed.x;
                    return ' Score: ' + (v > 0 ? '+' : '') + v.toFixed(1) + ' / 100';
                  }
                }
              }
            }
          }
        });
      }

      const doughnutCtx = document.getElementById('newsDoughnutChart');
      if (doughnutCtx) {
        new Chart(doughnutCtx, {
          type: 'doughnut',
          data: {
            labels: doughnutData.labels,
            datasets: [{
              data: doughnutData.weights,
              backgroundColor: [
                'rgba(27, 107, 69, 0.85)',
                'rgba(168, 40, 40, 0.85)',
                'rgba(180, 123, 28, 0.85)'
              ],
              borderColor: [
                '#FFFDF9',
                '#FFFDF9',
                '#FFFDF9'
              ],
              borderWidth: 2,
              hoverOffset: 5
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '62%',
            plugins: {
              legend: {
                position: 'bottom',
                labels: {
                  color: '#1F1C18',
                  padding: 12,
                  font: { size: 11 }
                }
              },
              tooltip: {
                backgroundColor: '#1F1C18',
                titleColor: '#FAF6EE',
                bodyColor: '#FAF6EE',
                borderColor: '#9A6B16',
                borderWidth: 1,
                callbacks: {
                  label: function(context) {
                    return ' Total Priority Weight: ' + context.parsed;
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
    Render the Gold (`XAUUSD` / `GC=F`) desk research HTML report
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
