"""
CLI Entry Point for the Gold (`XAUUSD` / COMEX `GC=F`) Intraday News & Macro
Sentiment Trading Agent (`main.py`).

Wires together:
  `data_fetcher.py` -> `analyzer.py` -> `report_generator.py` (`xauusd_intraday_report.html`)
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
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
import warnings
import webbrowser
from typing import Any, Dict, List, Optional

warnings.filterwarnings("ignore", message=".*inplace keyword.*")
warnings.filterwarnings("ignore", module="yfinance.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

from analyzer import (
    GoldAnalysisReport,
    analyze_gold_market,
    build_deterministic_gold_analysis,
)
from config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GOLD_IMPACT_WEDGES,
    PRIMARY_ASSET,
    TARGET_TICKERS,
)
from data_fetcher import fetch_market_intelligence_payload
from report_generator import DEFAULT_REPORT_FILENAME, generate_html_report


def _build_demo_gold_payload_and_analysis() -> tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Build a realistic institutional Gold (`XAUUSD` / `GC=F`), `DXY`, `^TNX`, `^VIX`, and `GLD`
    dataset (hydrating live quotes from `data_fetcher.py` when online, with complete sample
    fallback if offline) and a full `GoldAnalysisReport` for `--demo` UI inspection.
    """
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        live_payload = fetch_market_intelligence_payload(target_tickers=TARGET_TICKERS)
        if (
            live_payload.get("intraday_breaking_news_count", 0) >= 10
            and (live_payload.get("gold_macro_snapshot") or {}).get("tickers")
        ):
            demo_analysis = build_deterministic_gold_analysis(live_payload)
            demo_analysis["analysis_engine"] = "demo_mode (Live Market Quotes + Quantitative Gold Engine)"
            return live_payload, demo_analysis
    except Exception:
        pass

    # Curated institutional sample fallback if executed completely offline
    sample_payload: Dict[str, Any] = {
        "generated_at_utc": now_utc,
        "fetch_latency_seconds": 1.42,
        "primary_asset": PRIMARY_ASSET,
        "target_tickers": TARGET_TICKERS,
        "gold_impact_wedges_config": GOLD_IMPACT_WEDGES,
        "intraday_breaking_news_count": 10,
        "wedge_headline_distribution": {
            "Fed / 10Y Yield / DXY": 4,
            "US Tier-1 Macro Prints": 2,
            "Geopolitics & Safe-Haven": 2,
            "CB Buying / COMEX / GLD": 2,
        },
        "gold_macro_snapshot": {
            "snapshot_utc": now_utc,
            "primary_asset": PRIMARY_ASSET,
            "tickers": {
                "GOLD_FUTURES": {
                    "ticker_key": "GOLD_FUTURES",
                    "symbol": "GC=F",
                    "display_symbol": "GC=F / XAUUSD",
                    "label": "COMEX Gold Futures (XAUUSD Proxy)",
                    "unit": "USD/oz",
                    "gold_relationship": "PRIMARY TARGET ASSET",
                    "price": 4062.40,
                    "previous_close": 4045.10,
                    "change": 17.30,
                    "change_pct": 0.43,
                    "five_day_change_pct": 1.65,
                    "day_high": 4084.10,
                    "day_low": 4040.50,
                    "gold_directional_effect": "BULLISH GOLD",
                    "regime_signal": "Active Gold bid (+0.43% intraday | 5d: +1.65%)",
                    "status": "live",
                },
                "DXY_DOLLAR_INDEX": {
                    "ticker_key": "DXY_DOLLAR_INDEX",
                    "symbol": "DX-Y.NYB",
                    "display_symbol": "DXY (DX-Y.NYB)",
                    "label": "US Dollar Index (DXY)",
                    "unit": "pts",
                    "gold_relationship": "INVERSE DRIVER (-r)",
                    "price": 102.18,
                    "previous_close": 102.52,
                    "change": -0.34,
                    "change_pct": -0.33,
                    "five_day_change_pct": -0.48,
                    "day_high": 102.58,
                    "day_low": 102.10,
                    "gold_directional_effect": "BULLISH GOLD",
                    "regime_signal": "Softening USD (-0.33% to 102.18) provides tailwind for XAUUSD",
                    "status": "live",
                },
                "US_10Y_YIELD": {
                    "ticker_key": "US_10Y_YIELD",
                    "symbol": "^TNX",
                    "display_symbol": "US 10Y (^TNX)",
                    "label": "US 10-Year Treasury Yield",
                    "unit": "%",
                    "gold_relationship": "INVERSE REAL-YIELD DRIVER (-r)",
                    "price": 4.915,
                    "previous_close": 4.965,
                    "change": -0.050,
                    "change_pct": -1.01,
                    "five_day_change_pct": -2.40,
                    "day_high": 4.980,
                    "day_low": 4.895,
                    "gold_directional_effect": "BULLISH GOLD",
                    "regime_signal": "Falling 10Y Yield (4.915%, -0.050%) compresses real rates & boosts Gold",
                    "status": "live",
                },
                "VIX_VOLATILITY": {
                    "ticker_key": "VIX_VOLATILITY",
                    "symbol": "^VIX",
                    "display_symbol": "VIX (^VIX)",
                    "label": "CBOE Volatility Index (VIX)",
                    "unit": "pts",
                    "gold_relationship": "SAFE-HAVEN FEAR GAUGE (+r on spikes)",
                    "price": 20.85,
                    "previous_close": 19.40,
                    "change": 1.45,
                    "change_pct": 7.47,
                    "five_day_change_pct": 11.20,
                    "day_high": 21.30,
                    "day_low": 19.25,
                    "gold_directional_effect": "BULLISH GOLD",
                    "regime_signal": "Elevated VIX (20.85, +7.47%) spurs safe-haven demand for Gold",
                    "status": "live",
                },
                "GLD_ETF": {
                    "ticker_key": "GLD_ETF",
                    "symbol": "GLD",
                    "display_symbol": "GLD ETF",
                    "label": "SPDR Gold Trust ETF",
                    "unit": "USD",
                    "gold_relationship": "INSTITUTIONAL FLOW PROXY (+r)",
                    "price": 372.80,
                    "previous_close": 368.35,
                    "change": 4.45,
                    "change_pct": 1.21,
                    "five_day_change_pct": 1.85,
                    "day_high": 373.40,
                    "day_low": 368.90,
                    "gold_directional_effect": "BULLISH GOLD",
                    "regime_signal": "Institutional GLD ETF bid (+1.21% | 5d: +1.85%) confirms bullion demand",
                    "status": "live",
                },
            },
            "cross_asset_gold_alignment": {
                "alignment_state": "ALIGNED BULLISH GOLD",
                "bullish_gold_drivers": ["GC=F", "DXY", "^TNX", "^VIX", "GLD"],
                "bearish_gold_drivers": [],
            },
        },
        "intraday_breaking_news": [],
        "ingestion_diagnostics": {
            "total_feeds_queried": 13,
            "active_feeds_returning_data": 13,
            "raw_gold_macro_articles_collected": 210,
            "deduplicated_top_count": 10,
            "feeds": [],
        },
    }

    demo_analysis = build_deterministic_gold_analysis(sample_payload)
    demo_analysis["analysis_engine"] = "demo_mode (Offline Sample Data)"
    return sample_payload, demo_analysis


def _print_missing_api_key_instructions() -> None:
    """Print clear instructions showing where to obtain and paste GEMINI_API_KEY."""
    env_path = Path(".env").resolve()
    example_path = Path(".env.example").resolve()
    print("\n" + "!" * 88)
    print(" [NOTICE] GEMINI_API_KEY IS NOT CONFIGURED IN `.env`")
    print("!" * 88)
    print(" To enable live Gemini 3.5 Flash LLM correlation & sentiment reasoning:")
    print("   1. Get a free API key from Google AI Studio: https://aistudio.google.com/apikey")
    print(f"   2. Copy `{example_path.name}` to `{env_path.name}` (in {env_path.parent})")
    print("   3. Paste your key on line 2 of `.env`:")
    print("      GEMINI_API_KEY=AIzaSyYourActualKeyHere")
    print("   4. Re-run: `python main.py` (or run `python main.py --demo` anytime for demo mode).")
    print("!" * 88 + "\n")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gold (XAUUSD / COMEX GC=F) Intraday News & Macro Sentiment Trading Agent"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Generate `xauusd_intraday_report.html` in demo mode without requiring GEMINI_API_KEY",
    )
    parser.add_argument(
        "--fetch-only",
        action="store_true",
        help="Fetch live Gold/Macro data and render the report using deterministic scoring without calling Gemini API",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=GEMINI_MODEL,
        help=f"Gemini model name (default: {GEMINI_MODEL})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_REPORT_FILENAME,
        help=f"Output path for the standalone Gold HTML report (default: {DEFAULT_REPORT_FILENAME})",
    )
    parser.add_argument(
        "--dump-json",
        type=str,
        default="",
        help="Optional path to save the raw JSON Gold market intelligence payload",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Automatically open the generated HTML report in the default browser",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start the Multi-User Gold Web Portal Server on http://127.0.0.1:8000",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for --serve mode (default: 8000)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args = parse_args(argv)

    has_live_api_key = bool(GEMINI_API_KEY and GEMINI_API_KEY != "your_gemini_api_key_here")

    if args.demo:
        logging.info("Running in --demo mode: building Gold (XAUUSD / GC=F) market & sentiment dataset...")
        payload, analysis = _build_demo_gold_payload_and_analysis()
    else:
        if not has_live_api_key and not args.fetch_only:
            _print_missing_api_key_instructions()

        logging.info("Fetching live Gold (XAUUSD / GC=F), DXY, ^TNX, ^VIX, GLD & 18 breaking headlines...")
        payload = fetch_market_intelligence_payload(target_tickers=TARGET_TICKERS)

        if args.fetch_only or not has_live_api_key:
            logging.info("Generating quantitative Gold correlation analysis via deterministic engine...")
            analysis = build_deterministic_gold_analysis(payload)
            analysis["analysis_engine"] = (
                "fetch_only_deterministic"
                if args.fetch_only
                else "deterministic_fallback (configure GEMINI_API_KEY in .env for Gemini LLM)"
            )
        else:
            logging.info("Running single-batch Gemini Free-Tier Gold Correlation Engine (%s)...", args.model)
            analysis = analyze_gold_market(payload, model_name=args.model)

    # Validate against GoldAnalysisReport Pydantic schema
    validated = GoldAnalysisReport.model_validate(
        {k: v for k, v in analysis.items() if k != "analysis_engine"}
    )

    if args.dump_json:
        with open(args.dump_json, "w", encoding="utf-8") as fh:
            json.dump({"payload": payload, "analysis": analysis}, fh, indent=2)
        logging.info("Saved combined Gold intelligence & analysis JSON to %s", args.dump_json)

    report_path = generate_html_report(
        payload=payload,
        analysis=analysis,
        output_path=args.output,
    )

    tickers = (payload.get("gold_macro_snapshot") or {}).get("tickers") or {}
    gc_quote = tickers.get("GOLD_FUTURES") or {}
    dxy_quote = tickers.get("DXY_DOLLAR_INDEX") or {}
    tnx_quote = tickers.get("US_10Y_YIELD") or {}
    cal_meta = payload.get("gold_economic_calendar") or {}

    print("=" * 88)
    print(" XAUUSD / COMEX GOLD (GC=F) INTRADAY SENTIMENT REPORT GENERATED")
    print("=" * 88)
    print(f"  • Gold Futures (GC=F)   : ${gc_quote.get('price', 0):,.2f}/oz ({gc_quote.get('change_pct', 0):+.2f}%)")
    print(f"  • US Dollar Index (DXY) : {dxy_quote.get('price', 0):.2f} ({dxy_quote.get('change_pct', 0):+.2f}%)")
    print(f"  • US 10Y Yield (^TNX)   : {tnx_quote.get('price', 0):.3f}% ({tnx_quote.get('change_pct', 0):+.2f}%)")
    print(f"  • Overall Gold Bias     : {validated.overall_bias}")
    print(f"  • Weighted Score        : {validated.weighted_sentiment_score:+.1f} / 100.0")
    print(f"  • Confidence Score      : {validated.confidence_score}%")
    print(f"  • Headlines Analyzed    : {len(validated.news_items)} (sorted by impact weight 10 -> 1)")
    print(
        f"  • Gold Econ Calendar    : {cal_meta.get('total_gold_events', 0)} USD/Gold Events "
        f"({cal_meta.get('high_impact_count', 0)} High-Impact Tier-1)"
    )
    print(f"  • Next Tier-1 Catalyst  : {cal_meta.get('next_major_gold_catalyst', 'N/A')}")
    print(f"  • Standalone HTML File  : {report_path}")
    print("=" * 88)

    if args.serve:
        from app import run_web_server

        run_web_server(host="0.0.0.0", port=args.port, open_browser=args.open, refresh_on_start=False)
        return 0

    if args.open:
        webbrowser.open(Path(report_path).as_uri())

    return 0


if __name__ == "__main__":
    sys.exit(main())
