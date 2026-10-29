"""
Gemini Free-Tier Gold (`XAUUSD` / `GC=F`) Correlation & Wedge Sentiment Engine (`analyzer.py`).

Uses the official `google-genai` SDK (`from google import genai` and `from google.genai import types`)
optimized for the Google AI Studio Free Tier:
- Sends the entire live Gold/Macro ticker snapshot (`GC=F`, `DX-Y.NYB`, `^TNX`, `^VIX`, `GLD`)
  AND all 18 breaking headlines in ONE single batched API request using
  `response_mime_type="application/json"` and `response_schema=GoldAnalysisReport`.
- Implements exponential backoff retry logic for HTTP 429 (`RESOURCE_EXHAUSTED`)
  and automatic model fallback (`gemini-2.5-flash` -> `gemini-2.0-flash` -> `gemini-3.8-flash`).
- Includes a quantitative deterministic Gold correlation fallback (`build_deterministic_gold_analysis`)
  when `.env` still has the placeholder API key so the pipeline and Pydantic schema can be
  verified end-to-end offline or in dry-run mode.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field
import requests

try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    genai = None
    types = None
    HAS_GENAI = False

from config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GEMINI_MODEL_CHAIN,
    GOLD_IMPACT_WEDGES,
    GROQ_API_KEY,
    GROQ_MODEL_CHAIN,
)

logger = logging.getLogger(__name__)

# Cache Gemini daily Free-Tier exhaustion so repeated refreshes skip straight to Groq in 0ms
_GEMINI_DAILY_QUOTA_EXHAUSTED: bool = False


# ---------------------------------------------------------------------------
# Required Pydantic Schema (`GoldAnalysisReport`)
# ---------------------------------------------------------------------------
class GoldWedgeBreakdown(BaseModel):
    wedge_name: str = Field(
        description="Name of the Gold Impact Wedge (Wedge 1 through Wedge 4)"
    )
    weight_percentage: int = Field(
        description="Configured portfolio weight percentage of this wedge (40, 30, 20, or 10)"
    )
    net_sentiment_score: float = Field(
        ge=-100.0,
        le=100.0,
        description="Net Gold sentiment score for this wedge from -100.0 (Max Bearish Gold) to +100.0 (Max Bullish Gold)",
    )
    summary: str = Field(
        description="Concise institutional synthesis of how this wedge is currently driving XAUUSD / GC=F"
    )


class GoldHeadlineAnalysis(BaseModel):
    headline: str = Field(description="Exact breaking news headline text")
    source_time: str = Field(description="Publisher source and UTC timestamp / relative age")
    gold_wedge_category: str = Field(
        description="Which of the 4 Gold Impact Wedges this headline belongs to"
    )
    impact_weight: int = Field(
        ge=1,
        le=10,
        description="Quantitative impact weight from 1 (low noise) to 10 (systemic Gold catalyst)",
    )
    gold_bias: Literal["BULLISH GOLD", "BEARISH GOLD", "NEUTRAL"] = Field(
        description="Directional impact on Gold (XAUUSD / GC=F)"
    )
    sentiment_score: int = Field(
        ge=-100,
        le=100,
        description="Headline Gold sentiment score from -100 (extremely bearish Gold) to +100 (extremely bullish Gold)",
    )
    gold_transmission_mechanism: str = Field(
        description="1 concise sentence explaining WHY this headline causes buying or selling in XAUUSD / GC"
    )


class GoldAnalysisReport(BaseModel):
    overall_bias: Literal[
        "STRONG BUY",
        "BUY",
        "NEUTRAL / CHOPPY",
        "SELL",
        "STRONG SELL",
    ] = Field(description="Overall directional trading bias for Gold (XAUUSD / GC=F)")
    weighted_sentiment_score: float = Field(
        ge=-100.0,
        le=100.0,
        description="Composite wedge-weighted sentiment score from -100.0 (Max Bearish Gold) to +100.0 (Max Bullish Gold)",
    )
    confidence_score: int = Field(
        ge=0,
        le=100,
        description=(
            "Confidence score (0-100). High (80-100) ONLY when DXY, ^TNX yields, US macro conditions, "
            "and high-weight breaking news align in the same direction for Gold; Low/Medium (30-65) "
            "when intraday news conflicts with macro/yields."
        ),
    )
    macro_condition_summary: str = Field(
        description=(
            "Detailed institutional evaluation of current DXY, US 10Y Yields (^TNX), VIX, and "
            "Fed/Inflation conditions and how they are impacting XAUUSD right now."
        )
    )
    why_buy_or_sell_thesis: List[str] = Field(
        min_length=4,
        max_length=6,
        description=(
            "List of 4-6 clear institutional bullet points explaining WHY the bias is Buying or "
            "Selling Gold based on the weighted wedges, plus the #1 intraday invalidation risk."
        ),
    )
    wedge_breakdown: List[GoldWedgeBreakdown] = Field(
        min_length=4,
        max_length=4,
        description="Breakdown of all 4 Gold Impact Wedges (40%, 30%, 20%, 10%)",
    )
    news_items: List[GoldHeadlineAnalysis] = Field(
        description="Analyzed breakdown of the intraday breaking news headlines"
    )


# ---------------------------------------------------------------------------
# System Prompt Encoding Institutional Gold (`XAUUSD` / `GC=F`) Mechanics
# ---------------------------------------------------------------------------
GOLD_SYSTEM_INSTRUCTION = """You are a Senior Institutional Precious Metals Quantitative Strategist specializing exclusively in Gold (`XAUUSD` spot bullion and COMEX Gold Futures `GC=F`).

You will receive a single batched JSON payload containing:
1. Live intraday prices, % changes, and intraday high/low ranges for:
   - `GC=F` (COMEX Gold Futures / XAUUSD proxy)
   - `DX-Y.NYB` (US Dollar Index / DXY — primary inverse driver)
   - `^TNX` (US 10-Year Treasury Yield — real yield opportunity-cost inverse driver)
   - `^VIX` (CBOE Volatility Index — safe-haven fear gauge)
   - `GLD` (SPDR Gold Trust ETF — institutional Gold flow proxy)
2. The top 18 deduplicated intraday breaking news headlines (with timestamps and pre-tagged wedge metadata).

APPLY THESE EXACT INSTITUTIONAL GOLD (`XAUUSD` / `GC=F`) CORRELATION & WEDGE SCORING RULES:

1. THE 4 GOLD IMPACT WEDGES & WEIGHTS:
   - Wedge 1 (40% Weight / Impact Weight 8-10): Fed/FOMC Policy, US 10Y Treasury Yield (`^TNX`), and US Dollar Index (`DXY`).
   - Wedge 2 (30% Weight / Impact Weight 7-9): US Tier-1 Economic Data (CPI, Core PCE, NFP, Jobless Claims, GDP, ISM PMI).
   - Wedge 3 (20% Weight / Impact Weight 8-10 on spikes): Geopolitical Risk, Tariffs, War Escalation & Safe-Haven Flows (`^VIX`).
   - Wedge 4 (10% Weight / Impact Weight 3-6): Central Bank Gold Buying, COMEX physical positioning & ETF (`GLD`) flows.

2. INVERSE YIELD & USD RULE (Wedge 1 - 40%):
   - Because physical Gold pays zero coupon/yield, rising `^TNX` (US 10Y Yields) and a rising `DXY` (US Dollar Index) increase the opportunity cost and foreign-currency price of holding Gold -> BEARISH (`SELL`) for `XAUUSD`.
   - Falling `^TNX` yields, dovish Federal Reserve commentary, rate-cut pricing, and a weaker `DXY` -> BULLISH (`BUY`) for `XAUUSD`.

3. US ECONOMIC DATA RULE (Wedge 2 - 30%):
   - Hotter-than-expected CPI/PCE inflation or booming NFP/GDP prints that force the Fed to delay rate cuts (or hike rates) -> BEARISH Gold (`SELL`).
   - Cooling inflation, rising unemployment/jobless claims, weak NFP/ISM PMI (which accelerates Fed rate cuts), OR Stagflation (weak real growth + sticky inflation) -> BULLISH Gold (`BUY`).

4. GEOPOLITICAL SAFE-HAVEN OVERRIDE RULE (Wedge 3 - 20%, Weight 8-10 on spikes):
   - Acute geopolitical escalation, military strikes, war headlines, sovereign debt/fiscal crises, tariff shocks, or banking stress -> High-Weight BULLISH Gold (`+80` to `+100`).
   - CRITICAL OVERRIDE: During acute geopolitical panic or systemic fear, safe-haven Gold buying OVERRIDES a rising `DXY` (Gold and `DXY` rally together as global capital flees into both USD and bullion).

5. CONFIDENCE SCORE CALIBRATION (`0` to `100`):
   - Output High Confidence (`80-100%`) ONLY when `DXY`, `^TNX` yields, US macro conditions, and high-weight breaking news all align in the same direction for Gold (e.g., if all are bearish for Gold, output `SELL` or `STRONG SELL` with `85-98%` confidence; if all are bullish, output `BUY` or `STRONG BUY` with `85-98%` confidence).
   - Output Low/Medium Confidence (`30-65%`) when intraday news conflicts with macro/yields (e.g., strong DXY/Yields pulling Gold down while geopolitical headlines or central bank buying support safe-haven demand).

Analyze all 18 provided breaking news items in `news_items`, provide all 4 wedges in `wedge_breakdown`, provide 4-6 actionable institutional bullets in `why_buy_or_sell_thesis` (always including the #1 intraday invalidation risk as the final bullet), and return strictly valid JSON matching `GoldAnalysisReport`.
"""


# ---------------------------------------------------------------------------
# Deterministic Quantitative Fallback Engine (Used when API Key is Placeholder)
# ---------------------------------------------------------------------------
_BULLISH_GOLD_CUES: Dict[str, Tuple[int, str]] = {
    "rate cut": (+65, "Dovish Fed rate-cut expectations lower real yields and reduce the opportunity cost of holding non-yielding XAUUSD."),
    "dovish": (+60, "Dovish central bank rhetoric compresses sovereign bond yields, driving institutional inflows into Gold."),
    "yields fall": (+60, "Declining Treasury yields reduce real rate headwinds and directly boost XAUUSD valuation."),
    "falling yields": (+60, "Falling US yields lower carry costs for bullion, supporting upside in COMEX Gold futures."),
    "war": (+85, "Acute military conflict triggers immediate safe-haven capital rotation into physical Gold, overriding USD strength."),
    "iran": (+80, "Middle East geopolitical tensions inject geopolitical risk premia into XAUUSD safe-haven flows."),
    "israel": (+80, "Escalating Middle East hostilities drive urgent geopolitical tail-risk hedging in Gold."),
    "ukraine": (+75, "Eastern European war escalation reinforces sovereign and institutional safe-haven demand for bullion."),
    "tariff": (+70, "Trade war and tariff uncertainty stoke stagflationary fears and safe-haven demand for Gold."),
    "trade war": (+75, "Global trade fragmentation accelerates de-dollarization and safe-haven accumulation of XAUUSD."),
    "safe haven": (+80, "Explicit safe-haven flows lift spot XAUUSD demand regardless of short-term DXY fluctuations."),
    "safe-haven": (+80, "Heightened cross-asset risk aversion channels defensive liquidity directly into Gold."),
    "geopolitical concerns": (+65, "Rising geopolitical concerns stimulate defensive safe-haven bids in bullion and commodities."),
    "central bank": (+55, "Sovereign central bank reserve diversification provides price-inelastic physical floor support under XAUUSD."),
    "gold reserves": (+60, "Official sector bullion accumulation tightens physical supply and supports structural Gold upside."),
    "unemployment": (+55, "Labor market softening increases probability of Fed easing, pressuring real yields and lifting Gold."),
    "jobless claims": (+50, "Rising jobless claims signal cooling US labor demand, pulling forward Fed rate-cut expectations for XAUUSD."),
    "stagflation": (+80, "Stagflationary macro dynamics (sticky inflation + slowing growth) represent the strongest historical regime for Gold."),
    "recession": (+70, "Recessionary tail-risk drives defensive allocation into Gold and compresses front-end Treasury yields."),
    "gold rallies": (+55, "Strong upside momentum in spot XAUUSD attracts systematic trend-following inflows."),
    "gold surges": (+65, "Breakout buying in Gold futures signals aggressive institutional accumulation."),
    "gold climbs": (+45, "Sustained buying interest lifts COMEX Gold futures alongside supportive macro flows."),
    "record high": (+65, "Breakout to record highs reinforces structural bullish momentum in XAUUSD."),
    "inflows": (+50, "Bullion ETF inflows confirm institutional real-money accumulation of physical Gold."),
}

_BEARISH_GOLD_CUES: Dict[str, Tuple[int, str]] = {
    "rate hike": (-75, "Hawkish rate-hike risk pushes real yields and DXY higher, penalizing zero-yielding Gold."),
    "hawkish": (-65, "Hawkish Fed positioning lifts real Treasury yields and strengthens DXY against XAUUSD."),
    "wait-and-see": (-45, "Fed patience on rate cuts keeps real interest rates elevated for longer, capping Gold upside."),
    "hold interest rates steady": (-40, "FOMC rate pause keeps real opportunity cost elevated for zero-coupon Gold."),
    "holds interest rates steady": (-40, "FOMC rate pause keeps real opportunity cost elevated for zero-coupon Gold."),
    "holds interest rate steady": (-40, "FOMC rate pause keeps real opportunity cost elevated for zero-coupon Gold."),
    "rate hold": (-45, "FOMC rate hold keeps real yields elevated and pressures non-yielding XAUUSD."),
    "holds key rate steady": (-40, "Federal Reserve rate hold amid data dependency delays monetary easing and pressures Gold."),
    "keeps rates unchanged": (-45, "FOMC holding policy rates unchanged despite inflation sustains elevated real yields against XAUUSD."),
    "keeps interest rates unchanged": (-45, "Federal Reserve keeping interest rates unchanged maintains high carry costs for non-yielding bullion."),
    "leaves rates steady": (-40, "Fed leaving policy rates steady keeps real Treasury yields elevated and caps Gold upside."),
    "skips rate hike": (-35, "Fed pausing without signaling rate cuts keeps restrictive policy in place, weighing on Gold."),
    "elevated inflation": (-45, "Persistent elevated inflation prevents near-term Fed rate cuts and keeps US real yields high."),
    "starting to work": (-35, "Fed confidence in restrictive policy posture signals no urgency to cut rates, pressuring XAUUSD."),
    "reduce the number of rate-setting meetings": (-25, "Structural shift toward fewer FOMC rate-setting meetings signals policy steadiness and higher-for-longer rates."),
    "meeting schedule will change": (-20, "FOMC schedule adjustment reinforces deliberate, unhurried monetary policy path."),
    "higher for longer": (-65, "Higher-for-longer policy rates sustain elevated opportunity costs for holding non-yielding bullion."),
    "inflation quickens": (-50, "Sticky inflation forces central banks to maintain restrictive policy rates, pressuring Gold via higher yields."),
    "cpi challenge": (-45, "Persistent core CPI inflation delays FOMC monetary easing and props up US Treasury yields."),
    "worrying about inflation": (-35, "Renewed inflation concern risks keeping Fed policy restrictive and 10Y yields elevated near 5%."),
    "strong dollar": (-65, "A strengthening US Dollar Index (DXY) makes USD-denominated Gold more expensive for international buyers."),
    "stronger dollar": (-65, "A firmer US Dollar Index (DXY) and hawkish Fed rate outlook weigh directly on USD-denominated bullion."),
    "weigh on bullion": (-60, "Macro headwinds from a firmer USD and elevated yields trigger institutional selling in XAUUSD."),
    "prices decline": (-55, "Active intraday distribution in Gold futures reflects pressure from real yields and USD firmness."),
    "pulls back": (-45, "Intraday pullback in XAUUSD reflects pre-FOMC profit-taking and pressure from elevated US yields."),
    "pull back": (-45, "Intraday pullback in XAUUSD reflects pre-FOMC profit-taking and pressure from elevated US yields."),
    "edges lower": (-40, "Gold edging lower ahead of central bank commentary reflects cautious positioning and real-yield headwinds."),
    "edge lower": (-40, "Gold edging lower ahead of central bank commentary reflects cautious positioning and real-yield headwinds."),
    "gold slips": (-40, "Softening spot Gold prices reflect rising real-rate opportunity costs."),
    "gold drops": (-50, "Sharp downside pressure in XAUUSD reflects institutional liquidation on higher yields."),
    "gold price drops": (-50, "Intraday drop in spot Gold reflects selling pressure following the FOMC rate decision."),
    "price drops": (-45, "Intraday price decline in XAUUSD reflects macro rate headwinds."),
    "gold falls": (-45, "Declining bullion prices reflect headwinds from elevated sovereign bond yields."),
    "remain below": (-30, "Capped upside below key technical resistance reflects pre-FOMC caution among bullion traders."),
    "halts below": (-35, "Stalled XAUUSD recovery below resistance signals persistent macro headwinds from yields and DXY."),
    "support at risk": (-40, "Pre-FOMC positioning places key XAUUSD technical support levels under downside pressure."),
    "yields rise": (-65, "Rising sovereign bond yields directly increase the real opportunity cost of holding non-yielding XAUUSD."),
    "hike bets": (-60, "Rising market probability of a Fed rate hike strengthens DXY and real yields against Gold."),
    "december hike": (-60, "Elevated December rate-hike pricing lifts Treasury yields and pressures non-yielding bullion."),
    "raises policy rate": (-35, "Global central bank policy rate hikes reinforce a higher-rate regime that weighs on Gold."),
    "rate pause": (-35, "Fed rate pause without dovish guidance keeps real yields elevated and caps XAUUSD."),
    "inflation forecast": (-35, "Upwardly revised inflation bars keep the Fed hawkish and real yields elevated."),
    "diplomacy a chance": (-20, "Diplomatic openness reduces immediate war-escalation risk, moderating safe-haven Gold demand."),
    "nuclear inspectors": (-20, "Diplomatic mediation over nuclear inspections tempers immediate geopolitical escalation premia in Gold."),
    "ceasefire": (-60, "Geopolitical de-escalation unwinds war-risk premia embedded in spot Gold prices."),
    "seeks peace": (-45, "Peace negotiations reduce acute geopolitical tail-risk premia in safe-haven Gold."),
    "draft deal": (-45, "Diplomatic progress toward de-escalation moderates urgent geopolitical safe-haven bidding in XAUUSD."),
    "outflows": (-50, "Gold ETF redemptions add physical supply back to the market and weigh on GC=F."),
}

_NEUTRAL_EDUCATIONAL_CUES: Tuple[str, ...] = (
    "playbook",
    "trader's guide",
    "traders guide",
    "confirmation checklist",
    "breakout or trap",
    "due thursday",
    "interchange fee",
)


def _cue_matches(cue: str, text: str) -> bool:
    """Match a keyword/phrase cue using word boundaries to prevent substring collisions (e.g., 'war' in 'towards')."""
    return bool(re.search(rf"\b{re.escape(cue)}\b", text))


def _infer_wedge_id(wedge_name: str, fallback_id: int = 4) -> int:
    lower = (wedge_name or "").lower()
    if "wedge 1" in lower or "fed" in lower or "yield" in lower or "dxy" in lower:
        return 1
    if "wedge 2" in lower or "tier-1" in lower or "cpi" in lower or "pce" in lower:
        return 2
    if "wedge 3" in lower or "geopolit" in lower or "safe-haven" in lower or "vix" in lower:
        return 3
    if "wedge 4" in lower or "central bank" in lower or "gld" in lower or "comex" in lower:
        return 4
    return fallback_id


def _score_headline_deterministically(item: Dict[str, Any]) -> GoldHeadlineAnalysis:
    """
    Score a single headline using deterministic institutional Gold correlation rules.
    """
    headline = item.get("headline", "")
    summary = item.get("summary", "")
    combined = f"{headline} {summary}".lower()
    wedge = item.get("gold_wedge") or {}
    wedge_name = wedge.get(
        "wedge_name",
        GOLD_IMPACT_WEDGES["WEDGE_4_CENTRAL_BANK_ETF_FLOWS"]["wedge_name"],
    )
    w_id = int(wedge.get("wedge_id") or _infer_wedge_id(wedge_name, 4))
    impact_weight = int(wedge.get("impact_weight", 6))

    if any(cue in headline.lower() for cue in _NEUTRAL_EDUCATIONAL_CUES):
        return GoldHeadlineAnalysis(
            headline=headline,
            source_time=item.get("source_time") or f"{item.get('source', 'News')} | {item.get('published_utc', '')}",
            gold_wedge_category=wedge_name,
            impact_weight=min(impact_weight, 4),
            gold_bias="NEUTRAL",
            sentiment_score=0,
            gold_transmission_mechanism="Educational, technical, or scheduled preview item with neutral immediate impact on spot XAUUSD.",
        )

    matched_scores: List[int] = []
    mechanisms: List[str] = []

    # Check bearish cues first so explicit downside headlines prioritize bearish mechanism
    for cue, (score, mech) in _BEARISH_GOLD_CUES.items():
        if _cue_matches(cue, combined):
            matched_scores.append(score)
            mechanisms.append(mech)

    for cue, (score, mech) in _BULLISH_GOLD_CUES.items():
        if _cue_matches(cue, combined):
            # If diplomatic de-escalation cue already matched, skip generic 'iran'/'war' bullish cue
            if matched_scores and cue in ("iran", "israel", "ukraine", "war") and any(s < 0 for s in matched_scores):
                continue
            matched_scores.append(score)
            mechanisms.append(mech)

    if matched_scores:
        avg_score = int(round(sum(matched_scores) / len(matched_scores)))
        if w_id == 3 and avg_score > 0:
            avg_score = max(avg_score, 40)
            impact_weight = max(impact_weight, 9)
        mechanism = mechanisms[0]
    else:
        # Conservative default inference by wedge context (never artificially inflate to +65/+30)
        if w_id == 1:
            avg_score = -25
            mechanism = "Elevated US yields and cautious Fed policy keep real-rate opportunity costs in focus for XAUUSD."
        elif w_id == 2:
            avg_score = -15
            mechanism = "Sticky US inflation and macro expectations keep Fed rate-cut timing cautious for Gold."
        elif w_id == 3:
            avg_score = +20
            mechanism = "Ongoing geopolitical headlines maintain a modest safe-haven floor under Gold."
        else:
            avg_score = 0
            impact_weight = min(impact_weight, 4)
            mechanism = "Neutral precious metals positioning with no immediate directional catalyst."

    avg_score = max(-100, min(100, avg_score))
    if avg_score >= 15:
        gold_bias: Literal["BULLISH GOLD", "BEARISH GOLD", "NEUTRAL"] = "BULLISH GOLD"
    elif avg_score <= -15:
        gold_bias = "BEARISH GOLD"
    else:
        gold_bias = "NEUTRAL"

    return GoldHeadlineAnalysis(
        headline=headline,
        source_time=item.get("source_time") or f"{item.get('source', 'News')} | {item.get('published_utc', '')}",
        gold_wedge_category=wedge_name,
        impact_weight=max(1, min(10, impact_weight)),
        gold_bias=gold_bias,
        sentiment_score=avg_score,
        gold_transmission_mechanism=mechanism,
    )


def build_deterministic_gold_analysis(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a mathematically consistent, schema-validated `GoldAnalysisReport`
    using deterministic quantitative Gold correlation rules when `GEMINI_API_KEY`
    is not yet configured or when running in `--dry-run` mode.
    """
    snap = payload.get("gold_macro_snapshot") or {}
    tickers = snap.get("tickers") or {}

    gc = tickers.get("GOLD_FUTURES") or {}
    dxy = tickers.get("DXY_DOLLAR_INDEX") or {}
    tnx = tickers.get("US_10Y_YIELD") or {}
    vix = tickers.get("VIX_VOLATILITY") or {}
    gld = tickers.get("GLD_ETF") or {}

    gc_price = gc.get("price") or 0.0
    gc_chg = gc.get("change_pct") or 0.0
    dxy_price = dxy.get("price") or 0.0
    dxy_chg = dxy.get("change_pct") or 0.0
    tnx_price = tnx.get("price") or 0.0
    tnx_chg = tnx.get("change_pct") or 0.0
    vix_price = vix.get("price") or 0.0
    vix_chg = vix.get("change_pct") or 0.0
    gld_price = gld.get("price") or 0.0
    gld_chg = gld.get("change_pct") or 0.0

    # Score all 18 headlines
    raw_news = payload.get("intraday_breaking_news") or []
    scored_news: List[GoldHeadlineAnalysis] = [
        _score_headline_deterministically(item) for item in raw_news
    ]

    # Group headline scores by wedge
    wedge_headline_scores: Dict[int, List[Tuple[int, int]]] = {1: [], 2: [], 3: [], 4: []}
    for raw_item, analyzed in zip(raw_news, scored_news):
        w_id = (raw_item.get("gold_wedge") or {}).get("wedge_id", 4)
        wedge_headline_scores.setdefault(w_id, []).append(
            (analyzed.sentiment_score, analyzed.impact_weight)
        )

    def _weighted_news_avg(pairs: List[Tuple[int, int]], default_val: float) -> float:
        if not pairs:
            return default_val
        num = sum(s * w for s, w in pairs)
        den = sum(w for _, w in pairs)
        return round(num / max(den, 1), 1)

    # Compute live market adjustments for each wedge
    # Wedge 1 (40%): Inverse DXY & 10Y Yield + Wedge 1 news
    macro_w1_score = 0.0
    if tnx_price >= 4.50:
        macro_w1_score -= 25.0
    if tnx_chg > 0.25:
        macro_w1_score -= 20.0
    elif tnx_chg < -0.25:
        macro_w1_score += 25.0

    if dxy_chg > 0.15:
        macro_w1_score -= 20.0
    elif dxy_chg < -0.15:
        macro_w1_score += 20.0

    w1_news = _weighted_news_avg(wedge_headline_scores[1], -15.0)
    w1_score = max(-100.0, min(100.0, round(0.55 * macro_w1_score + 0.45 * w1_news, 1)))

    # Wedge 2 (30%): US Tier-1 Economic Data
    w2_score = max(-100.0, min(100.0, _weighted_news_avg(wedge_headline_scores[2], 10.0)))

    # Wedge 3 (20%): Geopolitical Risk & Safe-Haven (^VIX)
    vix_bonus = 25.0 if vix_price >= 20.0 else (10.0 if vix_price >= 17.0 else -10.0)
    w3_news = _weighted_news_avg(wedge_headline_scores[3], 45.0)
    w3_score = max(-100.0, min(100.0, round(0.35 * vix_bonus + 0.65 * w3_news, 1)))

    # Wedge 4 (10%): Central Bank Buying, COMEX & GLD ETF flows
    flow_bonus = 35.0 if (gc_chg > 0.2 or gld_chg > 0.2) else (-35.0 if (gc_chg < -0.2 or gld_chg < -0.2) else 15.0)
    w4_news = _weighted_news_avg(wedge_headline_scores[4], 25.0)
    w4_score = max(-100.0, min(100.0, round(0.50 * flow_bonus + 0.50 * w4_news, 1)))

    # Composite Weighted Sentiment Score (40% W1 + 30% W2 + 20% W3 + 10% W4)
    composite_score = round(
        0.40 * w1_score + 0.30 * w2_score + 0.20 * w3_score + 0.10 * w4_score,
        1,
    )

    # Determine Overall Bias
    if composite_score >= 45.0:
        overall_bias: Literal["STRONG BUY", "BUY", "NEUTRAL / CHOPPY", "SELL", "STRONG SELL"] = "STRONG BUY"
    elif composite_score >= 15.0:
        overall_bias = "BUY"
    elif composite_score <= -45.0:
        overall_bias = "STRONG SELL"
    elif composite_score <= -15.0:
        overall_bias = "SELL"
    else:
        overall_bias = "NEUTRAL / CHOPPY"

    # Calibrate Confidence Score (80-100% ONLY when all wedges/macro align; 30-65% when conflicting)
    wedge_signs = [1 if s > 10 else (-1 if s < -10 else 0) for s in (w1_score, w2_score, w3_score, w4_score)]
    all_positive = all(sign >= 0 for sign in wedge_signs) and sum(wedge_signs) >= 3
    all_negative = all(sign <= 0 for sign in wedge_signs) and sum(wedge_signs) <= -3

    if all_positive or all_negative:
        confidence_score = min(96, max(82, int(80 + abs(composite_score) * 0.18)))
    else:
        # Conflicting cross-currents between Yields/DXY and Safe-Haven/News
        confidence_score = min(64, max(38, int(45 + abs(composite_score) * 0.30)))

    macro_summary = (
        f"COMEX Gold Futures (`GC=F`) are trading at ${gc_price:,.2f}/oz ({gc_chg:+.2f}% intraday, "
        f"range ${gc.get('day_low', gc_price):,.2f}-${gc.get('day_high', gc_price):,.2f}) alongside SPDR `GLD` "
        f"at ${gld_price:,.2f} ({gld_chg:+.2f}%). Across primary macro drivers, the US 10-Year Treasury Yield "
        f"(`^TNX`) sits at {tnx_price:.3f}% ({tnx_chg:+.2f}%) and the US Dollar Index (`DXY`) is at "
        f"{dxy_price:.2f} ({dxy_chg:+.2f}%), while the CBOE `VIX` reads {vix_price:.2f} ({vix_chg:+.2f}%). "
        f"Wedge 1 (Fed/Yields/DXY at {w1_score:+.1f}) and Wedge 3 (Geopolitics/Safe-Haven at {w3_score:+.1f}) "
        f"produce a composite Gold sentiment score of {composite_score:+.1f} ({overall_bias}) with "
        f"{confidence_score}% conviction."
    )

    wedge_breakdown = [
        GoldWedgeBreakdown(
            wedge_name=GOLD_IMPACT_WEDGES["WEDGE_1_FED_YIELD_DXY"]["wedge_name"],
            weight_percentage=40,
            net_sentiment_score=w1_score,
            summary=(
                f"US 10Y Yield at {tnx_price:.3f}% ({tnx_chg:+.2f}%) and DXY at {dxy_price:.2f} ({dxy_chg:+.2f}%) "
                f"combine with FOMC policy headlines for a {w1_score:+.1f} net score on zero-yielding XAUUSD."
            ),
        ),
        GoldWedgeBreakdown(
            wedge_name=GOLD_IMPACT_WEDGES["WEDGE_2_US_TIER1_ECON"]["wedge_name"],
            weight_percentage=30,
            net_sentiment_score=w2_score,
            summary=(
                f"US Tier-1 inflation (CPI/Core PCE), labor (NFP/claims), and GDP prints register a "
                f"{w2_score:+.1f} net score for Gold rate-cut timing."
            ),
        ),
        GoldWedgeBreakdown(
            wedge_name=GOLD_IMPACT_WEDGES["WEDGE_3_GEOPOLITICAL_SAFE_HAVEN"]["wedge_name"],
            weight_percentage=20,
            net_sentiment_score=w3_score,
            summary=(
                f"Geopolitical risk, tariff headlines, and VIX at {vix_price:.2f} generate a "
                f"{w3_score:+.1f} safe-haven flow score for physical bullion."
            ),
        ),
        GoldWedgeBreakdown(
            wedge_name=GOLD_IMPACT_WEDGES["WEDGE_4_CENTRAL_BANK_ETF_FLOWS"]["wedge_name"],
            weight_percentage=10,
            net_sentiment_score=w4_score,
            summary=(
                f"Central bank bullion accumulation, COMEX futures positioning, and GLD ETF action "
                f"(${gld_price:.2f}, {gld_chg:+.2f}%) contribute a {w4_score:+.1f} structural flow score."
            ),
        ),
    ]

    why_thesis = [
        (
            f"Wedge 1 (40% Weight | Score {w1_score:+.1f}): US 10Y Treasury Yield (`^TNX`) at {tnx_price:.3f}% "
            f"({tnx_chg:+.2f}%) and `DXY` at {dxy_price:.2f} ({dxy_chg:+.2f}%) dictate real-rate opportunity "
            f"cost for non-yielding `XAUUSD`."
        ),
        (
            f"Wedge 2 (30% Weight | Score {w2_score:+.1f}): US Tier-1 macro headlines (Core CPI/PCE, labor market "
            f"claims, and GDP) shape FOMC rate expectations and real-yield direction."
        ),
        (
            f"Wedge 3 (20% Weight | Score {w3_score:+.1f}): Geopolitical and tariff headlines alongside `^VIX` at "
            f"{vix_price:.2f} provide safe-haven tail-risk support under dips in `GC=F`."
        ),
        (
            f"Wedge 4 (10% Weight | Score {w4_score:+.1f}): Sovereign central bank reserve demand and `GLD` ETF "
            f"price action (${gld_price:.2f}, {gld_chg:+.2f}%) anchor physical bullion flows."
        ),
        (
            f"INVALIDATION RISK (#1 Intraday Trigger): A breakout in `DXY` above {(dxy_price * 1.005):.2f} paired "
            f"with a `^TNX` 10Y yield spike above {(tnx_price + 0.06):.2f}% (or sudden geopolitical de-escalation) "
            f"would invalidate upside setups and force long liquidation below ${gc.get('day_low', gc_price):,.2f}/oz."
        ),
    ]

    report = GoldAnalysisReport(
        overall_bias=overall_bias,
        weighted_sentiment_score=composite_score,
        confidence_score=confidence_score,
        macro_condition_summary=macro_summary,
        why_buy_or_sell_thesis=why_thesis,
        wedge_breakdown=wedge_breakdown,
        news_items=scored_news,
    )
    return _reconcile_and_validate_analysis(report.model_dump(), payload)


def _reconcile_and_validate_analysis(
    output: Dict[str, Any],
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Enforce mathematical and directional consistency across all headlines, the 4 Gold
    Impact Wedges, the composite `weighted_sentiment_score`, and `overall_bias`
    so results never contradict each other or jump arbitrarily across models.
    """
    snap = payload.get("gold_macro_snapshot") or {}
    tickers = snap.get("tickers") or {}
    gc = tickers.get("GOLD_FUTURES") or {}
    dxy = tickers.get("DXY_DOLLAR_INDEX") or {}
    tnx = tickers.get("US_10Y_YIELD") or {}
    vix = tickers.get("VIX_VOLATILITY") or {}
    gld = tickers.get("GLD_ETF") or {}

    gc_chg = float(gc.get("change_pct") or 0.0)
    dxy_chg = float(dxy.get("change_pct") or 0.0)
    tnx_price = float(tnx.get("price") or 0.0)
    tnx_chg = float(tnx.get("change_pct") or 0.0)
    vix_price = float(vix.get("price") or 0.0)
    gld_chg = float(gld.get("change_pct") or 0.0)

    # 1. Reconcile every headline item for score <-> bias & impact_weight consistency
    explicit_bearish_phrases = (
        "pulls back",
        "pull back",
        "edges lower",
        "edge lower",
        "prices decline",
        "gold slips",
        "gold drops",
        "gold falls",
        "weigh on bullion",
    )
    # Map raw breaking news metadata by normalized headline for cross-model anchoring
    raw_news_map: Dict[str, Dict[str, Any]] = {}
    for raw_item in payload.get("intraday_breaking_news") or []:
        hl_key = (raw_item.get("headline") or "").strip().lower()
        if hl_key:
            raw_news_map[hl_key] = raw_item

    reconciled_news: List[Dict[str, Any]] = []
    wedge_scores_by_id: Dict[int, List[Tuple[int, int]]] = {1: [], 2: [], 3: [], 4: []}

    for item in output.get("news_items") or []:
        n = dict(item)
        hl_lower = (n.get("headline") or "").strip().lower()
        raw_ref = raw_news_map.get(hl_lower)
        det_ref = _score_headline_deterministically(raw_ref) if raw_ref else None

        score = int(n.get("sentiment_score", 0))
        bias = (n.get("gold_bias") or "NEUTRAL").upper()
        weight = int(n.get("impact_weight", 5))
        wedge_cat = n.get("gold_wedge_category") or ""

        # Lock wedge category to pre-classified wedge so different LLMs never bucket headlines into different wedges
        if raw_ref and (raw_ref.get("gold_wedge") or {}).get("wedge_name"):
            wedge_cat = raw_ref["gold_wedge"]["wedge_name"]
            n["gold_wedge_category"] = wedge_cat

        # Blend LLM score with deterministic rule score for cross-model stability
        if det_ref is not None:
            if any(cue in hl_lower for cue in _NEUTRAL_EDUCATIONAL_CUES):
                score = 0
                weight = min(weight, 4)
            elif det_ref.sentiment_score != 0:
                score = int(round(0.25 * score + 0.75 * det_ref.sentiment_score))
                if (det_ref.sentiment_score <= -20 and score > -15) or (
                    det_ref.sentiment_score >= 20 and score < 15
                ):
                    score = det_ref.sentiment_score

        # Prevent explicitly bearish Gold price-action headlines from being scored positive
        if any(p in hl_lower for p in explicit_bearish_phrases) and score > -20:
            score = -40
            bias = "BEARISH GOLD"
            n["gold_transmission_mechanism"] = (
                "Intraday downside pressure in spot XAUUSD reflects pre-event caution and firm US real yields."
            )

        # Align borderline LLM scores with explicit LLM bias labels
        if bias == "BEARISH GOLD" and -14 <= score < -5:
            score = -20
        elif bias == "BULLISH GOLD" and 5 < score <= 14:
            score = 20

        # Strictly enforce bias label from score thresholds
        if score >= 15:
            bias = "BULLISH GOLD"
        elif score <= -15:
            bias = "BEARISH GOLD"
        else:
            bias = "NEUTRAL"
            score = 0 if abs(score) <= 8 else score
            # Prevent neutral/educational/noise items from carrying a 9/10 High weight
            if abs(score) <= 10:
                weight = min(weight, 4)

        n["sentiment_score"] = max(-100, min(100, score))
        n["gold_bias"] = bias
        n["impact_weight"] = max(1, min(10, weight))
        reconciled_news.append(n)

        w_id = _infer_wedge_id(wedge_cat, 4)
        wedge_scores_by_id[w_id].append((n["sentiment_score"], n["impact_weight"]))

    output["news_items"] = reconciled_news

    # 2. Compute mathematically grounded Wedge Scores (blending headline scores + live ticker regime)
    def _w_avg(pairs: List[Tuple[int, int]], fallback: float) -> float:
        if not pairs:
            return fallback
        num = sum(s * w for s, w in pairs)
        den = sum(w for _, w in pairs)
        return round(num / max(den, 1), 1)

    macro_w1 = 0.0
    if tnx_price >= 4.50:
        macro_w1 -= 20.0
    if tnx_chg > 0.25:
        macro_w1 -= 20.0
    elif tnx_chg < -0.25:
        macro_w1 += 20.0
    if dxy_chg > 0.15:
        macro_w1 -= 15.0
    elif dxy_chg < -0.15:
        macro_w1 += 10.0

    w1_calc = max(-100.0, min(100.0, round(0.45 * macro_w1 + 0.55 * _w_avg(wedge_scores_by_id[1], -20.0), 1)))
    w2_calc = max(-100.0, min(100.0, round(_w_avg(wedge_scores_by_id[2], -10.0 if tnx_chg > 0.2 else 10.0), 1)))
    vix_adj = 25.0 if vix_price >= 20.0 else (5.0 if vix_price >= 17.0 else -10.0)
    w3_calc = max(-100.0, min(100.0, round(0.30 * vix_adj + 0.70 * _w_avg(wedge_scores_by_id[3], 20.0), 1)))
    flow_adj = 30.0 if (gc_chg > 0.2 or gld_chg > 0.2) else (-30.0 if (gc_chg < -0.2 or gld_chg < -0.2) else 0.0)
    w4_calc = max(-100.0, min(100.0, round(0.50 * flow_adj + 0.50 * _w_avg(wedge_scores_by_id[4], -10.0), 1)))

    calc_by_weight = {40: w1_calc, 30: w2_calc, 20: w3_calc, 10: w4_calc}
    existing_wedges = output.get("wedge_breakdown") or []
    if len(existing_wedges) == 4:
        for w_obj in existing_wedges:
            wt_pct = int(w_obj.get("weight_percentage", 0))
            if wt_pct in calc_by_weight:
                w_obj["net_sentiment_score"] = calc_by_weight[wt_pct]

    # 3. Compute exact composite Weighted Sentiment Score = 40% W1 + 30% W2 + 20% W3 + 10% W4
    composite = round(
        0.40 * w1_calc + 0.30 * w2_calc + 0.20 * w3_calc + 0.10 * w4_calc,
        1,
    )
    output["weighted_sentiment_score"] = composite

    # 4. Derive Overall Bias consistently from composite score & cross-asset/news polarity
    if composite >= 35.0:
        output["overall_bias"] = "STRONG BUY"
    elif composite >= 10.0:
        output["overall_bias"] = "BUY"
    elif composite <= -35.0:
        output["overall_bias"] = "STRONG SELL"
    elif composite <= -10.0:
        output["overall_bias"] = "SELL"
    else:
        output["overall_bias"] = "NEUTRAL / CHOPPY"

    # 5. Calibrate Confidence Score smoothly from weighted wedge alignment & news consensus
    direction = -1 if composite < 0 else (1 if composite > 0 else 0)
    aligned_weight = 0
    for wt_pct, w_val in ((40, w1_calc), (30, w2_calc), (20, w3_calc), (10, w4_calc)):
        if direction < 0 and w_val <= -5.0:
            aligned_weight += wt_pct
        elif direction > 0 and w_val >= 5.0:
            aligned_weight += wt_pct
        elif abs(w_val) < 8.0:
            aligned_weight += int(wt_pct * 0.5)

    if aligned_weight >= 80 and abs(composite) >= 15.0:
        output["confidence_score"] = min(94, max(78, int(round(72 + abs(composite) * 0.45))))
    elif aligned_weight >= 65 and abs(composite) >= 10.0:
        output["confidence_score"] = min(77, max(65, int(round(60 + abs(composite) * 0.45))))
    else:
        output["confidence_score"] = min(64, max(45, int(round(45 + abs(composite) * 0.45))))

    return output


# ---------------------------------------------------------------------------
# Groq Automatic Fallback Engine (Triggered when Gemini hits HTTP 429 Quota)
# ---------------------------------------------------------------------------
def _normalize_groq_raw_dict(raw_text: str) -> Dict[str, Any]:
    """
    Clean and normalize LLM JSON output (stripping markdown fences if any, mapping
    shorthand enum values like 'NEUTRAL' -> 'NEUTRAL / CHOPPY' or 'BULLISH' -> 'BULLISH GOLD')
    before Pydantic schema validation.
    """
    cleaned = (raw_text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("LLM response is not a JSON object")

    raw_bias = str(parsed.get("overall_bias") or "NEUTRAL / CHOPPY").upper().strip()
    if "STRONG BUY" in raw_bias:
        parsed["overall_bias"] = "STRONG BUY"
    elif "BUY" in raw_bias:
        parsed["overall_bias"] = "BUY"
    elif "STRONG SELL" in raw_bias:
        parsed["overall_bias"] = "STRONG SELL"
    elif "SELL" in raw_bias:
        parsed["overall_bias"] = "SELL"
    else:
        parsed["overall_bias"] = "NEUTRAL / CHOPPY"

    parsed["weighted_sentiment_score"] = max(
        -100.0, min(100.0, float(parsed.get("weighted_sentiment_score") or 0.0))
    )
    parsed["confidence_score"] = max(
        0, min(100, int(round(float(parsed.get("confidence_score") or 55))))
    )

    bullets = parsed.get("why_buy_or_sell_thesis")
    if not isinstance(bullets, list):
        bullets = [str(bullets)] if bullets else []
    bullets = [str(b).strip() for b in bullets if str(b).strip()]
    while len(bullets) < 4:
        bullets.append(
            "Invalidation Trigger: A sharp reversal in US 10Y Treasury yields (^TNX) or DXY would invalidate the current intraday XAUUSD bias."
        )
    parsed["why_buy_or_sell_thesis"] = bullets[:6]

    # Normalize and guarantee all 4 Gold Impact Wedges (40%, 30%, 20%, 10%) with scores clamped to [-100, +100]
    default_wedges = [
        {
            "wedge_name": GOLD_IMPACT_WEDGES["WEDGE_1_FED_YIELD_DXY"]["wedge_name"],
            "weight_percentage": 40,
            "net_sentiment_score": -15.0,
            "summary": "Fed policy rate hold and elevated US 10Y Treasury yields (^TNX) exert opportunity-cost pressure on non-yielding Gold.",
        },
        {
            "wedge_name": GOLD_IMPACT_WEDGES["WEDGE_2_US_TIER1_ECON"]["wedge_name"],
            "weight_percentage": 30,
            "net_sentiment_score": -10.0,
            "summary": "Upcoming US Advance GDP and Core PCE inflation prints keep macro positioning cautious.",
        },
        {
            "wedge_name": GOLD_IMPACT_WEDGES["WEDGE_3_GEOPOLITICAL_SAFE_HAVEN"]["wedge_name"],
            "weight_percentage": 20,
            "net_sentiment_score": 10.0,
            "summary": "Geopolitical risk premia and safe-haven hedging provide partial downside buffer for XAUUSD.",
        },
        {
            "wedge_name": GOLD_IMPACT_WEDGES["WEDGE_4_CENTRAL_BANK_ETF_FLOWS"]["wedge_name"],
            "weight_percentage": 10,
            "net_sentiment_score": 10.0,
            "summary": "Central bank bullion demand and GLD ETF flows remain structurally supportive.",
        },
    ]
    raw_wedges = parsed.get("wedge_breakdown")
    if not isinstance(raw_wedges, list):
        raw_wedges = []

    norm_wedges: List[Dict[str, Any]] = []
    for idx, def_w in enumerate(default_wedges):
        src_w = raw_wedges[idx] if idx < len(raw_wedges) and isinstance(raw_wedges[idx], dict) else {}
        raw_w_score = src_w.get("net_sentiment_score", def_w["net_sentiment_score"])
        try:
            clamped_w_score = max(-100.0, min(100.0, float(raw_w_score)))
        except Exception:
            clamped_w_score = def_w["net_sentiment_score"]

        norm_wedges.append(
            {
                "wedge_name": str(src_w.get("wedge_name") or def_w["wedge_name"]).strip(),
                "weight_percentage": def_w["weight_percentage"],
                "net_sentiment_score": clamped_w_score,
                "summary": str(src_w.get("summary") or def_w["summary"]).strip(),
            }
        )
    parsed["wedge_breakdown"] = norm_wedges

    norm_items: List[Dict[str, Any]] = []
    for item in parsed.get("news_items") or []:
        if not isinstance(item, dict):
            continue
        gb = str(item.get("gold_bias") or "NEUTRAL").upper().strip()
        if "BULL" in gb or "BUY" in gb:
            gb_norm = "BULLISH GOLD"
        elif "BEAR" in gb or "SELL" in gb:
            gb_norm = "BEARISH GOLD"
        else:
            gb_norm = "NEUTRAL"

        norm_items.append(
            {
                "headline": str(item.get("headline") or "Gold Macro Update").strip(),
                "source_time": str(item.get("source_time") or "Live Feed | UTC").strip(),
                "gold_wedge_category": str(
                    item.get("gold_wedge_category")
                    or GOLD_IMPACT_WEDGES["WEDGE_1_FED_YIELD_DXY"]["wedge_name"]
                ).strip(),
                "impact_weight": max(1, min(10, int(round(float(item.get("impact_weight") or 6))))),
                "gold_bias": gb_norm,
                "sentiment_score": max(
                    -100, min(100, int(round(float(item.get("sentiment_score") or 0))))
                ),
                "gold_transmission_mechanism": str(
                    item.get("gold_transmission_mechanism")
                    or "Transmits into XAUUSD via US real yields (^TNX) and DXY positioning."
                ).strip(),
            }
        )
    parsed["news_items"] = norm_items
    return parsed


def _analyze_with_groq(
    batched_prompt_payload: Dict[str, Any],
    max_retries_per_model: int = 2,
    base_backoff_seconds: float = 2.0,
) -> Optional[Dict[str, Any]]:
    """
    Automatic fallback to Groq API (`https://api.groq.com/openai/v1/chat/completions`)
    when Google Gemini Free Tier hits HTTP 429 quota limits.
    """
    if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
        return None

    # Build a token-efficient payload to stay well under Groq Free-Tier 8,000 TPM limit
    raw_tickers = batched_prompt_payload.get("live_gold_and_macro_tickers") or {}
    compact_tickers = {
        k: {
            "symbol": v.get("display_symbol") or v.get("symbol"),
            "price": v.get("price"),
            "change_pct": v.get("change_pct"),
            "day_high": v.get("day_high"),
            "day_low": v.get("day_low"),
            "role": v.get("gold_relationship"),
        }
        for k, v in raw_tickers.items()
        if isinstance(v, dict)
    }

    compact_headlines = [
        {
            "headline": h.get("headline"),
            "source_time": h.get("source_time"),
            "wedge": h.get("pre_classified_wedge"),
            "weight": h.get("suggested_impact_weight"),
        }
        for h in (batched_prompt_payload.get("intraday_breaking_headlines") or [])
    ]

    compact_payload_str = json.dumps(
        {"tickers": compact_tickers, "headlines": compact_headlines},
        separators=(",", ":"),
    )

    groq_system_prompt = (
        f"{GOLD_SYSTEM_INSTRUCTION}\n\n"
        "OUTPUT STRICT VALID JSON ONLY matching this exact structure:\n"
        "{\n"
        '  "overall_bias": "STRONG BUY" | "BUY" | "NEUTRAL / CHOPPY" | "SELL" | "STRONG SELL",\n'
        '  "weighted_sentiment_score": float (-100.0 to 100.0),\n'
        '  "confidence_score": int (0 to 100),\n'
        '  "macro_condition_summary": "string",\n'
        '  "why_buy_or_sell_thesis": ["bullet 1", "bullet 2", "bullet 3", "bullet 4", "invalidation risk bullet"],\n'
        '  "wedge_breakdown": [\n'
        '    {"wedge_name": "Wedge 1: Fed Policy, US 10Y Real Yields (^TNX) & DXY", "weight_percentage": 40, "net_sentiment_score": float, "summary": "string"},\n'
        '    {"wedge_name": "Wedge 2: US Tier-1 Macro Data (CPI, PCE, NFP, GDP)", "weight_percentage": 30, "net_sentiment_score": float, "summary": "string"},\n'
        '    {"wedge_name": "Wedge 3: Geopolitical Risk, Tariffs & Safe-Haven Flows (^VIX)", "weight_percentage": 20, "net_sentiment_score": float, "summary": "string"},\n'
        '    {"wedge_name": "Wedge 4: Central Bank Buying, COMEX & GLD ETF Flows", "weight_percentage": 10, "net_sentiment_score": float, "summary": "string"}\n'
        "  ],\n"
        '  "news_items": [\n'
        '    {"headline": "exact headline", "source_time": "source | time", "gold_wedge_category": "string", "impact_weight": int (1-10), "gold_bias": "BULLISH GOLD" | "BEARISH GOLD" | "NEUTRAL", "sentiment_score": int (-100 to 100), "gold_transmission_mechanism": "1 concise sentence"}\n'
        "  ]\n"
        "}\n"
        "Include all 18 headlines in `news_items` and keep each `gold_transmission_mechanism` to 1 short sentence."
    )

    for groq_model in GROQ_MODEL_CHAIN:
        for attempt in range(max_retries_per_model):
            try:
                logger.info(
                    "Sending single batched Gold analysis request to Groq fallback model '%s' (attempt %d/%d)...",
                    groq_model,
                    attempt + 1,
                    max_retries_per_model,
                )
                reasoning_mode = "none" if "qwen" in groq_model.lower() else "low"
                req_body: Dict[str, Any] = {
                    "model": groq_model,
                    "reasoning_effort": reasoning_mode,
                    "messages": [
                        {"role": "system", "content": groq_system_prompt},
                        {
                            "role": "user",
                            "content": f"Return JSON GoldAnalysisReport for:\n{compact_payload_str}",
                        },
                    ],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.15,
                    "max_tokens": 3200,
                }
                resp = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {GROQ_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json=req_body,
                    timeout=45,
                )

                if resp.status_code in (429, 503) and attempt < max_retries_per_model - 1:
                    sleep_sec = base_backoff_seconds * (2 ** attempt)
                    logger.warning(
                        "Groq HTTP %d on '%s' (attempt %d/%d). Backing off %.1fs...",
                        resp.status_code,
                        groq_model,
                        attempt + 1,
                        max_retries_per_model,
                        sleep_sec,
                    )
                    time.sleep(sleep_sec)
                    continue

                resp.raise_for_status()
                data = resp.json()
                raw_content = (
                    (data.get("choices") or [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                normalized_dict = _normalize_groq_raw_dict(raw_content)
                validated = GoldAnalysisReport.model_validate(normalized_dict)
                output = validated.model_dump()

                # Ensure all 18 breaking headlines are present even if the LLM abbreviated the list
                existing_titles = [
                    (item.get("headline") or "").strip().lower()
                    for item in output.get("news_items", [])
                ]
                for raw_h in (batched_prompt_payload.get("intraday_breaking_headlines") or []):
                    h_text = (raw_h.get("headline") or "").strip()
                    h_low = h_text.lower()
                    already_present = any(
                        h_low == et or (len(h_low) >= 30 and h_low[:30] == et[:30])
                        for et in existing_titles
                    )
                    if h_text and not already_present:
                        fallback_item = _score_headline_deterministically(
                            {
                                "headline": h_text,
                                "summary": raw_h.get("summary", ""),
                                "source_time": raw_h.get("source_time", ""),
                                "gold_wedge": {
                                    "wedge_name": raw_h.get("pre_classified_wedge")
                                    or GOLD_IMPACT_WEDGES["WEDGE_4_CENTRAL_BANK_ETF_FLOWS"]["wedge_name"],
                                    "impact_weight": raw_h.get("suggested_impact_weight") or 6,
                                },
                            }
                        )
                        output.setdefault("news_items", []).append(fallback_item.model_dump())
                        existing_titles.append(h_low)

                output["analysis_engine"] = f"groq_api ({groq_model})"
                logger.info("Groq fallback model '%s' succeeded!", groq_model)
                return output

            except Exception as exc:
                logger.warning(
                    "Groq fallback model '%s' failed (attempt %d/%d): %s",
                    groq_model,
                    attempt + 1,
                    max_retries_per_model,
                    exc,
                )
                break

    return None


# ---------------------------------------------------------------------------
# Single-Request Batched Gemini Free-Tier Analyzer with Exponential Backoff
# ---------------------------------------------------------------------------
def analyze_gold_market(
    payload: Dict[str, Any],
    model_name: str = GEMINI_MODEL,
    max_retries_per_model: int = 3,
    base_backoff_seconds: float = 2.0,
    allow_fallback_on_missing_key: bool = True,
) -> Dict[str, Any]:
    """
    Send the entire live Gold/Macro ticker snapshot (`GC=F`, `DX-Y.NYB`, `^TNX`, `^VIX`, `GLD`)
    AND all 18 breaking headlines in ONE single batched Gemini API request using
    `response_mime_type="application/json"` and `response_schema=GoldAnalysisReport`.

    Includes exponential backoff retry logic for HTTP 429 (`RESOURCE_EXHAUSTED`),
    automatic fallback across `GEMINI_MODEL_CHAIN`, and automatic failover to
    Groq API (`GROQ_MODEL_CHAIN`) when Gemini hits its daily Free-Tier quota.
    """
    # Compact, clean single-request batched payload containing all 5 tickers + all 18 headlines
    batched_prompt_payload = {
        "generated_at_utc": payload.get("generated_at_utc"),
        "primary_asset": payload.get("primary_asset"),
        "gold_impact_wedges_weights": {
            k: {
                "wedge_name": v["wedge_name"],
                "weight_percentage": v["weight_percentage"],
                "weight_range": v["weight_range"],
            }
            for k, v in GOLD_IMPACT_WEDGES.items()
        },
        "live_gold_and_macro_tickers": (payload.get("gold_macro_snapshot") or {}).get("tickers"),
        "cross_asset_gold_alignment": (payload.get("gold_macro_snapshot") or {}).get(
            "cross_asset_gold_alignment"
        ),
        "intraday_breaking_headlines": [
            {
                "rank": item.get("rank"),
                "headline": item.get("headline"),
                "summary": item.get("summary"),
                "source_time": item.get("source_time"),
                "pre_classified_wedge": (item.get("gold_wedge") or {}).get("wedge_name"),
                "suggested_impact_weight": (item.get("gold_wedge") or {}).get("impact_weight"),
            }
            for item in (payload.get("intraday_breaking_news") or [])
        ],
    }

    has_gemini_key = bool(GEMINI_API_KEY and GEMINI_API_KEY != "your_gemini_api_key_here")
    has_groq_key = bool(GROQ_API_KEY and GROQ_API_KEY != "your_groq_api_key_here")

    if not has_gemini_key and not has_groq_key:
        if allow_fallback_on_missing_key:
            logger.warning(
                "Neither GEMINI_API_KEY nor GROQ_API_KEY in `.env` is configured. "
                "Running deterministic Gold correlation fallback engine to validate schema."
            )
            result = build_deterministic_gold_analysis(payload)
            result["analysis_engine"] = "deterministic_fallback (set GEMINI_API_KEY or GROQ_API_KEY in .env)"
            return result
        raise ValueError(
            "GEMINI_API_KEY and GROQ_API_KEY are missing in `.env`. "
            "Please configure a valid API key in `.env`."
        )

    global _GEMINI_DAILY_QUOTA_EXHAUSTED
    last_error: Optional[Exception] = None

    if HAS_GENAI and has_gemini_key and not _GEMINI_DAILY_QUOTA_EXHAUSTED:
        client = genai.Client(api_key=GEMINI_API_KEY)
        models_to_try = list(dict.fromkeys([model_name] + GEMINI_MODEL_CHAIN))

        for candidate_model in models_to_try:
            for attempt in range(max_retries_per_model):
                try:
                    logger.info(
                        "Sending single batched Gold analysis request to Gemini model '%s' (attempt %d/%d)...",
                        candidate_model,
                        attempt + 1,
                        max_retries_per_model,
                    )
                    response = client.models.generate_content(
                        model=candidate_model,
                        contents=json.dumps(batched_prompt_payload, indent=2),
                        config=types.GenerateContentConfig(
                            system_instruction=GOLD_SYSTEM_INSTRUCTION,
                            response_mime_type="application/json",
                            response_schema=GoldAnalysisReport,
                            temperature=0.15,
                        ),
                    )

                    validated = GoldAnalysisReport.model_validate_json(response.text)
                    output = _reconcile_and_validate_analysis(validated.model_dump(), payload)
                    output["analysis_engine"] = f"gemini_api ({candidate_model})"
                    return output

                except Exception as exc:
                    last_error = exc
                    err_str = str(exc)
                    is_daily_quota_exhausted = (
                        "GenerateRequestsPerDay" in err_str
                        or "per day limit exceeded" in err_str.lower()
                    )
                    if is_daily_quota_exhausted:
                        _GEMINI_DAILY_QUOTA_EXHAUSTED = True
                        logger.warning(
                            "Gemini daily Free-Tier quota (20 req/day) exhausted on '%s'. "
                            "Switching immediately to Groq fallback for this session...",
                            candidate_model,
                        )
                        break

                    is_rate_limit = (
                        "429" in err_str
                        or "RESOURCE_EXHAUSTED" in err_str.upper()
                        or "QUOTA" in err_str.upper()
                        or "503" in err_str
                    )
                    if is_rate_limit and attempt < max_retries_per_model - 1:
                        sleep_sec = base_backoff_seconds * (2 ** attempt)
                        logger.warning(
                            "HTTP 429 / rate-limit encountered on '%s' (attempt %d/%d). "
                            "Backing off for %.1fs before retry...",
                            candidate_model,
                            attempt + 1,
                            max_retries_per_model,
                            sleep_sec,
                        )
                        time.sleep(sleep_sec)
                        continue

                    logger.warning(
                        "Model '%s' failed with error: %s. Trying next fallback model if available...",
                        candidate_model,
                        exc,
                    )
                    break

    # Automatic failover to Groq API when Gemini quota is exhausted
    if has_groq_key:
        logger.info("Failing over to Groq API automatic fallback engine...")
        groq_result = _analyze_with_groq(
            batched_prompt_payload=batched_prompt_payload,
            max_retries_per_model=2,
            base_backoff_seconds=base_backoff_seconds,
        )
        if groq_result is not None:
            engine_label = groq_result.get("analysis_engine")
            reconciled_groq = _reconcile_and_validate_analysis(groq_result, payload)
            reconciled_groq["analysis_engine"] = engine_label
            return reconciled_groq

    if allow_fallback_on_missing_key:
        logger.warning(
            "All Gemini & Groq models failed (%s). Falling back to deterministic Gold correlation engine.",
            last_error,
        )
        result = build_deterministic_gold_analysis(payload)
        result["analysis_engine"] = "Quantitative Gold Engine (Auto-Fallback)"
        return result

    raise RuntimeError(f"Gold analysis failed across all Gemini and Groq models: {last_error}") from last_error


# Backwards-compatible alias for main.py
analyze_market_payload = analyze_gold_market


if __name__ == "__main__":
    import sys
    from data_fetcher import fetch_market_intelligence_payload

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    print("=" * 95)
    print(" GOLD (XAUUSD / COMEX GC=F) CORRELATION & WEDGE ANALYZER — VERIFICATION TEST")
    print("=" * 95)

    live_payload = fetch_market_intelligence_payload()
    analysis_result = analyze_gold_market(live_payload)

    # Strictly re-validate against GoldAnalysisReport Pydantic schema
    validated_schema = GoldAnalysisReport.model_validate(
        {k: v for k, v in analysis_result.items() if k != "analysis_engine"}
    )

    print(f"\n[1] GOLD ANALYSIS SUMMARY (Engine: {analysis_result.get('analysis_engine')})")
    print(f"    • Overall XAUUSD / GC Bias : {validated_schema.overall_bias}")
    print(f"    • Weighted Sentiment Score : {validated_schema.weighted_sentiment_score:+.1f} / 100.0")
    print(f"    • Confidence Score         : {validated_schema.confidence_score}%")
    print(f"    • Macro Condition Summary  : {validated_schema.macro_condition_summary}")

    print("\n[2] 4 GOLD IMPACT WEDGES BREAKDOWN")
    for w in validated_schema.wedge_breakdown:
        print(
            f"    • {w.wedge_name:<70} | Wt: {w.weight_percentage:>2}% | "
            f"Score: {w.net_sentiment_score:+6.1f}"
        )
        print(f"      -> {w.summary}")

    print("\n[3] INSTITUTIONAL BUY / SELL THESIS & INVALIDATION RISK")
    for idx, bullet in enumerate(validated_schema.why_buy_or_sell_thesis, start=1):
        print(f"    {idx}. {bullet}")

    print(f"\n[4] ANALYZED BREAKING HEADLINES ({len(validated_schema.news_items)} items)")
    print("-" * 95)
    for idx, item in enumerate(validated_schema.news_items[:18], start=1):
        print(
            f" {idx:>2}. [{item.gold_bias:<12} | Score:{item.sentiment_score:+4d} | Wt:{item.impact_weight:>2}/10] "
            f"{item.headline}"
        )
        print(f"     Wedge: {item.gold_wedge_category} | Source: {item.source_time}")
        print(f"     Mechanism: {item.gold_transmission_mechanism}")
    print("=" * 95)
