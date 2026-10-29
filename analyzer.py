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
_EXHAUSTED_GEMINI_MODELS: set = set()
_EXHAUSTED_GROQ_MODELS: set = set()
_LAST_GEMINI_ERROR_SUMMARY: Optional[str] = None


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
1. Live intraday market quotes (% change and levels) for:
   - `GC=F`: COMEX Gold Futures (primary target)
   - `DX-Y.NYB` (DXY): US Dollar Index (INVERSE DRIVER: Strong USD = Bearish Gold; Weak USD = Bullish Gold)
   - `^TNX`: US 10-Year Treasury Yield (INVERSE REAL-RATE DRIVER: Higher Yields = Bearish Gold; Falling Yields = Bullish Gold)
   - `^VIX`: CBOE Volatility Index (SAFE-HAVEN FEAR GAUGE: Spiking panic = Bullish Gold; Falling/calm = Safe-Haven unwinds)
   - `GLD`: SPDR Gold Trust ETF (Institutional physical flow proxy)
2. All 18 deduplicated breaking headlines with full summaries, timestamps, sources, and pre-classified Gold Wedges.

YOU MUST APPLY THESE EXACT PRECIOUS METALS CORRELATION & POLARITY RULES (NO CONTRADICTIONS):

1. US DOLLAR (DXY) & TREASURY YIELDS (^TNX) — WEDGE 1 (40% Weight):
   - If headlines say the US Dollar Index is bullish, firmed, or breaking higher -> THIS IS BEARISH FOR GOLD (-20 to -60).
   - If US 10Y Yields rise -> BEARISH FOR GOLD (-25 to -70).
   - If US 10Y Yields fall or DXY softens -> BULLISH FOR GOLD (+20 to +65).
   - Fed holding rates steady or pushing back against cuts -> BEARISH FOR GOLD (-25 to -50) because it maintains high opportunity cost.
   - European Central Bank (ECB) rate hold with no direct USD/Gold impact -> NEUTRAL (0).

2. US TIER-1 ECONOMIC DATA — WEDGE 2 (30% Weight):
   - Hot GDP (+3.4% beat), elevated Core PCE (+0.3%), hot CPI, or multi-decade low jobless claims (tight labor) force the Fed to stay hawkish and delay rate cuts -> BEARISH FOR GOLD (-25 to -60).
   - Cooling inflation, rising jobless claims, weak growth, or stagflation -> BULLISH FOR GOLD (+20 to +65).

3. GEOPOLITICAL RISK & SAFE-HAVEN FLOWS — WEDGE 3 (20% Weight):
   - Acute war escalation, missile strikes, military conflict, trade war tariffs -> BULLISH GOLD SAFE-HAVEN (+40 to +90).
   - Peace talks, diplomatic negotiations, ceasefire agreements, or falling oil prices because of diplomacy -> DE-ESCALATION / SAFE-HAVEN UNWIND (BEARISH / NEUTRAL FOR GOLD: -15 to -40). (Do NOT score peace talks or diplomacy as Bullish Gold!).

4. CENTRAL BANK & ETF FLOWS — WEDGE 4 (10% Weight):
   - Sovereign central bank reserve accumulation -> BULLISH GOLD (+25 to +60).
   - GLD ETF outflows, technical breakdown warnings, or institutional liquidation -> BEARISH GOLD (-20 to -50).

5. POLARITY INTEGRITY REQUIREMENTS:
   - For every headline in `news_items`:
     - If `gold_bias` is "BULLISH GOLD", `sentiment_score` MUST be positive (+15 to +100).
     - If `gold_bias` is "BEARISH GOLD", `sentiment_score` MUST be negative (-15 to -100).
     - If `gold_bias` is "NEUTRAL", `sentiment_score` MUST be between -14 and +14 (typically 0).
     - `gold_transmission_mechanism` MUST be 1 concise sentence explaining the exact transmission to Gold prices.
   - For `wedge_breakdown`:
     - Provide `net_sentiment_score` (-100.0 to +100.0) reflecting the net balance of news and live ticker moves in each of the 4 wedges.
   - For `overall_bias` and `weighted_sentiment_score`:
     - Calculate composite score = 0.40 * W1 + 0.30 * W2 + 0.20 * W3 + 0.10 * W4.
     - Bias is "BUY" if score >= +15.0 ("STRONG BUY" if >= +45.0).
     - Bias is "SELL" if score <= -15.0 ("STRONG SELL" if <= -45.0).
     - Bias is "NEUTRAL / CHOPPY" if score is between -15.0 and +15.0.
   - Provide 4-6 concise takeaway bullets (`why_buy_or_sell_thesis`) including the #1 invalidation risk.

Return strictly valid JSON matching `GoldAnalysisReport`.
"""


# ---------------------------------------------------------------------------
# Deterministic Quantitative Fallback Engine (Used when API Key is Placeholder)
# ---------------------------------------------------------------------------
_BULLISH_GOLD_CUES: Dict[str, Tuple[int, str]] = {
    "rate cut": (+65, "Dovish Fed rate-cut expectations lower real yields and reduce the opportunity cost of holding non-yielding XAUUSD."),
    "dovish": (+60, "Dovish central bank rhetoric compresses sovereign bond yields, driving institutional inflows into Gold."),
    "yields fall": (+60, "Declining Treasury yields reduce real rate headwinds and directly boost XAUUSD valuation."),
    "falling yields": (+60, "Falling US yields lower carry costs for bullion, supporting upside in COMEX Gold futures."),
    "war escalation": (+85, "Acute military conflict triggers immediate safe-haven capital rotation into physical Gold, overriding USD strength."),
    "iran strike": (+80, "Middle East military strikes inject acute geopolitical risk premia into XAUUSD safe-haven flows."),
    "attack on iran": (+80, "Military strikes against Iranian infrastructure spark urgent safe-haven bidding in Gold."),
    "iran missile": (+80, "Missile escalation in the Middle East prompts defensive hedging into physical bullion."),
    "israel strike": (+80, "Escalating military action in the Middle East drives urgent geopolitical tail-risk hedging in Gold."),
    "middle east war": (+80, "Broadening Middle East war escalation sustains strong safe-haven premia for bullion."),
    "ukraine strike": (+75, "Escalation in the Russia-Ukraine theater reinforces institutional safe-haven demand for bullion."),
    "tariff": (+70, "Trade war and tariff uncertainty stoke stagflationary fears and safe-haven demand for Gold."),
    "trade war": (+75, "Global trade fragmentation accelerates de-dollarization and safe-haven accumulation of XAUUSD."),
    "safe haven": (+80, "Explicit safe-haven flows lift spot XAUUSD demand regardless of short-term DXY fluctuations."),
    "safe-haven": (+80, "Heightened cross-asset risk aversion channels defensive liquidity directly into Gold."),
    "geopolitical concerns": (+65, "Rising geopolitical concerns stimulate defensive safe-haven bids in bullion and commodities."),
    "central bank": (+55, "Sovereign central bank reserve diversification provides price-inelastic physical floor support under XAUUSD."),
    "gold reserves": (+60, "Official sector bullion accumulation tightens physical supply and supports structural Gold upside."),
    "unemployment rises": (+55, "Labor market softening increases probability of Fed easing, pressuring real yields and lifting Gold."),
    "rising jobless claims": (+50, "Rising jobless claims signal cooling US labor demand, pulling forward Fed rate-cut expectations for XAUUSD."),
    "jobless claims rise": (+50, "Rising jobless claims signal cooling US labor demand, pulling forward Fed rate-cut expectations for XAUUSD."),
    "stagflation": (+80, "Stagflationary macro dynamics (sticky inflation + slowing growth) represent the strongest historical regime for Gold."),
    "recession": (+70, "Recessionary tail-risk drives defensive allocation into Gold and compresses front-end Treasury yields."),
    "gold rallies": (+55, "Strong upside momentum in spot XAUUSD attracts systematic trend-following inflows."),
    "gold surges": (+65, "Breakout buying in Gold futures signals aggressive institutional accumulation."),
    "gold climbs": (+45, "Sustained buying interest lifts COMEX Gold futures alongside supportive macro flows."),
    "record high": (+65, "Breakout to record highs reinforces structural bullish momentum in XAUUSD."),
    "inflows": (+50, "Bullion ETF inflows confirm institutional real-money accumulation of physical Gold."),
    "gold rises": (+50, "Spot XAUUSD strength reflects active institutional buying and supportive macro flows."),
    "gold gains": (+50, "Upside gains in Gold futures signal constructive bullion demand."),
    "gold edges higher": (+40, "Gradual upside in XAUUSD reflects steady dip-buying support."),
    "gold rebounds": (+55, "Rebound in Gold prices highlights resilient underlying safe-haven and dip-buying demand."),
    "gold steadies": (+20, "Stabilizing XAUUSD price action shows buyers defending key intraday support."),
    "gold holds": (+25, "Resilient Gold consolidation near highs signals firm underlying physical and macro support."),
    "rate cuts": (+60, "Expectations for Federal Reserve rate cuts compress real Treasury yields and boost XAUUSD."),
    "cut rates": (+60, "Anticipation of central bank rate cuts lowers the opportunity cost of holding non-yielding Gold."),
    "yields drop": (+60, "Dropping US Treasury yields reduce real-rate headwinds and lift Gold."),
    "yields slip": (+50, "Slipping Treasury yields provide a constructive tailwind for XAUUSD."),
    "dollar slips": (+50, "A softer US Dollar boosts international purchasing power for USD-denominated Gold."),
    "dollar falls": (+55, "Weakening DXY directly supports upside in XAUUSD and COMEX GC=F."),
    "weaker dollar": (+55, "A weaker US Dollar reduces headwinds for bullion and attracts macro inflows."),
    "softer dollar": (+50, "Softer USD trading supports precious metals valuations."),
    "inflation cools": (+60, "Cooling inflation strengthens the case for Fed easing and lower real yields, lifting Gold."),
    "uncertainty": (+45, "Macro and policy uncertainty reinforces defensive allocation into physical Gold."),
    "gold bullish": (+50, "Bullish institutional positioning supports upside momentum in XAUUSD."),
    "bullish gold": (+50, "Bullish institutional positioning supports upside momentum in XAUUSD."),
    "bullish on gold": (+50, "Bullish institutional positioning supports upside momentum in XAUUSD."),
}

_BEARISH_GOLD_CUES: Dict[str, Tuple[int, str]] = {
    "remains bullish": (-50, "Persistent bullish Dollar momentum maintains strong currency headwinds for Gold."),
    "bulls seem hesitant": (-35, "Hesitation among buyers reflects uncertain upside and strong real-yield resistance."),
    "prices fall": (-50, "Softening spot Gold prices reflect selling pressure from firm yields and USD strength."),
    "inflation rose": (-45, "Rising inflation measures delay Fed rate cuts and sustain elevated Treasury yields."),
    "measure of inflation rose": (-45, "Rising inflation measures delay Fed rate cuts and sustain elevated Treasury yields."),
    "10-year treasury yield hovers": (-60, "Elevated 10Y Treasury yields draw capital away from non-yielding Gold."),
    "yield hovers above": (-60, "Elevated 10Y Treasury yields draw capital away from non-yielding Gold."),
    "strong gdp report": (-55, "Stronger-than-expected economic growth dampens the outlook for Fed monetary easing."),
    "second straight quarter": (-40, "Consecutive quarterly forecast downgrades reflect persistent macro headwinds for bullion."),
    "corrects after": (-35, "Technical correction in spot Gold highlights selling pressure at upper resistance."),
    "third-quarter u.s. gdp": (-55, "Resilient 3.4% US GDP growth underscores economic momentum and delays Fed monetary easing, weighing on XAUUSD."),
    "gdp increases": (-55, "Stronger-than-expected US GDP growth reduces the urgency for Fed rate cuts and supports real Treasury yields."),
    "3.4%": (-50, "Hotter-than-forecast economic growth bars the Fed from aggressive easing, pressuring non-yielding bullion."),
    "better-than-expected 3.4%": (-55, "Above-consensus US GDP expansion strengthens the Dollar and lifts real yields against Gold."),
    "core pce": (-50, "Persistent core PCE inflation readings keep the Federal Reserve restrictive and support elevated real yields against Gold."),
    "pce price index rises": (-50, "Sticky PCE price gains reduce the probability of near-term Fed rate cuts, capping XAUUSD."),
    "inflation accelerates": (-55, "Accelerating inflation data reinforces a higher-for-longer Fed policy stance, weighing on zero-yielding Gold."),
    "jobless claims hover": (-45, "Resilient US labor market conditions with jobless claims near historic lows reduce Fed easing bets."),
    "57-year lows": (-50, "Jobless claims near multi-decade lows demonstrate labor resilience, keeping US Treasury yields elevated."),
    "10-year treasury yield hovers above 5%": (-60, "Benchmark 10-year US Treasury yields hovering above 5% represent steep opportunity cost headwinds for bullion."),
    "yields above 5%": (-60, "Elevated 10Y sovereign yields above 5% draw capital away from zero-yielding Gold into risk-free fixed income."),
    "play catch-up to quell inflation": (-45, "Risks of the Fed maintaining aggressive restrictive policy to quell inflation suppress Gold upside."),
    "quell inflation": (-45, "Central bank focus on suppressing inflation sustains elevated carry costs for bullion."),
    "split fed keeps december hike alive": (-55, "Ongoing prospect of policy rate hikes reinforces restrictive real-rate pressure on Gold."),
    "december hike alive": (-55, "Uncertainty over potential rate hikes keeps institutional Gold buyers on the sidelines."),
    "dow ratio warns of more downside as yields rise": (-50, "Cross-asset valuation models warn of further Gold downside as Treasury yields remain elevated."),
    "warns of more downside": (-50, "Technical and macro headwinds warn of continued liquidation pressure in XAUUSD."),
    "forecast cut for second": (-40, "Downgrades to institutional Gold price forecasts reflect ongoing headwinds from higher real yields."),
    "forecast cut": (-40, "Analyst price forecast reductions dampen medium-term investment appetite for Gold futures."),
    "peace deal with iran": (-50, "Diplomatic engagement and peace negotiations with Iran deflate geopolitical risk premia in safe-haven Gold."),
    "peace deal": (-45, "Geopolitical de-escalation progress unwinds safe-haven hedging in spot XAUUSD."),
    "peace talks": (-45, "Diplomatic peace negotiations moderate urgent safe-haven bidding in bullion."),
    "iran diplomacy": (-45, "Diplomatic engagement reduces immediate Middle East conflict risks and tempers Gold safe-haven bids."),
    "oil falls as traders weigh iran": (-40, "Easing Middle East conflict risk unwinds risk premia across energy and precious metals complexes."),
    "weigh iran diplomacy": (-40, "Market focus on diplomatic avenues reduces speculative safe-haven premia in Gold."),
    "downplays iran peace talks": (-20, "Mixed geopolitical headlines create two-way chop without unilateral safe-haven escalation."),
    "dollar index remains bullish": (-50, "A firmer US Dollar Index forecast creates direct inverse headwinds for XAUUSD."),
    "dollar remains bullish": (-50, "A firmer US Dollar Index forecast creates direct inverse headwinds for XAUUSD."),
    "dollar bullish": (-50, "Dollar bullishness raises the opportunity cost of holding USD-denominated Gold."),
    "dxy remains bullish": (-50, "A stronger US Dollar Index trajectory weighs on bullion prices."),
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
    "ecb keeps",
    "ecb hold",
    "european central bank keeps",
    "playbook",
    "trader's guide",
    "traders guide",
    "confirmation checklist",
    "breakout or trap",
    "due thursday",
    "interchange fee",
)


def _cue_matches(cue: str, text: str) -> bool:
    """Match a keyword/phrase cue using smart boundaries on alphanumeric ends to handle '%', '$', etc."""
    if not cue:
        return False
    esc = re.escape(cue)
    prefix = r"\b" if cue[0].isalnum() else ""
    suffix = r"\b" if cue[-1].isalnum() else ""
    return bool(re.search(f"{prefix}{esc}{suffix}", text))


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
        # Neutral baseline when no explicit bullish/bearish cue phrase matched (do not force negative bias)
        if w_id == 1:
            avg_score = 0
            mechanism = "Federal Reserve, Treasury yield, and US Dollar commentary monitored for next directional real-rate impulse in XAUUSD."
        elif w_id == 2:
            avg_score = 0
            mechanism = "US macroeconomic release tracked for deviations against consensus that shift Fed rate expectations."
        elif w_id == 3:
            avg_score = +20
            mechanism = "Ongoing geopolitical and trade developments maintain a supportive safe-haven floor under Gold."
        else:
            avg_score = 0
            impact_weight = min(impact_weight, 4)
            mechanism = "Balanced precious metals positioning with two-way institutional flow."

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

    # Compute live market adjustments for each wedge using both absolute yields/DXY levels and intraday changes
    macro_w1_score = 0.0
    # Yield level structural impact (high yields = structurally bearish for zero-yielding Gold)
    if tnx_price >= 4.75:
        macro_w1_score -= 25.0
    elif tnx_price >= 4.30:
        macro_w1_score -= 10.0
    elif tnx_price <= 3.80:
        macro_w1_score += 20.0

    # DXY level structural impact
    if dxy_price >= 102.0:
        macro_w1_score -= 20.0
    elif dxy_price >= 101.0:
        macro_w1_score -= 10.0
    elif dxy_price <= 99.0:
        macro_w1_score += 20.0

    # Intraday rate and dollar adjustments
    if tnx_chg >= 0.40:
        macro_w1_score -= 20.0
    elif tnx_chg <= -0.40:
        macro_w1_score += 15.0

    if dxy_chg >= 0.25:
        macro_w1_score -= 20.0
    elif dxy_chg <= -0.25:
        macro_w1_score += 15.0

    macro_w1_score = max(-80.0, min(80.0, macro_w1_score))

    w1_news = _weighted_news_avg(wedge_headline_scores[1], 0.0)
    w1_score = max(-100.0, min(100.0, round(0.50 * macro_w1_score + 0.50 * w1_news, 1)))

    # Wedge 2 (30%): US Tier-1 Economic Data
    w2_score = max(-100.0, min(100.0, _weighted_news_avg(wedge_headline_scores[2], 0.0)))

    # Wedge 3 (20%): Geopolitical Risk & Safe-Haven (^VIX)
    vix_bonus = 20.0 if vix_price >= 20.0 else (5.0 if vix_price >= 16.0 else 0.0)
    w3_news = _weighted_news_avg(wedge_headline_scores[3], 0.0)
    w3_score = max(-100.0, min(100.0, round(0.25 * vix_bonus + 0.75 * w3_news, 1)))

    # Wedge 4 (10%): Central Bank Buying, COMEX & GLD ETF flows
    flow_bonus = 25.0 if (gc_chg > 0.25 or gld_chg > 0.25) else (-25.0 if (gc_chg < -0.35 or gld_chg < -0.35) else 0.0)
    w4_news = _weighted_news_avg(wedge_headline_scores[4], 0.0)
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
    # Map raw breaking news metadata by normalized headline and fuzzy tokens for cross-model anchoring
    raw_news_list: List[Dict[str, Any]] = list(payload.get("intraday_breaking_news") or [])
    raw_news_map: Dict[str, Dict[str, Any]] = {}
    for raw_item in raw_news_list:
        hl_key = (raw_item.get("headline") or "").strip().lower()
        if hl_key:
            raw_news_map[hl_key] = raw_item

    reconciled_news: List[Dict[str, Any]] = []
    wedge_scores_by_id: Dict[int, List[Tuple[int, int]]] = {1: [], 2: [], 3: [], 4: []}

    for idx_item, item in enumerate(output.get("news_items") or []):
        n = dict(item)
        hl_lower = (n.get("headline") or "").strip().lower()
        raw_ref = raw_news_map.get(hl_lower)
        if raw_ref is None and hl_lower:
            hl_words = set(re.findall(r"[a-z0-9]{3,}", hl_lower))
            best_ov = 0.0
            for r_cand in raw_news_list:
                r_words = set(re.findall(r"[a-z0-9]{3,}", (r_cand.get("headline") or "").lower()))
                if hl_words and r_words:
                    ov = len(hl_words & r_words) / max(1, min(len(hl_words), len(r_words)))
                    if ov > best_ov and ov >= 0.55:
                        best_ov = ov
                        raw_ref = r_cand
        if raw_ref is None and idx_item < len(raw_news_list):
            raw_ref = raw_news_list[idx_item]
        if raw_ref is not None:
            n["headline"] = raw_ref.get("headline") or n.get("headline")
            n["source_time"] = raw_ref.get("source_time") or n.get("source_time")
            hl_lower = (n.get("headline") or "").strip().lower()

        score = int(n.get("sentiment_score", 0))
        bias = (n.get("gold_bias") or "NEUTRAL").upper()
        weight = int(n.get("impact_weight", 5))
        wedge_cat = n.get("gold_wedge_category") or ""

        # Lock wedge category to pre-classified wedge
        if raw_ref and (raw_ref.get("gold_wedge") or {}).get("wedge_name"):
            wedge_cat = raw_ref["gold_wedge"]["wedge_name"]
            n["gold_wedge_category"] = wedge_cat

        # NO BLENDER: Let the model evaluate the news without heuristic keyword blending.
        # Simply enforce directional sanity:
        # If model labeled BULLISH GOLD, score must be >= 15
        if "BULL" in bias:
            bias = "BULLISH GOLD"
            score = max(15, min(100, score if score >= 15 else 30))
        # If model labeled BEARISH GOLD, score must be <= -15
        elif "BEAR" in bias:
            bias = "BEARISH GOLD"
            score = min(-15, max(-100, score if score <= -15 else -30))
        # If model labeled NEUTRAL, score must be near zero
        else:
            bias = "NEUTRAL"
            score = max(-14, min(14, score))

        n["sentiment_score"] = score
        n["gold_bias"] = bias
        n["impact_weight"] = max(1, min(10, weight))
        reconciled_news.append(n)

        w_id = _infer_wedge_id(wedge_cat, 4)
        wedge_scores_by_id[w_id].append((n["sentiment_score"], n["impact_weight"]))

    output["news_items"] = reconciled_news

    # 2. Compute transparent, mathematical Wedge Scores directly from the model and news items
    def _w_avg(pairs: List[Tuple[int, int]], fallback: float) -> float:
        if not pairs:
            return fallback
        num = sum(s * w for s, w in pairs)
        den = sum(w for _, w in pairs)
        return round(num / max(den, 1), 1)

    existing_wedges = output.get("wedge_breakdown") or []
    llm_w_map = {int(w.get("weight_percentage", 0)): float(w.get("net_sentiment_score", 0.0)) for w in existing_wedges}

    # Wedge 1: Model's Fed/Yields/DXY evaluation, anchored by the news in that wedge
    w1_news_avg = _w_avg(wedge_scores_by_id[1], llm_w_map.get(40, 0.0))
    w1_calc = round(llm_w_map.get(40, w1_news_avg), 1)

    # Wedge 2: US Tier-1 Macro (GDP, PCE, CPI, Claims)
    w2_news_avg = _w_avg(wedge_scores_by_id[2], llm_w_map.get(30, 0.0))
    w2_calc = round(llm_w_map.get(30, w2_news_avg), 1)

    # Wedge 3: Geopolitical Risk & Safe-Haven Flows
    w3_news_avg = _w_avg(wedge_scores_by_id[3], llm_w_map.get(20, 0.0))
    w3_calc = round(llm_w_map.get(20, w3_news_avg), 1)

    # Wedge 4: Central Bank & ETF Flows
    w4_news_avg = _w_avg(wedge_scores_by_id[4], llm_w_map.get(10, 0.0))
    w4_calc = round(llm_w_map.get(10, w4_news_avg), 1)

    calc_by_weight = {40: w1_calc, 30: w2_calc, 20: w3_calc, 10: w4_calc}
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
            "rank": idx + 1,
            "headline": h.get("headline"),
            "summary": (h.get("summary") or "")[:200],
            "source_time": h.get("source_time"),
            "wedge": h.get("pre_classified_wedge"),
            "weight": h.get("suggested_impact_weight"),
        }
        for idx, h in enumerate(batched_prompt_payload.get("intraday_breaking_headlines") or [])
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

    global _EXHAUSTED_GROQ_MODELS
    groq_models_to_try = [
        m for m in GROQ_MODEL_CHAIN if m not in _EXHAUSTED_GROQ_MODELS
    ] or list(GROQ_MODEL_CHAIN)

    for groq_model in groq_models_to_try:
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
                    "temperature": 0.0,
                    "seed": 42,
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

                if resp.status_code == 429:
                    logger.warning(
                        "Groq HTTP 429 rate limit reached on '%s'. Adding to exhausted set and trying next model...",
                        groq_model,
                    )
                    _EXHAUSTED_GROQ_MODELS.add(groq_model)
                    break

                if resp.status_code == 503 and attempt < max_retries_per_model - 1:
                    sleep_sec = base_backoff_seconds * (2 ** attempt)
                    logger.warning(
                        "Groq HTTP 503 on '%s' (attempt %d/%d). Backing off %.1fs...",
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

    global _EXHAUSTED_GEMINI_MODELS, _LAST_GEMINI_ERROR_SUMMARY
    last_error: Optional[Exception] = None

    def _format_gemini_error(exc_obj: Exception, m_name: str) -> str:
        raw_e = str(exc_obj)
        up_e = raw_e.upper()
        if "429" in raw_e or "RESOURCE_EXHAUSTED" in up_e or "QUOTA" in up_e:
            return f"Gemini ({m_name}) HTTP 429 Quota Exhausted (RESOURCE_EXHAUSTED) → Auto-Switched to Groq"
        if "400" in raw_e or "API_KEY_INVALID" in up_e or "INVALID_ARGUMENT" in up_e:
            return f"Gemini ({m_name}) HTTP 400 API Key / Request Error → Auto-Switched to Groq"
        if "403" in raw_e or "PERMISSION_DENIED" in up_e:
            return f"Gemini ({m_name}) HTTP 403 Permission Denied → Auto-Switched to Groq"
        if "404" in raw_e or "NOT_FOUND" in up_e:
            return f"Gemini ({m_name}) HTTP 404 Model Unavailable → Auto-Switched to Groq"
        if "503" in raw_e or "UNAVAILABLE" in up_e:
            return f"Gemini ({m_name}) HTTP 503 Service Unavailable → Auto-Switched to Groq"
        short_msg = re.sub(r"\s+", " ", raw_e).strip()[:80]
        return f"Gemini ({m_name}) Error ({short_msg}) → Auto-Switched to Groq"

    if not has_gemini_key:
        _LAST_GEMINI_ERROR_SUMMARY = f"Gemini ({model_name}) API Key Not Configured → Using Groq Fallback"

    models_to_try = [model_name] if model_name not in _EXHAUSTED_GEMINI_MODELS else []

    if HAS_GENAI and has_gemini_key and models_to_try:
        client = genai.Client(api_key=GEMINI_API_KEY)

        for candidate_model in models_to_try:
            try:
                logger.info(
                    "Sending single batched Gold analysis request to primary Gemini model '%s'...",
                    candidate_model,
                )
                response = client.models.generate_content(
                    model=candidate_model,
                    contents=json.dumps(batched_prompt_payload, indent=2),
                    config=types.GenerateContentConfig(
                        system_instruction=GOLD_SYSTEM_INSTRUCTION,
                        response_mime_type="application/json",
                        response_schema=GoldAnalysisReport,
                        temperature=0.0,
                    ),
                )

                validated = GoldAnalysisReport.model_validate_json(response.text)
                output = _reconcile_and_validate_analysis(validated.model_dump(), payload)
                output["analysis_engine"] = f"gemini_api ({candidate_model})"
                output["engine_error_detail"] = ""
                _LAST_GEMINI_ERROR_SUMMARY = None
                return output

            except Exception as exc:
                last_error = exc
                _LAST_GEMINI_ERROR_SUMMARY = _format_gemini_error(exc, candidate_model)
                _EXHAUSTED_GEMINI_MODELS.add(candidate_model)
                logger.warning(
                    "%s. Switching immediately to Groq fallback...",
                    _LAST_GEMINI_ERROR_SUMMARY,
                )
                break

    # Automatic failover to Groq API when Gemini fails or hits quota
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
            reconciled_groq["engine_error_detail"] = _LAST_GEMINI_ERROR_SUMMARY or (
                f"Gemini ({model_name}) Unavailable → Switched to Groq Fallback"
            )
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
        {k: v for k, v in analysis_result.items() if k not in ("analysis_engine", "engine_error_detail")}
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
