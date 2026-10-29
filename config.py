"""
Configuration module for the Gold (`XAUUSD` / COMEX `GC=F`) Intraday News &
Macro Sentiment Trading Agent.

Locks the target asset strictly onto Gold (`XAUUSD` / `GC=F`), defines the
5 live Gold & cross-asset macro driver tickers (`GC=F`, `DX-Y.NYB`, `^TNX`,
`^VIX`, `GLD`), specifies the 4 Quantitative Gold Impact Wedges (40% / 30% /
20% / 10%), and configures targeted Gold & Macro RSS ingestion queries.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List
from dotenv import load_dotenv

# Load environment variables from Render Secret Files (/etc/secrets/.env), local .env, and .env.example fallback
load_dotenv("/etc/secrets/.env", override=True)
load_dotenv(".env", override=True)
load_dotenv(".env.example", override=False)

# ---------------------------------------------------------------------------
# API & Free-Tier Model Fallback Configuration (Gemini Primary -> Groq Fallback)
# ---------------------------------------------------------------------------
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.5-flash").strip()
GEMINI_FALLBACK_MODELS: List[str] = [
    GEMINI_MODEL,
    "gemini-3.5-flash",
]
# Deduplicate model list while preserving order
GEMINI_MODEL_CHAIN: List[str] = list(dict.fromkeys(GEMINI_FALLBACK_MODELS))

GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip()
GROQ_FALLBACK_MODELS: List[str] = [
    GROQ_MODEL,
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
]
GROQ_MODEL_CHAIN: List[str] = list(dict.fromkeys(GROQ_FALLBACK_MODELS))

# ---------------------------------------------------------------------------
# Gold-Only Economic Calendar Feeds (Forex Factory JSON/XML & JBlanked API)
# ---------------------------------------------------------------------------
FOREX_FACTORY_JSON_URL: str = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
FOREX_FACTORY_XML_URL: str = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"
JBLANKED_API_KEY: str = os.getenv("JBLANKED_API_KEY", "").strip()
JBLANKED_CALENDAR_URL: str = "https://www.jblanked.com/news/api/mql5/calendar/week/"

# Non-Gold USD calendar noise (energy storage, mortgage/housing index minutiae)
GOLD_CALENDAR_EXCLUDED_PATTERNS: List[str] = [
    "crude oil",
    "natural gas",
    "api weekly",
    "hpi ",
    "new home sales",
    "pending home sales",
    "existing home sales",
    "goods trade balance",
    "wholesale inventories",
    "richmond manufacturing",
]

REQUEST_TIMEOUT_SECONDS: int = 12
MAX_BREAKING_NEWS_ITEMS: int = 18
DEDUP_SIMILARITY_THRESHOLD: float = 0.42

DEFAULT_USER_AGENT: str = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/128.0.0.0 Safari/537.36"
)

# ---------------------------------------------------------------------------
# Locked Target Asset: Gold (`XAUUSD` / COMEX Gold Futures `GC=F`)
# ---------------------------------------------------------------------------
PRIMARY_ASSET: Dict[str, str] = {
    "spot_symbol": "XAUUSD",
    "futures_symbol": "GC=F",
    "etf_symbol": "GLD",
    "name": "Gold (XAUUSD / COMEX Gold Futures GC=F)",
    "exchange": "COMEX / Spot Bullion",
    "currency": "USD/oz",
}

TARGET_TICKERS: List[str] = ["GC=F", "DX-Y.NYB", "^TNX", "^VIX", "GLD"]

# ---------------------------------------------------------------------------
# Live Gold & Macro Driver Snapshot Tickers
# ---------------------------------------------------------------------------
GOLD_MARKET_SNAPSHOT_TICKERS: Dict[str, Dict[str, Any]] = {
    "GOLD_FUTURES": {
        "symbol": "GC=F",
        "fallback_symbols": ["XAUUSD=X", "GLD"],
        "display_symbol": "GC=F / XAUUSD",
        "label": "COMEX Gold Futures (XAUUSD Proxy)",
        "unit": "USD/oz",
        "gold_relationship": "PRIMARY TARGET ASSET",
        "description": "Front-month COMEX Gold Futures tracking spot XAUUSD bullion price.",
    },
    "DXY_DOLLAR_INDEX": {
        "symbol": "DX-Y.NYB",
        "fallback_symbols": ["DX=F", "UUP"],
        "display_symbol": "DXY (DX-Y.NYB)",
        "label": "US Dollar Index (DXY)",
        "unit": "pts",
        "gold_relationship": "INVERSE DRIVER (-r)",
        "description": (
            "Primary inverse currency denominator for XAUUSD; a rising DXY makes "
            "USD-denominated Gold more expensive globally (Bearish Gold)."
        ),
    },
    "US_10Y_YIELD": {
        "symbol": "^TNX",
        "fallback_symbols": ["IEF", "TLT"],
        "display_symbol": "US 10Y (^TNX)",
        "label": "US 10-Year Treasury Yield",
        "unit": "%",
        "gold_relationship": "INVERSE REAL-YIELD DRIVER (-r)",
        "description": (
            "Opportunity-cost benchmark for zero-yielding Gold; rising yields "
            "pressure XAUUSD (Bearish), while falling yields boost XAUUSD (Bullish)."
        ),
    },
    "VIX_VOLATILITY": {
        "symbol": "^VIX",
        "fallback_symbols": ["VIXY"],
        "display_symbol": "VIX (^VIX)",
        "label": "CBOE Volatility Index (VIX)",
        "unit": "pts",
        "gold_relationship": "SAFE-HAVEN FEAR GAUGE (+r on spikes)",
        "description": (
            "Measures equity tail-risk and systemic stress; sharp VIX spikes (>20-25) "
            "trigger safe-haven rotation into physical Gold and GC=F."
        ),
    },
    "GLD_ETF": {
        "symbol": "GLD",
        "fallback_symbols": ["IAU"],
        "display_symbol": "GLD ETF",
        "label": "SPDR Gold Trust ETF",
        "unit": "USD",
        "gold_relationship": "INSTITUTIONAL FLOW PROXY (+r)",
        "description": (
            "Largest physically backed Gold ETF; tracks Western institutional and "
            "real-money allocation flows into bullion."
        ),
    },
}

# ---------------------------------------------------------------------------
# The 4 Quantitative Gold Impact Wedges (Sum = 100%)
# ---------------------------------------------------------------------------
GOLD_IMPACT_WEDGES: Dict[str, Dict[str, Any]] = {
    "WEDGE_1_FED_YIELD_DXY": {
        "wedge_id": 1,
        "wedge_name": "Wedge 1: Fed/FOMC Policy, US 10Y Yield (^TNX) & US Dollar (DXY)",
        "short_name": "Fed / 10Y Yield / DXY",
        "weight_percentage": 40,
        "weight_range": (8, 10),
        "default_impact_weight": 9,
        "description": (
            "Dominant 40% macro driver of XAUUSD. Because Gold pays zero coupon, "
            "hawkish Fed policy, rising 10Y Treasury yields (^TNX), and a strengthening "
            "US Dollar Index (DXY) exert direct downward pressure (SELL). Conversely, "
            "dovish Fed shifts, falling yields, and DXY weakness fuel Gold breakouts (BUY)."
        ),
        "keywords": [
            "federal reserve", "fomc", "powell", "fed rate", "rate cut", "rate hike", "rate hold",
            "interest rate", "hawkish", "dovish", "dot plot", "fed minutes", "fed votes",
            "treasury yield", "10-year yield", "10y yield", "real yield", "bond yield",
            "treasuries", "dxy", "dollar index", "us dollar", "greenback",
            "fed governor", "warsh", "waller", "goolsbee", "bowman", "williams", "fedwatch",
        ],
    },
    "WEDGE_2_US_TIER1_ECON": {
        "wedge_id": 2,
        "wedge_name": "Wedge 2: US Tier-1 Economic Data (CPI, Core PCE, NFP, GDP, ISM PMI)",
        "short_name": "US Tier-1 Macro Prints",
        "weight_percentage": 30,
        "weight_range": (7, 9),
        "default_impact_weight": 8,
        "description": (
            "30% macro catalyst wedge. Hotter-than-expected CPI/PCE inflation or strong "
            "NFP/GDP prints that delay Fed rate cuts push yields/DXY higher (BEARISH Gold). "
            "Cooling inflation, rising unemployment/jobless claims, weak NFP/ISM PMI, or "
            "stagflationary data (weak growth + sticky inflation) are BULLISH Gold."
        ),
        "keywords": [
            "cpi", "consumer price index", "core pce", "pce", "inflation", "ppi",
            "nonfarm payroll", "nfp", "jobs report", "unemployment", "jobless claims",
            "initial claims", "gdp", "economic growth", "ism", "pmi", "manufacturing pmi",
            "services pmi", "retail sales", "consumer confidence", "stagflation", "recession",
            "jolts", "adp",
        ],
    },
    "WEDGE_3_GEOPOLITICAL_SAFE_HAVEN": {
        "wedge_id": 3,
        "wedge_name": "Wedge 3: Geopolitical Risk, Tariffs, War Escalation & Safe-Haven Flows (^VIX)",
        "short_name": "Geopolitics & Safe-Haven",
        "weight_percentage": 20,
        "weight_range": (8, 10),
        "default_impact_weight": 9,
        "description": (
            "20% tail-risk wedge (Weight 8-10 on acute spikes). War escalation, military "
            "strikes, trade/tariff shocks, sovereign debt/fiscal crises, or banking stress "
            "trigger urgent safe-haven buying in XAUUSD. CRITICAL OVERRIDE: During acute "
            "geopolitical panic, safe-haven Gold buying OVERRIDES a rising DXY (both rally together)."
        ),
        "keywords": [
            "geopolitical", "safe haven", "safe-haven", "war", "military", "missile",
            "airstrike", "middle east", "iran", "israel", "ukraine", "russia", "taiwan",
            "tariff", "trade war", "sanctions", "ceasefire", "nuclear",
            "sovereign debt", "debt ceiling", "banking stress", "vix", "risk-off",
        ],
    },
    "WEDGE_4_CENTRAL_BANK_ETF_FLOWS": {
        "wedge_id": 4,
        "wedge_name": "Wedge 4: Central Bank Gold Buying, COMEX Positioning & ETF (GLD) Flows",
        "short_name": "CB Buying / COMEX / GLD",
        "weight_percentage": 10,
        "weight_range": (3, 6),
        "default_impact_weight": 5,
        "description": (
            "10% structural & physical flow wedge. PBoC/EM central bank sovereign bullion "
            "purchases, World Gold Council reserves data, CFTC COT managed-money positioning, "
            "COMEX vault deliveries, and SPDR GLD ETF inflows/outflows."
        ),
        "keywords": [
            "central bank gold", "central bank buying", "gold reserves", "pboc",
            "world gold council", "comex", "gld", "gold etf", "etf inflows",
            "etf outflows", "bullion", "physical gold", "xauusd", "xau/usd",
            "gold price", "gold futures", "spot gold", "precious metals", "kitco",
            "cot report", "cftc", "sovereign wealth",
        ],
    },
}

# Backwards-compatible alias
IMPACT_WEIGHT_TIERS = GOLD_IMPACT_WEDGES

# ---------------------------------------------------------------------------
# Targeted Google News RSS Queries (Strictly Locked to Gold & Its 4 Wedges)
# ---------------------------------------------------------------------------
GOLD_GOOGLE_NEWS_QUERIES: List[Dict[str, str]] = [
    {
        "source": "Google News (Gold / XAUUSD / COMEX)",
        "query": '("XAUUSD" OR "Gold price" OR "Gold futures" OR "COMEX Gold") when:2d',
        "wedge_hint": "WEDGE_4_CENTRAL_BANK_ETF_FLOWS",
    },
    {
        "source": "Google News (Fed / FOMC / US 10Y Yields)",
        "query": '("Federal Reserve" OR "FOMC" OR "US Treasury yields" OR "Fed rate") when:2d',
        "wedge_hint": "WEDGE_1_FED_YIELD_DXY",
    },
    {
        "source": "Google News (US CPI / NFP / Core PCE / DXY)",
        "query": '("US CPI" OR "Nonfarm Payrolls" OR "Core PCE" OR "US Dollar Index") when:2d',
        "wedge_hint": "WEDGE_2_US_TIER1_ECON",
    },
    {
        "source": "Google News (Geopolitics / Safe Haven / CB Gold)",
        "query": '("geopolitical gold" OR "safe haven gold" OR "central bank gold" OR "geopolitical tensions market") when:2d',
        "wedge_hint": "WEDGE_3_GEOPOLITICAL_SAFE_HAVEN",
    },
]

# Direct Financial RSS Feeds Relevant to Gold & Macro Drivers
GOLD_DIRECT_RSS_FEEDS: List[Dict[str, str]] = [
    {
        "source": "Yahoo Finance Gold Futures (GC=F)",
        "url": "https://feeds.finance.yahoo.com/rss/2.0/headline?s=GC=F&region=US&lang=en-US",
        "wedge_hint": "WEDGE_4_CENTRAL_BANK_ETF_FLOWS",
        "require_keyword_filter": "true",
    },
    {
        "source": "Yahoo Finance SPDR Gold Trust (GLD)",
        "url": "https://feeds.finance.yahoo.com/rss/2.0/headline?s=GLD&region=US&lang=en-US",
        "wedge_hint": "WEDGE_4_CENTRAL_BANK_ETF_FLOWS",
        "require_keyword_filter": "true",
    },
    {
        "source": "Yahoo Finance 10Y Treasury (^TNX)",
        "url": "https://feeds.finance.yahoo.com/rss/2.0/headline?s=%5ETNX&region=US&lang=en-US",
        "wedge_hint": "WEDGE_1_FED_YIELD_DXY",
        "require_keyword_filter": "true",
    },
    {
        "source": "Kitco Gold & Precious Metals",
        "url": "https://www.kitco.com/rss/KitcoNews.xml",
        "wedge_hint": "WEDGE_4_CENTRAL_BANK_ETF_FLOWS",
        "require_keyword_filter": "true",
    },
    {
        "source": "Investing.com Commodities & Gold",
        "url": "https://www.investing.com/rss/news_11.rss",
        "wedge_hint": "WEDGE_4_CENTRAL_BANK_ETF_FLOWS",
        "require_keyword_filter": "true",
    },
    {
        "source": "Investing.com Economic Indicators",
        "url": "https://www.investing.com/rss/news_95.rss",
        "wedge_hint": "WEDGE_2_US_TIER1_ECON",
        "require_keyword_filter": "true",
    },
    {
        "source": "CNBC Economy & Fed",
        "url": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258",
        "wedge_hint": "WEDGE_1_FED_YIELD_DXY",
        "require_keyword_filter": "true",
    },
    {
        "source": "CNBC Top Financial News",
        "url": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
        "wedge_hint": "WEDGE_2_US_TIER1_ECON",
        "require_keyword_filter": "true",
    },
    {
        "source": "ForexLive Macro & FX",
        "url": "https://www.forexlive.com/feed/news",
        "wedge_hint": "WEDGE_1_FED_YIELD_DXY",
        "require_keyword_filter": "true",
    },
]


# Noise patterns (retail chart uploads, junior exploration drill PRs, non-financial safe-haven stories, single-stock PR)
EXCLUDED_NOISE_PATTERNS: List[str] = [
    "chart image by",
    "g/t au",
    "g/t ag",
    "g/t gold",
    "g/t silver",
    "intersects",
    "grading ",
    "hectares",
    "pre-feasibility",
    "shovelnose",
    "airborne magnetic",
    "drill results",
    "drilling results",
    "preliminary economic assessment",
    "updated pea",
    "lot sizing blueprint",
    "baby box",
    "safe haven baby",
    "safe haven location",
    "newborn",
    "infant",
    "stock pops",
    "stock jumps",
    "stock plunges",
    "shares pop",
    "shares jump",
    "shares surge",
    "revenue forecast",
    "quarterly revenue",
    "eps beats",
]


def _kw_in_text(kw: str, normalized_text: str) -> bool:
    """Word-boundary match for keywords so short acronyms ('ism', 'ppi', 'war') do not match inside unrelated words."""
    return bool(re.search(rf"\b{re.escape(kw)}\b", normalized_text))


def is_headline_gold_or_macro_relevant(text: str) -> bool:
    """
    Check whether a general financial headline is directly relevant to Gold (`XAUUSD`/`GC=F`)
    or one of the 4 Gold Impact Wedges (Fed/Yields/DXY, US Tier-1 Macro, Geopolitics/Safe-Haven,
    Central Bank/ETF Gold flows). Filters out unrelated single-stock corporate PR and retail chart spam.
    """
    normalized = (text or "").lower()
    if any(noise in normalized for noise in EXCLUDED_NOISE_PATTERNS):
        return False
    for wedge in GOLD_IMPACT_WEDGES.values():
        if any(_kw_in_text(kw, normalized) for kw in wedge["keywords"]):
            return True
    return False


def classify_gold_wedge(
    text: str,
    fallback_wedge_key: str = "WEDGE_4_CENTRAL_BANK_ETF_FLOWS",
) -> Dict[str, Any]:
    """
    Classify a headline/summary into one of the 4 Gold Impact Wedges and compute
    its initial quantitative impact weight (1 to 10).
    """
    normalized = (text or "").lower()

    best_key = fallback_wedge_key if fallback_wedge_key in GOLD_IMPACT_WEDGES else "WEDGE_4_CENTRAL_BANK_ETF_FLOWS"
    best_matches: List[str] = []
    best_score = -1

    # Priority order: Wedge 1 (40%), Wedge 3 (20% acute spikes), Wedge 2 (30%), Wedge 4 (10%)
    evaluation_order = [
        "WEDGE_1_FED_YIELD_DXY",
        "WEDGE_3_GEOPOLITICAL_SAFE_HAVEN",
        "WEDGE_2_US_TIER1_ECON",
        "WEDGE_4_CENTRAL_BANK_ETF_FLOWS",
    ]

    for w_key in evaluation_order:
        wedge = GOLD_IMPACT_WEDGES[w_key]
        matches = [kw for kw in wedge["keywords"] if _kw_in_text(kw, normalized)]
        if matches:
            # Weight by number of keyword hits and wedge priority
            score = len(matches) * 10 + (1 if w_key == fallback_wedge_key else 0)
            if score > best_score:
                best_score = score
                best_key = w_key
                best_matches = matches

    wedge = GOLD_IMPACT_WEDGES[best_key]
    min_w, max_w = wedge["weight_range"]
    if len(best_matches) >= 3:
        impact_weight = max_w
    elif len(best_matches) == 2:
        impact_weight = min(max_w, wedge["default_impact_weight"])
    elif len(best_matches) == 1:
        impact_weight = wedge["default_impact_weight"]
    else:
        impact_weight = min_w

    return {
        "wedge_key": best_key,
        "wedge_id": wedge["wedge_id"],
        "wedge_name": wedge["wedge_name"],
        "short_name": wedge["short_name"],
        "weight_percentage": wedge["weight_percentage"],
        "impact_weight": int(impact_weight),
        "matched_keywords": best_matches[:5],
    }
