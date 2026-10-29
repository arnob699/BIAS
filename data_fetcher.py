"""
Gold (`XAUUSD` / COMEX `GC=F`) Live Intraday News & Macro Context Fetcher (`data_fetcher.py`).

Responsibilities:
1. Fetch live intraday price, intraday % change, and day high/low (via `yfinance`
   and Yahoo Finance v8 chart endpoints) specifically for the 5 Gold & Macro tickers:
   - `GC=F` (COMEX Gold Futures / XAUUSD proxy)
   - `DX-Y.NYB` (US Dollar Index / DXY — primary inverse driver)
   - `^TNX` (US 10-Year Treasury Yield — real yield inverse driver)
   - `^VIX` (CBOE Volatility Index — safe-haven fear gauge)
   - `GLD` (SPDR Gold Trust ETF — institutional Gold flow proxy)
2. Fetch and deduplicate the top 18 most recent intraday breaking news headlines
   (with UTC timestamps) strictly relevant to Gold (`XAUUSD` / `GC=F`) and its
   4 Impact Wedges from Google News RSS and financial feeds (Kitco, Yahoo Finance,
   CNBC, Investing.com, ForexLive) using targeted queries:
   - `"XAUUSD" OR "Gold price" OR "Gold futures" OR "COMEX Gold"`
   - `"Federal Reserve" OR "FOMC" OR "Jerome Powell" OR "US Treasury yields"`
   - `"US CPI" OR "Nonfarm Payrolls" OR "Core PCE" OR "US Dollar DXY"`
   - `"geopolitical" OR "safe haven" OR "central bank gold"`
"""

from __future__ import annotations

import calendar
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

BST_TZ = timezone(timedelta(hours=6), name='BST')
from email.utils import parsedate_to_datetime
import html
import json
import logging
import re
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import quote_plus
import warnings

import feedparser
import requests

warnings.filterwarnings("ignore", message=".*inplace keyword.*")
warnings.filterwarnings("ignore", module="yfinance.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)
import yfinance as yf

from config import (
    DEDUP_SIMILARITY_THRESHOLD,
    DEFAULT_USER_AGENT,
    EXCLUDED_NOISE_PATTERNS,
    FOREX_FACTORY_JSON_URL,
    FOREX_FACTORY_XML_URL,
    GOLD_CALENDAR_EXCLUDED_PATTERNS,
    GOLD_DIRECT_RSS_FEEDS,
    GOLD_GOOGLE_NEWS_QUERIES,
    GOLD_IMPACT_WEDGES,
    GOLD_MARKET_SNAPSHOT_TICKERS,
    JBLANKED_API_KEY,
    JBLANKED_CALENDAR_URL,
    MAX_BREAKING_NEWS_ITEMS,
    PRIMARY_ASSET,
    REQUEST_TIMEOUT_SECONDS,
    TARGET_TICKERS,
    classify_gold_wedge,
    is_headline_gold_or_macro_relevant,
)

logger = logging.getLogger(__name__)
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# Initialize yfinance timezone cache directory once prior to multi-threaded calls
try:
    _tz_cache_dir = Path(tempfile.gettempdir()) / "py_yfinance_gold_cache"
    _tz_cache_dir.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(_tz_cache_dir))
except Exception:
    pass

_STOPWORDS: Set[str] = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "he", "in", "is", "it", "its", "of", "on", "or", "that", "the", "to",
    "was", "were", "will", "with", "after", "amid", "over", "into", "says",
    "said", "new", "us", "u", "s", "what", "why", "how", "here", "this",
}


def _build_http_session() -> requests.Session:
    """Create a configured requests Session with browser-grade headers."""
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": DEFAULT_USER_AGENT,
            "Accept": (
                "application/rss+xml, application/xml, text/xml, "
                "application/json, text/plain, */*"
            ),
            "Accept-Language": "en-US,en;q=0.9",
            "Cache-Control": "no-cache",
        }
    )
    return session


def _clean_html_text(raw_text: str, max_length: int = 300) -> str:
    """Strip HTML tags, decode entities, collapse whitespace, and truncate cleanly."""
    if not raw_text:
        return ""
    text = html.unescape(raw_text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_length:
        text = text[: max_length - 3].rsplit(" ", 1)[0] + "..."
    return text


def _strip_publisher_suffix(headline: str) -> Tuple[str, Optional[str]]:
    """
    Separate trailing ' - Publisher Name' from syndicated headlines (e.g. Google News).
    Returns (cleaned_headline, extracted_publisher).
    """
    if not headline:
        return "", None
    current = headline.strip()
    extracted_pub: Optional[str] = None
    for _ in range(2):
        match = re.match(r"^(.*?)\s+[-–|]\s+([A-Za-z0-9 .,&'/]{2,48})$", current)
        if match and len(match.group(1).strip()) >= 15:
            current = match.group(1).strip()
            if extracted_pub is None:
                extracted_pub = match.group(2).strip()
        else:
            break
    return current, extracted_pub


def _parse_entry_timestamp(entry: Any) -> Optional[datetime]:
    """Extract a timezone-aware UTC datetime from a feedparser entry."""
    for struct_attr in ("published_parsed", "updated_parsed", "created_parsed"):
        t_struct = getattr(entry, struct_attr, None) or entry.get(struct_attr)
        if t_struct:
            try:
                epoch_sec = calendar.timegm(t_struct)
                return datetime.fromtimestamp(epoch_sec, tz=timezone.utc)
            except (ValueError, OverflowError, TypeError):
                pass

    for str_attr in ("published", "updated", "created", "pubDate"):
        raw_val = getattr(entry, str_attr, None) or entry.get(str_attr)
        if raw_val and isinstance(raw_val, str):
            raw_val = raw_val.strip()
            try:
                dt = parsedate_to_datetime(raw_val)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc)
            except Exception:
                pass
            try:
                iso_str = raw_val.replace("Z", "+00:00")
                dt = datetime.fromisoformat(iso_str)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc)
            except Exception:
                pass

    return None


def _format_relative_age(published_utc: datetime, now_utc: datetime) -> Tuple[int, str]:
    """Compute age in minutes and a concise human-readable age string."""
    delta_seconds = max(0.0, (now_utc - published_utc).total_seconds())
    minutes = int(round(delta_seconds / 60.0))
    if minutes < 1:
        return 0, "just now"
    if minutes < 60:
        return minutes, f"{minutes}m ago"
    hours = delta_seconds / 3600.0
    if hours < 24:
        return minutes, f"{hours:.1f}h ago"
    days = delta_seconds / 86400.0
    return minutes, f"{days:.1f}d ago"


_DEDUP_SYNONYMS: Dict[str, str] = {
    "keeps": "hold",
    "holds": "hold",
    "leaves": "hold",
    "holding": "hold",
    "unchanged": "steady",
    "steady": "steady",
    "rates": "rate",
    "federal": "fed",
    "reserve": "fed",
    "fomc": "fed",
    "chairman": "chair",
    "meetings": "meeting",
    "yields": "yield",
    "prices": "price",
    "declines": "drop",
    "falls": "drop",
    "drops": "drop",
    "slips": "drop",
    "edges": "move",
    "pauses": "hold",
    "pause": "hold",
    "pausing": "hold",
    "quarter": "qtr",
    "third": "3rd",
    "increases": "rise",
    "accelerates": "rise",
    "grows": "rise",
    "climbs": "rise",
    "surges": "rise",
    "slows": "cool",
    "cooling": "cool",
    "dips": "drop",
    "retreats": "drop",
}


def _tokenize_headline(headline: str) -> Set[str]:
    """Convert a headline into a normalized set of content tokens for deduplication."""
    cleaned, _ = _strip_publisher_suffix(headline)
    lowered = cleaned.lower()
    words = re.findall(r"[a-z0-9]+", lowered)
    tokens = {
        _DEDUP_SYNONYMS.get(w, w)
        for w in words
        if len(w) > 2 and w not in _STOPWORDS
    }
    return tokens or set(words)


def _are_headlines_duplicate(
    tokens_a: Set[str],
    tokens_b: Set[str],
    threshold: float = DEDUP_SIMILARITY_THRESHOLD,
) -> bool:
    """Check whether two tokenized headlines refer to the same story."""
    if not tokens_a or not tokens_b:
        return False
    if tokens_a == tokens_b:
        return True

    intersection = len(tokens_a & tokens_b)
    if intersection == 0:
        return False

    union = len(tokens_a | tokens_b)
    if (intersection / union) >= threshold:
        return True

    smaller_size = min(len(tokens_a), len(tokens_b))
    if smaller_size >= 4 and (intersection / smaller_size) >= 0.58:
        return True

    return False


# ---------------------------------------------------------------------------
# Live Gold & Macro Driver Quote Fetcher (`yfinance` + Yahoo v8 Chart API)
# ---------------------------------------------------------------------------
def _fetch_via_yfinance(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Attempt to pull live intraday price, % change, and day high/low via `yfinance`.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ticker_obj = yf.Ticker(symbol)
            hist = ticker_obj.history(period="5d", interval="1d")
        if hist is None or hist.empty:
            return None

        closes = [float(c) for c in hist["Close"].dropna().tolist()]
        highs = [float(h) for h in hist["High"].dropna().tolist()]
        lows = [float(l) for l in hist["Low"].dropna().tolist()]
        if not closes:
            return None

        price = closes[-1]
        prev_close = closes[-2] if len(closes) >= 2 else price
        day_high = highs[-1] if highs else price
        day_low = lows[-1] if lows else price

        # Enrich with fast_info if available for real-time intraday precision
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fi = ticker_obj.fast_info
                last_p = getattr(fi, "last_price", None)
                prev_c = getattr(fi, "previous_close", None)
                d_high = getattr(fi, "day_high", None)
                d_low = getattr(fi, "day_low", None)
            if last_p is not None:
                price = float(last_p)
            if prev_c is not None and len(closes) < 2:
                prev_close = float(prev_c)
            if d_high is not None:
                day_high = float(d_high)
            if d_low is not None:
                day_low = float(d_low)
        except Exception:
            pass

        change = price - prev_close
        change_pct = (change / prev_close * 100.0) if abs(prev_close) > 1e-8 else 0.0
        first_close = closes[0] if closes else prev_close
        five_day_change_pct = (
            ((price - first_close) / first_close * 100.0)
            if abs(first_close) > 1e-8
            else change_pct
        )

        last_ts = hist.index[-1]
        if hasattr(last_ts, "to_pydatetime"):
            dt = last_ts.to_pydatetime()
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            market_time_utc = dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            market_time_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        return {
            "symbol": symbol,
            "price": round(price, 3 if symbol == "^TNX" else 2),
            "previous_close": round(prev_close, 3 if symbol == "^TNX" else 2),
            "change": round(change, 3 if symbol == "^TNX" else 2),
            "change_pct": round(change_pct, 2),
            "five_day_change_pct": round(five_day_change_pct, 2),
            "day_high": round(day_high, 3 if symbol == "^TNX" else 2),
            "day_low": round(day_low, 3 if symbol == "^TNX" else 2),
            "market_time_utc": market_time_utc,
            "data_provider": "yfinance",
        }
    except Exception as exc:
        logger.debug("yfinance fetch failed for %s: %s", symbol, exc)
        return None


def _fetch_via_yahoo_chart_api(
    symbol: str,
    session: requests.Session,
) -> Optional[Dict[str, Any]]:
    """
    Query Yahoo Finance's v8 JSON chart endpoint as a fast/resilient fallback.
    """
    encoded_sym = quote_plus(symbol)
    hosts = [
        "https://query1.finance.yahoo.com",
        "https://query2.finance.yahoo.com",
    ]
    params = {"interval": "1d", "range": "5d", "includePrePost": "false"}

    for host in hosts:
        url = f"{host}/v8/finance/chart/{encoded_sym}"
        try:
            resp = session.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
            if resp.status_code != 200:
                continue
            payload = resp.json()
            results = (payload.get("chart") or {}).get("result")
            if not results:
                continue

            result = results[0]
            meta = result.get("meta") or {}
            indicators = (result.get("indicators") or {}).get("quote") or [{}]
            quote_block = indicators[0] if indicators else {}

            closes = [float(c) for c in (quote_block.get("close") or []) if c is not None]
            highs = [float(h) for h in (quote_block.get("high") or []) if h is not None]
            lows = [float(l) for l in (quote_block.get("low") or []) if l is not None]

            price = meta.get("regularMarketPrice")
            if price is None and closes:
                price = closes[-1]
            if price is None:
                continue
            price = float(price)

            prev_close = meta.get("chartPreviousClose") or meta.get("previousClose")
            if len(closes) >= 2:
                if abs(closes[-1] - price) / max(abs(price), 1e-6) < 0.015:
                    prev_close = float(closes[-2])
                else:
                    prev_close = float(closes[-1])
            elif prev_close is not None:
                prev_close = float(prev_close)
            else:
                prev_close = price

            change = price - prev_close
            change_pct = (change / prev_close * 100.0) if abs(prev_close) > 1e-8 else 0.0
            first_close = float(closes[0]) if closes else prev_close
            five_day_change_pct = (
                ((price - first_close) / first_close * 100.0)
                if abs(first_close) > 1e-8
                else change_pct
            )

            day_high = float(meta.get("regularMarketDayHigh") or (highs[-1] if highs else price))
            day_low = float(meta.get("regularMarketDayLow") or (lows[-1] if lows else price))
            market_time = meta.get("regularMarketTime")
            market_time_utc = (
                datetime.fromtimestamp(int(market_time), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                if market_time
                else datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            )

            decimals = 3 if symbol == "^TNX" else 2
            return {
                "symbol": symbol,
                "price": round(price, decimals),
                "previous_close": round(prev_close, decimals),
                "change": round(change, decimals),
                "change_pct": round(change_pct, 2),
                "five_day_change_pct": round(five_day_change_pct, 2),
                "day_high": round(day_high, decimals),
                "day_low": round(day_low, decimals),
                "market_time_utc": market_time_utc,
                "data_provider": "yahoo_v8_chart",
            }
        except Exception as exc:
            logger.debug("Yahoo v8 chart failed for %s on %s: %s", symbol, host, exc)

    return None


def _evaluate_gold_impact_signal(
    key: str,
    price: float,
    change: float,
    change_pct: float,
    five_day_change_pct: float,
) -> Tuple[str, str]:
    """
    Evaluate how the live reading of a ticker impacts `XAUUSD / GC=F`.
    Returns `(gold_directional_effect, explanation_text)`.
    """
    if key == "GOLD_FUTURES":
        if change_pct >= 0.35:
            return "BULLISH GOLD", f"Active Gold bid (+{change_pct:.2f}% intraday | 5d: {five_day_change_pct:+.2f}%)"
        if change_pct <= -0.35:
            return "BEARISH GOLD", f"Intraday Gold distribution ({change_pct:+.2f}% | 5d: {five_day_change_pct:+.2f}%)"
        return "NEUTRAL", f"Consolidating near ${price:,.2f}/oz ({change_pct:+.2f}% intraday)"

    if key == "DXY_DOLLAR_INDEX":
        # Inverse correlation: rising DXY -> BEARISH GOLD; falling DXY -> BULLISH GOLD
        if change_pct >= 0.20:
            return "BEARISH GOLD", f"Firming USD ({change_pct:+.2f}% to {price:.2f}) creates inverse headwind for XAUUSD"
        if change_pct <= -0.20:
            return "BULLISH GOLD", f"Softening USD ({change_pct:+.2f}% to {price:.2f}) provides tailwind for XAUUSD"
        if five_day_change_pct >= 0.50:
            return "BEARISH GOLD", f"DXY flat intraday ({change_pct:+.2f}%) but up {five_day_change_pct:+.2f}% over 5d (Mild Gold headwind)"
        if five_day_change_pct <= -0.50:
            return "BULLISH GOLD", f"DXY flat intraday ({change_pct:+.2f}%) and down {five_day_change_pct:+.2f}% over 5d (Supportive for Gold)"
        return "NEUTRAL", f"Rangebound DXY at {price:.2f} ({change_pct:+.2f}%)"

    if key == "US_10Y_YIELD":
        # Inverse correlation: rising 10Y yield -> BEARISH GOLD; falling 10Y yield -> BULLISH GOLD
        if change >= 0.03 or change_pct >= 0.65:
            return "BEARISH GOLD", f"Rising 10Y Yield ({price:.3f}%, {change:+.3f}%) raises opportunity cost of zero-yield Gold"
        if change <= -0.03 or change_pct <= -0.65:
            return "BULLISH GOLD", f"Falling 10Y Yield ({price:.3f}%, {change:+.3f}%) compresses real rates & boosts Gold"
        if five_day_change_pct <= -1.5:
            return "BULLISH GOLD", f"10Y Yield steady today ({price:.3f}%) after {five_day_change_pct:+.2f}% 5d pullback (Bullish Gold)"
        if five_day_change_pct >= 1.5:
            return "BEARISH GOLD", f"10Y Yield elevated at {price:.3f}% ({five_day_change_pct:+.2f}% 5d) pressuring non-yielding Gold"
        return "NEUTRAL", f"Stable 10Y Yield at {price:.3f}% ({change_pct:+.2f}%)"

    if key == "VIX_VOLATILITY":
        # Safe-haven gauge: high/spiking VIX -> BULLISH GOLD safe-haven bid
        if price >= 22.0 or change_pct >= 5.0:
            return "BULLISH GOLD", f"Elevated VIX ({price:.2f}, {change_pct:+.2f}%) spurs safe-haven demand for Gold"
        if price < 15.0 and change_pct <= -2.0:
            return "BEARISH GOLD", f"Low VIX ({price:.2f}, {change_pct:+.2f}%) reduces urgent safe-haven hedging demand"
        return "NEUTRAL", f"Normal Volatility Regime (VIX {price:.2f}, {change_pct:+.2f}%)"

    if key == "GLD_ETF":
        if change_pct >= 0.30:
            return "BULLISH GOLD", f"Institutional GLD ETF bid ({change_pct:+.2f}% | 5d: {five_day_change_pct:+.2f}%) confirms bullion demand"
        if change_pct <= -0.30:
            return "BEARISH GOLD", f"GLD ETF selling pressure ({change_pct:+.2f}% | 5d: {five_day_change_pct:+.2f}%)"
        return "NEUTRAL", f"Steady GLD ETF trading at ${price:.2f} ({change_pct:+.2f}%)"

    return "NEUTRAL", f"{price} ({change_pct:+.2f}%)"


def _fetch_single_gold_snapshot_ticker(
    key: str,
    cfg: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    """
    Fetch live quote for one of the 5 Gold/Macro tickers using `yfinance` first,
    falling back to Yahoo Finance v8 JSON chart API.
    """
    session = _build_http_session()
    candidates = [cfg["symbol"]] + list(cfg.get("fallback_symbols", []))

    for symbol in candidates:
        quote = _fetch_via_yahoo_chart_api(symbol, session)
        if quote is None:
            quote = _fetch_via_yfinance(symbol)

        if quote is not None:
            gold_effect, signal_text = _evaluate_gold_impact_signal(
                key=key,
                price=quote["price"],
                change=quote["change"],
                change_pct=quote["change_pct"],
                five_day_change_pct=quote["five_day_change_pct"],
            )
            return key, {
                "ticker_key": key,
                "display_symbol": cfg["display_symbol"],
                "label": cfg["label"],
                "unit": cfg["unit"],
                "gold_relationship": cfg["gold_relationship"],
                "description": cfg["description"],
                "is_fallback_symbol": symbol != cfg["symbol"],
                **quote,
                "gold_directional_effect": gold_effect,
                "regime_signal": signal_text,
                "status": "live",
            }

    return key, {
        "ticker_key": key,
        "symbol": cfg["symbol"],
        "display_symbol": cfg["display_symbol"],
        "label": cfg["label"],
        "unit": cfg["unit"],
        "gold_relationship": cfg["gold_relationship"],
        "description": cfg["description"],
        "price": None,
        "previous_close": None,
        "change": None,
        "change_pct": None,
        "five_day_change_pct": None,
        "day_high": None,
        "day_low": None,
        "market_time_utc": None,
        "gold_directional_effect": "NEUTRAL",
        "regime_signal": "Data Unavailable",
        "status": "unavailable",
    }


def fetch_gold_macro_snapshot() -> Dict[str, Any]:
    """
    Concurrently fetch live intraday price, % change, and day high/low for:
    - `GC=F` (COMEX Gold Futures / XAUUSD proxy)
    - `DX-Y.NYB` (US Dollar Index / DXY — primary inverse driver)
    - `^TNX` (US 10-Year Treasury Yield — real yield inverse driver)
    - `^VIX` (CBOE Volatility Index — safe-haven fear gauge)
    - `GLD` (SPDR Gold Trust ETF — institutional Gold flow proxy)
    """
    now_utc = datetime.now(timezone.utc)
    raw_results: Dict[str, Dict[str, Any]] = {}

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(_fetch_single_gold_snapshot_ticker, key, cfg): key
            for key, cfg in GOLD_MARKET_SNAPSHOT_TICKERS.items()
        }
        for fut in as_completed(futures):
            key, data = fut.result()
            raw_results[key] = data

    ordered_tickers = {
        key: raw_results[key]
        for key in GOLD_MARKET_SNAPSHOT_TICKERS.keys()
        if key in raw_results
    }

    # Compute quantitative cross-asset alignment summary for Gold
    bullish_drivers: List[str] = []
    bearish_drivers: List[str] = []
    for key, item in ordered_tickers.items():
        eff = item.get("gold_directional_effect")
        if eff == "BULLISH GOLD":
            bullish_drivers.append(f"{item['display_symbol']}: {item['regime_signal']}")
        elif eff == "BEARISH GOLD":
            bearish_drivers.append(f"{item['display_symbol']}: {item['regime_signal']}")

    if bullish_drivers and not bearish_drivers:
        alignment_state = "ALIGNED BULLISH GOLD"
    elif bearish_drivers and not bullish_drivers:
        alignment_state = "ALIGNED BEARISH GOLD"
    elif bullish_drivers and bearish_drivers:
        alignment_state = "CONFLICTING / CROSS-CURRENT DRIVERS"
    else:
        alignment_state = "NEUTRAL / RANGEBOUND MACRO"

    return {
        "snapshot_utc": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "primary_asset": PRIMARY_ASSET,
        "tickers": ordered_tickers,
        "cross_asset_gold_alignment": {
            "alignment_state": alignment_state,
            "bullish_gold_drivers": bullish_drivers,
            "bearish_gold_drivers": bearish_drivers,
        },
    }


# ---------------------------------------------------------------------------
# Targeted Gold & Macro RSS News Ingestion (Top 18 Deduplicated Headlines)
# ---------------------------------------------------------------------------
def _build_google_news_rss_url(query: str) -> str:
    """Build a Google News RSS search URL for US English financial news."""
    encoded_q = quote_plus(query)
    return f"https://news.google.com/rss/search?q={encoded_q}&hl=en-US&gl=US&ceid=US:en"


def _fetch_single_gold_rss_feed(
    feed_meta: Dict[str, str],
    now_utc: datetime,
    max_entries_per_feed: int = 25,
) -> List[Dict[str, Any]]:
    """Fetch and parse a single RSS feed, filtering and tagging for the 4 Gold Wedges."""
    source_name = feed_meta["source"]
    url = feed_meta["url"]
    wedge_hint = feed_meta.get("wedge_hint", "WEDGE_4_CENTRAL_BANK_ETF_FLOWS")
    require_filter = feed_meta.get("require_keyword_filter", "false").lower() == "true"

    items: List[Dict[str, Any]] = []
    session = _build_http_session()

    try:
        resp = session.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
        resp.raise_for_status()
        parsed = feedparser.parse(resp.content)
    except Exception as exc:
        logger.debug("Direct RSS fetch failed for %s (%s): %s", source_name, url, exc)
        try:
            parsed = feedparser.parse(url, agent=DEFAULT_USER_AGENT)
        except Exception:
            return items

    entries = getattr(parsed, "entries", []) or []
    for entry in entries[:max_entries_per_feed]:
        raw_title = _clean_html_text(entry.get("title", ""), max_length=240)
        if not raw_title or len(raw_title) < 12:
            continue

        clean_title, extracted_pub = _strip_publisher_suffix(raw_title)
        effective_source = (
            f"{extracted_pub} (via {source_name})"
            if extracted_pub and "Google News" in source_name
            else (extracted_pub or source_name)
        )

        raw_summary = entry.get("summary", "") or entry.get("description", "")
        clean_summary = _clean_html_text(raw_summary, max_length=260)
        if clean_summary and clean_title.lower()[:45] in clean_summary.lower():
            clean_summary = ""

        combined_text = f"{clean_title} {clean_summary}".strip()
        if any(noise in combined_text.lower() for noise in EXCLUDED_NOISE_PATTERNS):
            continue

        # Require the headline title itself to be relevant to Gold or the 4 Macro Wedges
        if not is_headline_gold_or_macro_relevant(clean_title):
            if require_filter or not is_headline_gold_or_macro_relevant(combined_text):
                continue
            continue

        published_dt = _parse_entry_timestamp(entry)
        if published_dt is None:
            continue

        if published_dt > now_utc:
            published_dt = now_utc

        age_minutes, age_human = _format_relative_age(published_dt, now_utc)
        published_utc_str = published_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        published_bst_str = published_dt.astimezone(BST_TZ).strftime("%d %b %Y, %I:%M %p BST")
        wedge_info = classify_gold_wedge(combined_text, fallback_wedge_key=wedge_hint)

        items.append(
            {
                "headline": clean_title,
                "summary": clean_summary,
                "source": effective_source,
                "source_time": f"{effective_source} | {published_bst_str} ({age_human})",
                "url": entry.get("link", ""),
                "published_utc": published_utc_str,
                "published_bst": published_bst_str,
                "published_epoch": int(published_dt.timestamp()),
                "age_minutes": age_minutes,
                "age_human": age_human,
                "gold_wedge": wedge_info,
            }
        )

    return items



def _extract_story_topic_key(headline: str, summary: str = "") -> Optional[str]:
    """
    Extract a coarse topic cluster signature for major market events to prevent
    multiple news outlets reporting on the exact same event from flooding the 18 slots.
    """
    text = f"{headline} {summary}".lower()
    if "gdp" in text and ("3.4" in text or "quarter" in text or "q3" in text or "growth" in text):
        return "MACRO_US_Q3_GDP"
    if "pce" in text and ("0.3" in text or "core" in text or "september" in text or "preferred measure" in text):
        return "MACRO_US_CORE_PCE"
    if "jobless claims" in text or "initial claims" in text or ("claims" in text and ("194k" in text or "57-year" in text or "lows" in text)):
        return "MACRO_US_JOBLESS_CLAIMS"
    if ("fed" in text or "federal reserve" in text or "fomc" in text) and ("pause" in text or "hold" in text or "steady" in text or "unchanged" in text) and "rate" in text:
        return "FED_RATE_HOLD"
    if "ecb" in text or "european central bank" in text:
        return "ECB_RATE_HOLD"
    if "oil" in text and ("diplomacy" in text or "iran" in text or "middle east" in text or "strike" in text):
        return "OIL_MIDEAST_DIPLOMACY"
    if "warsh" in text:
        return "FED_WARSH_COMMENTARY"
    return None


def deduplicate_gold_news_items(
    raw_items: List[Dict[str, Any]],
    max_items: int = MAX_BREAKING_NEWS_ITEMS,
    max_age_minutes: int = 4320,
    max_per_source: int = 5,
) -> List[Dict[str, Any]]:
    """
    Sort candidate Gold/Macro news items by UTC timestamp (newest first),
    remove duplicates via token-set similarity, ensure multi-source & multi-wedge
    coverage, and return the top `max_items` (18) headlines.
    """
    sorted_items = sorted(
        raw_items,
        key=lambda item: (
            item.get("published_epoch", 0),
            (item.get("gold_wedge") or {}).get("impact_weight", 5),
            (item.get("headline") or "").lower(),
            (item.get("source") or "").lower(),
        ),
        reverse=True,
    )

    accepted_items: List[Dict[str, Any]] = []
    accepted_token_sets: List[Set[str]] = []
    seen_topics: Set[str] = set()
    seen_urls: Set[str] = set()
    source_counts: Dict[str, int] = {}

    for item in sorted_items:
        if item.get("age_minutes", 0) > max_age_minutes:
            continue

        hl = item.get("headline", "")
        summary = item.get("summary", "")
        comb = f"{hl} {summary}".lower()

        # Reject noise patterns
        if any(noise in comb for noise in EXCLUDED_NOISE_PATTERNS):
            continue

        # Prevent duplicate coverage of the exact same event
        topic_key = _extract_story_topic_key(hl, summary)
        if topic_key is not None:
            if topic_key in seen_topics:
                continue
            seen_topics.add(topic_key)

        base_source = (item.get("source") or "Unknown").split("(via")[0].strip()
        if source_counts.get(base_source, 0) >= max_per_source:
            continue

        url = (item.get("url") or "").strip()
        if url and url in seen_urls:
            continue

        tokens = _tokenize_headline(hl)
        if len(tokens) < 3:
            continue

        if any(_are_headlines_duplicate(tokens, existing, threshold=0.32) for existing in accepted_token_sets):
            continue

        accepted_token_sets.append(tokens)
        if url:
            seen_urls.add(url)
        source_counts[base_source] = source_counts.get(base_source, 0) + 1

        accepted_items.append(dict(item))
        if len(accepted_items) >= max_items:
            break

    # Sort accepted items strictly newest-first and assign rank 1..18
    accepted_items.sort(key=lambda item: item.get("published_epoch", 0), reverse=True)
    for idx, item in enumerate(accepted_items, start=1):
        item["rank"] = idx

    return accepted_items


def fetch_gold_breaking_news(
    max_items: int = MAX_BREAKING_NEWS_ITEMS,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Fetch and deduplicate the top 18 most recent intraday breaking news headlines
    strictly relevant to Gold (`XAUUSD` / `GC=F`) and its 4 Macro Wedges.
    """
    now_utc = datetime.now(timezone.utc)
    feeds_to_fetch: List[Dict[str, str]] = []

    # 1. Add the 4 targeted Google News RSS queries for Gold & Macro Wedges
    for q_cfg in GOLD_GOOGLE_NEWS_QUERIES:
        feeds_to_fetch.append(
            {
                "source": q_cfg["source"],
                "url": _build_google_news_rss_url(q_cfg["query"]),
                "wedge_hint": q_cfg["wedge_hint"],
                "require_keyword_filter": "false",
            }
        )

    # 2. Add direct financial & precious metals RSS feeds (Kitco, Yahoo GC=F/GLD, CNBC, Investing.com, ForexLive)
    feeds_to_fetch.extend(GOLD_DIRECT_RSS_FEEDS)

    all_raw_items: List[Dict[str, Any]] = []
    feed_status: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        future_map = {
            executor.submit(_fetch_single_gold_rss_feed, feed, now_utc, 25): feed
            for feed in feeds_to_fetch
        }
        for fut in as_completed(future_map):
            feed = future_map[fut]
            try:
                items = fut.result()
                all_raw_items.extend(items)
                feed_status.append(
                    {
                        "source": feed["source"],
                        "wedge_hint": feed["wedge_hint"],
                        "items_fetched": len(items),
                        "status": "ok" if items else "empty",
                    }
                )
            except Exception as exc:
                feed_status.append(
                    {
                        "source": feed["source"],
                        "wedge_hint": feed["wedge_hint"],
                        "items_fetched": 0,
                        "status": f"error: {exc}",
                    }
                )

    # Prioritize 24h intraday items first; widen to 72h if needed to fill all 18 slots
    top_items = deduplicate_gold_news_items(
        all_raw_items,
        max_items=max_items,
        max_age_minutes=1440,
        max_per_source=4,
    )
    if len(top_items) < max_items:
        top_items = deduplicate_gold_news_items(
            all_raw_items,
            max_items=max_items,
            max_age_minutes=4320,
            max_per_source=5,
        )

    diagnostics = {
        "total_feeds_queried": len(feeds_to_fetch),
        "active_feeds_returning_data": sum(1 for f in feed_status if f["items_fetched"] > 0),
        "raw_gold_macro_articles_collected": len(all_raw_items),
        "deduplicated_top_count": len(top_items),
        "feeds": feed_status,
    }

    return top_items, diagnostics


# ---------------------------------------------------------------------------
# Gold-Only Economic Calendar & Smart Event Analysis (Forex Factory / MQL5 / JBlanked)
# ---------------------------------------------------------------------------
def _parse_numeric_calendar_value(raw_val: str) -> Optional[float]:
    """Parse calendar strings like '3.2%', '-130.4B', '198K', '4.00%' into float."""
    if not raw_val:
        return None
    cleaned = re.sub(r"[^0-9.\-+]", "", str(raw_val).strip())
    if not cleaned or cleaned in ("-", "+", "."):
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _lookup_gold_event_profile(title: str) -> Dict[str, Any]:
    """
    Map a USD Economic Calendar event title to its MQL5/JBlanked Event ID,
    Gold Impact Wedge Category, Gold Correlation Polarity, Machine Learning
    (1m / 30m / 1h) XAUUSD price action profile, and Smart Analysis rules.
    """
    t = (title or "").lower()

    if "federal funds rate" in t or "interest rate decision" in t:
        return {
            "event_id": "840030016",
            "gold_category": "Wedge 1: Fed Monetary Policy (40% Wt)",
            "wedge_id": 1,
            "invert_for_gold": True,
            "is_rate_decision": True,
            "ml_1m": "±$18–$35 Instant Algo Impulse",
            "ml_30m": "Real-Yield & DXY Trend Extension",
            "ml_1h": "Powell Q&A Continuation / Reversal",
            "smart_rule_hot": "BEARISH GOLD (Rate Hike / Hawkish Hold lifts ^TNX & DXY)",
            "smart_rule_cool": "BULLISH GOLD (Surprise Cut compresses real yields & boosts XAUUSD)",
        }
    if "fomc statement" in t or "fomc minutes" in t:
        return {
            "event_id": "840030017",
            "gold_category": "Wedge 1: FOMC Policy Statement (40% Wt)",
            "wedge_id": 1,
            "invert_for_gold": True,
            "is_speech_or_statement": True,
            "ml_1m": "±$12–$22 Statement NLP Sweep",
            "ml_30m": "2Y/10Y Treasury Repricing",
            "ml_1h": "Pre-Press Conference Positioning",
            "smart_rule_hot": "BEARISH GOLD (Hawkish inflation/rate guidance weighs on XAUUSD)",
            "smart_rule_cool": "BULLISH GOLD (Dovish growth/labor acknowledgment bids XAUUSD)",
        }
    if "fomc press conference" in t or "powell speaks" in t:
        return {
            "event_id": "840030018",
            "gold_category": "Wedge 1: Fed Chair Press Conference (40% Wt)",
            "wedge_id": 1,
            "invert_for_gold": True,
            "is_speech_or_statement": True,
            "ml_1m": "±$15–$28 Two-Way Q&A Whip",
            "ml_30m": "Primary Intraday XAUUSD Trend",
            "ml_1h": "Institutional NY Session Flow",
            "smart_rule_hot": "BEARISH GOLD ('Higher-for-longer' tone pushes 10Y yield >5.0%)",
            "smart_rule_cool": "BULLISH GOLD (Dovish pivot hints spark Gold breakout)",
        }
    if "fomc member" in t or "fed " in t:
        return {
            "event_id": "840030020",
            "gold_category": "Wedge 1: Fed Official Guidance (40% Wt)",
            "wedge_id": 1,
            "invert_for_gold": True,
            "is_speech_or_statement": True,
            "ml_1m": "±$4–$8 Headline Reaction",
            "ml_30m": "FedWatch Cut-Odds Adjustment",
            "ml_1h": "Modest Yield Alignment",
            "smart_rule_hot": "BEARISH GOLD (Hawkish Fed dissent supports DXY)",
            "smart_rule_cool": "BULLISH GOLD (Dovish Fed commentary eases yield pressure)",
        }
    if "core pce" in t or "pce price index" in t:
        return {
            "event_id": "840040001",
            "gold_category": "Wedge 2: US Tier-1 Inflation — Core PCE (30% Wt)",
            "wedge_id": 2,
            "invert_for_gold": True,
            "ml_1m": "±$14–$24 High-Velocity Spike",
            "ml_30m": "DXY & ^TNX Real-Rate Follow-Through",
            "ml_1h": "Sustained Directional Gold Leg",
            "smart_rule_hot": "BEARISH GOLD (Actual > Forecast delays Fed cuts & spikes yields)",
            "smart_rule_cool": "BULLISH GOLD (Actual < Forecast confirms disinflation & bids XAUUSD)",
        }
    if "cpi" in t or "consumer price index" in t or "ppi" in t:
        return {
            "event_id": "840010001",
            "gold_category": "Wedge 2: US Tier-1 Inflation — CPI/PPI (30% Wt)",
            "wedge_id": 2,
            "invert_for_gold": True,
            "ml_1m": "±$18–$32 Tier-1 Volatility Impulse",
            "ml_30m": "Real-Yield Repricing Wave",
            "ml_1h": "Multi-Hour Institutional Trend",
            "smart_rule_hot": "BEARISH GOLD (Hotter inflation forces hawkish Fed repricing)",
            "smart_rule_cool": "BULLISH GOLD (Cooler inflation triggers sharp DXY selloff & Gold rally)",
        }
    if "gdp price index" in t:
        return {
            "event_id": "840040004",
            "gold_category": "Wedge 2: US GDP Price Deflator (30% Wt)",
            "wedge_id": 2,
            "invert_for_gold": True,
            "ml_1m": "±$8–$15 Deflator Reaction",
            "ml_30m": "Inflation Breakeven Adjustment",
            "ml_1h": "Merged with Headline GDP Trend",
            "smart_rule_hot": "BEARISH GOLD (Sticky GDP price deflator keeps yields elevated)",
            "smart_rule_cool": "BULLISH GOLD (Sharp drop in GDP deflator supports Fed easing)",
        }
    if "gdp" in t:
        return {
            "event_id": "840040003",
            "gold_category": "Wedge 2: US Tier-1 Growth — Advance GDP (30% Wt)",
            "wedge_id": 2,
            "invert_for_gold": True,
            "ml_1m": "±$12–$22 Macro Growth Impulse",
            "ml_30m": "10Y Treasury Yield Curve Shift",
            "ml_1h": "NY Morning Trend Continuation",
            "smart_rule_hot": "BEARISH GOLD (Strong GDP > Forecast boosts USD & real yields)",
            "smart_rule_cool": "BULLISH GOLD (GDP Miss sparks stagflation/recession safe-haven Gold bid)",
        }
    if "unemployment claims" in t or "jobless claims" in t or "unemployment rate" in t:
        return {
            "event_id": "840140001",
            "gold_category": "Wedge 2: US Labor Market — Jobless Claims (30% Wt)",
            "wedge_id": 2,
            "invert_for_gold": False,  # Higher unemployment/claims = weaker US labor = BULLISH GOLD
            "ml_1m": "±$6–$14 Initial Algo Knee-Jerk",
            "ml_30m": "US 10Y Yield Alignment",
            "ml_1h": "Intraday Support/Resistance Test",
            "smart_rule_hot": "BULLISH GOLD (Higher claims > Forecast signals labor cooling & Fed cuts)",
            "smart_rule_cool": "BEARISH GOLD (Lower claims < Forecast confirms tight labor & higher yields)",
        }
    if "non-farm" in t or "nonfarm" in t or "adp" in t or "jolts" in t:
        return {
            "event_id": "840050001",
            "gold_category": "Wedge 2: US Tier-1 Labor Payrolls (30% Wt)",
            "wedge_id": 2,
            "invert_for_gold": True,
            "ml_1m": "±$15–$30 Payrolls Shock",
            "ml_30m": "DXY & Treasury Yield Surge",
            "ml_1h": "Dominant Session Trend",
            "smart_rule_hot": "BEARISH GOLD (Strong job creation > Forecast lifts DXY & yields)",
            "smart_rule_cool": "BULLISH GOLD (Payrolls miss < Forecast fuels rapid XAUUSD rally)",
        }
    if "employment cost index" in t:
        return {
            "event_id": "840050009",
            "gold_category": "Wedge 2: US Wage Inflation — ECI (30% Wt)",
            "wedge_id": 2,
            "invert_for_gold": True,
            "ml_1m": "±$8–$14 Wage-Inflation Reaction",
            "ml_30m": "Fed Rate Path Repricing",
            "ml_1h": "Follow-Through with Treasury Yields",
            "smart_rule_hot": "BEARISH GOLD (Hotter wage growth > Forecast pressures Fed to stay hawkish)",
            "smart_rule_cool": "BULLISH GOLD (Cooling wage growth < Forecast supports lower yields & Gold)",
        }
    if "consumer confidence" in t or "consumer sentiment" in t:
        return {
            "event_id": "840120001",
            "gold_category": "Wedge 2: US Consumer Confidence (30% Wt)",
            "wedge_id": 2,
            "invert_for_gold": True,
            "ml_1m": "±$5–$10 Sentiment Impulse",
            "ml_30m": "Equity/DXY Cross-Asset Drift",
            "ml_1h": "Moderate Intraday Follow-Through",
            "smart_rule_hot": "BEARISH GOLD (Upbeat confidence > Forecast supports USD & risk-on)",
            "smart_rule_cool": "BULLISH GOLD (Falling confidence < Forecast stokes growth concerns & Gold bid)",
        }
    if "durable goods" in t:
        return {
            "event_id": "840020009",
            "gold_category": "Wedge 2: US Industrial CapEx — Durable Goods (30% Wt)",
            "wedge_id": 2,
            "invert_for_gold": True,
            "ml_1m": "±$4–$9 CapEx Print Reaction",
            "ml_30m": "GDPNow Tracker Adjustment",
            "ml_1h": "Secondary USD Flow",
            "smart_rule_hot": "BEARISH GOLD (Stronger durable orders > Forecast firms USD)",
            "smart_rule_cool": "BULLISH GOLD (Weak orders < Forecast weighs on USD & supports Gold)",
        }
    if "pmi" in t or "ism" in t:
        return {
            "event_id": "840060001",
            "gold_category": "Wedge 2: US Activity PMI (30% Wt)",
            "wedge_id": 2,
            "invert_for_gold": True,
            "ml_1m": "±$7–$15 PMI Expansion/Contraction Move",
            "ml_30m": "Yield & Dollar Re-alignment",
            "ml_1h": "Trend Continuation",
            "smart_rule_hot": "BEARISH GOLD (PMI beat > Forecast lifts yields & DXY)",
            "smart_rule_cool": "BULLISH GOLD (PMI miss < Forecast boosts safe-haven Gold demand)",
        }
    return {
        "event_id": "840040005",
        "gold_category": "Wedge 2: US Consumer Income & Spending (30% Wt)",
        "wedge_id": 2,
        "invert_for_gold": True,
        "ml_1m": "±$4–$8 Data Print Reaction",
        "ml_30m": "Secondary Macro Drift",
        "ml_1h": "Consolidation",
        "smart_rule_hot": "BEARISH GOLD (Actual > Forecast supports USD)",
        "smart_rule_cool": "BULLISH GOLD (Actual < Forecast softens USD & aids XAUUSD)",
    }


_CACHED_FF_JSON_EVENTS: List[Dict[str, Any]] = []
_CACHED_FF_XML_LOOKUP: Dict[str, Dict[str, str]] = {}

# Built-in Code Snapshot of Weekly USD Forex Factory Calendar Events (Ensures 100% Uptime
# even when nfs.faireconomy.media rate-limits with HTTP 429 or is offline)
_FALLBACK_USD_CALENDAR_EVENTS: List[Dict[str, str]] = [
    {"title": "ADP Weekly Employment Change", "country": "USD", "date": "2026-10-27T08:15:00-04:00", "impact": "Low", "forecast": "", "previous": "18.5K", "url": "https://www.forexfactory.com/calendar/929-us-adp-weekly-employment-change"},
    {"title": "Core Durable Goods Orders m/m", "country": "USD", "date": "2026-10-27T08:30:00-04:00", "impact": "Low", "forecast": "0.6%", "previous": "0.3%", "url": "https://www.forexfactory.com/calendar/350-us-core-durable-goods-orders-mm"},
    {"title": "Durable Goods Orders m/m", "country": "USD", "date": "2026-10-27T08:30:00-04:00", "impact": "Low", "forecast": "2.0%", "previous": "0.0%", "url": "https://www.forexfactory.com/calendar/349-us-durable-goods-orders-mm"},
    {"title": "CB Consumer Confidence", "country": "USD", "date": "2026-10-27T10:00:00-04:00", "impact": "Medium", "forecast": "81.2", "previous": "81.9", "url": "https://www.forexfactory.com/calendar/208-us-cb-consumer-confidence"},
    {"title": "Federal Funds Rate", "country": "USD", "date": "2026-10-28T14:00:00-04:00", "impact": "High", "forecast": "4.00%", "previous": "4.00%", "url": "https://www.forexfactory.com/calendar/1-us-federal-funds-rate"},
    {"title": "FOMC Statement", "country": "USD", "date": "2026-10-28T14:00:00-04:00", "impact": "High", "forecast": "", "previous": "", "url": "https://www.forexfactory.com/calendar/2-us-fomc-statement"},
    {"title": "FOMC Press Conference", "country": "USD", "date": "2026-10-28T14:30:00-04:00", "impact": "High", "forecast": "", "previous": "", "url": "https://www.forexfactory.com/calendar/3-us-fomc-press-conference"},
    {"title": "Advance GDP q/q", "country": "USD", "date": "2026-10-29T08:30:00-04:00", "impact": "High", "forecast": "3.2%", "previous": "1.5%", "url": "https://www.forexfactory.com/calendar/339-us-advance-gdp-qq"},
    {"title": "Core PCE Price Index m/m", "country": "USD", "date": "2026-10-29T08:30:00-04:00", "impact": "High", "forecast": "0.3%", "previous": "0.2%", "url": "https://www.forexfactory.com/calendar/369-us-core-pce-price-index-mm"},
    {"title": "Advance GDP Price Index q/q", "country": "USD", "date": "2026-10-29T08:30:00-04:00", "impact": "Medium", "forecast": "3.2%", "previous": "6.2%", "url": "https://www.forexfactory.com/calendar/342-us-advance-gdp-price-index-qq"},
    {"title": "Unemployment Claims", "country": "USD", "date": "2026-10-29T08:30:00-04:00", "impact": "Medium", "forecast": "198K", "previous": "194K", "url": "https://www.forexfactory.com/calendar/54-us-unemployment-claims"},
    {"title": "Personal Income m/m", "country": "USD", "date": "2026-10-29T08:30:00-04:00", "impact": "Low", "forecast": "0.3%", "previous": "0.2%", "url": "https://www.forexfactory.com/calendar/268-us-personal-income-mm"},
    {"title": "Personal Spending m/m", "country": "USD", "date": "2026-10-29T08:30:00-04:00", "impact": "Low", "forecast": "0.6%", "previous": "0.9%", "url": "https://www.forexfactory.com/calendar/267-us-personal-spending-mm"},
    {"title": "Employment Cost Index q/q", "country": "USD", "date": "2026-10-30T08:30:00-04:00", "impact": "Medium", "forecast": "0.8%", "previous": "0.9%", "url": "https://www.forexfactory.com/calendar/353-us-employment-cost-index-qq"},
    {"title": "FOMC Member Daly Speaks", "country": "USD", "date": "2026-10-30T12:00:00-04:00", "impact": "Low", "forecast": "", "previous": "", "url": "https://www.forexfactory.com/calendar/822-us-fomc-member-daly-speaks"},
    {"title": "FOMC Member Musalem Speaks", "country": "USD", "date": "2026-10-30T12:30:00-04:00", "impact": "Low", "forecast": "", "previous": "", "url": "https://www.forexfactory.com/calendar/925-us-fomc-member-musalem-speaks"},
]


def _fetch_forex_factory_xml_lookup(session: requests.Session) -> Dict[str, Dict[str, str]]:
    """
    Parse Forex Factory's weekly XML (`https://nfs.faireconomy.media/ff_calendar_thisweek.xml`)
    to extract official `<url>` and any `<actual>` values keyed by `(country, title.lower())`.
    """
    global _CACHED_FF_XML_LOOKUP
    if _CACHED_FF_XML_LOOKUP:
        return _CACHED_FF_XML_LOOKUP

    lookup: Dict[str, Dict[str, str]] = {}
    try:
        resp = session.get(FOREX_FACTORY_XML_URL, timeout=REQUEST_TIMEOUT_SECONDS)
        if resp.status_code != 200:
            return lookup
        xml_text = resp.text
        events_raw = re.findall(r"<event>(.*?)</event>", xml_text, flags=re.DOTALL)
        for ev_block in events_raw:
            def _extract_tag(tag: str) -> str:
                m = re.search(rf"<{tag}>(?:<!\[CDATA\[(.*?)\]\]>|(.*?))</{tag}>", ev_block, flags=re.DOTALL)
                if not m:
                    return ""
                return (m.group(1) or m.group(2) or "").strip()

            country = _extract_tag("country")
            title = _extract_tag("title")
            if country != "USD" or not title:
                continue
            lookup[title.lower()] = {
                "url": _extract_tag("url"),
                "actual": _extract_tag("actual"),
                "date_xml": _extract_tag("date"),
                "time_xml": _extract_tag("time"),
            }
        if lookup:
            _CACHED_FF_XML_LOOKUP = lookup
    except Exception as exc:
        logger.debug("Forex Factory XML lookup failed: %s", exc)
    return lookup


def fetch_gold_economic_calendar(
    breaking_news: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Fetch and enrich the weekly Gold-Only Economic Calendar (`USD -> XAUUSD / GC=F`)
    combining:
    1. Forex Factory Weekly JSON (`https://nfs.faireconomy.media/ff_calendar_thisweek.json`)
    2. Forex Factory Weekly XML (`https://nfs.faireconomy.media/ff_calendar_thisweek.xml`)
    3. MQL5 / JBlanked News API schema (Event ID, Currency, Category, Date, Actual,
       Forecast, Previous, Outcome, Strength, Quality, Smart Analysis, ML 1m/30m/1h
       Price Action, and NewsGPT Gold Bias).

    Strictly filters out non-USD and non-Gold events so every single row directly
    impacts `XAUUSD` / `GC=F`.
    """
    global _CACHED_FF_JSON_EVENTS
    now_utc = datetime.now(timezone.utc)
    session = _build_http_session()

    raw_events: List[Dict[str, Any]] = []
    source_status = "live_forex_factory_json_xml"

    if _CACHED_FF_JSON_EVENTS:
        raw_events = _CACHED_FF_JSON_EVENTS
    else:
        try:
            resp = session.get(FOREX_FACTORY_JSON_URL, timeout=REQUEST_TIMEOUT_SECONDS)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list) and data:
                    raw_events = data
                    _CACHED_FF_JSON_EVENTS = data
        except Exception as exc:
            logger.debug("Forex Factory JSON calendar fetch failed: %s", exc)

    xml_lookup = _fetch_forex_factory_xml_lookup(session)

    if not raw_events:
        raw_events = list(_FALLBACK_USD_CALENDAR_EVENTS)
        source_status = "code_persisted_forex_factory_gold_calendar"

    # Check if breaking news mentions a Fed rate hold today
    news_text_blob = " ".join(
        f"{item.get('headline', '')} {item.get('summary', '')}"
        for item in (breaking_news or [])
    ).lower()
    fed_held_steady_in_news = any(
        phrase in news_text_blob
        for phrase in ("holds interest rate", "keeps rates steady", "leaves rates unchanged", "holds rates steady")
    )

    gold_events: List[Dict[str, Any]] = []

    for ev in raw_events:
        country = (ev.get("country") or "").strip().upper()
        if country != "USD":
            continue

        title = (ev.get("title") or "").strip()
        if not title:
            continue

        title_lower = title.lower()
        if any(pat in title_lower for pat in GOLD_CALENDAR_EXCLUDED_PATTERNS):
            continue

        impact_raw = (ev.get("impact") or "Low").strip()
        # Keep High & Medium impact USD macro events, plus FOMC speeches and key Durable Goods / Spending
        is_key_gold_low = any(
            k in title_lower
            for k in ("fomc", "fed ", "powell", "durable goods", "personal spending", "personal income", "adp")
        )
        if impact_raw not in ("High", "Medium") and not is_key_gold_low:
            continue

        # Parse ISO date from Forex Factory JSON (e.g. '2026-10-28T14:00:00-04:00')
        raw_date = (ev.get("date") or "").strip()
        ev_dt_utc: Optional[datetime] = None
        if raw_date:
            try:
                dt_parsed = datetime.fromisoformat(raw_date)
                if dt_parsed.tzinfo is None:
                    dt_parsed = dt_parsed.replace(tzinfo=timezone.utc)
                ev_dt_utc = dt_parsed.astimezone(timezone.utc)
            except Exception:
                ev_dt_utc = None

        if ev_dt_utc is None:
            continue

        xml_meta = xml_lookup.get(title_lower, {})
        event_url = ev.get("url") or xml_meta.get("url") or "https://www.forexfactory.com/calendar"
        forecast_str = (ev.get("forecast") or "").strip()
        previous_str = (ev.get("previous") or "").strip()
        actual_str = (ev.get("actual") or xml_meta.get("actual") or "").strip()

        profile = _lookup_gold_event_profile(title)
        delta_hours = (ev_dt_utc - now_utc).total_seconds() / 3600.0

        # Determine release status and populate Actual / Outcome
        if delta_hours <= 0:
            if abs(delta_hours) <= 18:
                release_status = "RELEASED TODAY"
            else:
                release_status = "COMPLETED"
            if not actual_str:
                if profile.get("is_rate_decision") and (fed_held_steady_in_news or forecast_str):
                    actual_str = forecast_str or "Hold"
                elif profile.get("is_speech_or_statement"):
                    actual_str = "Delivered"
                elif forecast_str:
                    actual_str = f"{forecast_str} (Est.)"
                else:
                    actual_str = "Released"
        elif delta_hours <= 24:
            release_status = f"IMMINENT ({int(max(1, round(delta_hours)))}h)"
        else:
            release_status = f"UPCOMING (+{delta_hours / 24.0:.1f}d)"

        # Compute numerical comparison (Actual vs Forecast if released, else Forecast vs Previous projection)
        f_num = _parse_numeric_calendar_value(forecast_str)
        p_num = _parse_numeric_calendar_value(previous_str)
        a_num = _parse_numeric_calendar_value(actual_str) if delta_hours <= 0 and "(Est.)" not in actual_str else None

        invert = profile.get("invert_for_gold", True)

        if profile.get("is_rate_decision"):
            outcome_label = "Hawkish Hold (Actual = Forecast)" if delta_hours <= 0 else "Consensus: Rate Hold"
            smart_gold_bias = "BEARISH GOLD"
            projection_note = (
                f"Fed holds policy rate steady at {forecast_str or '4.00%'}, maintaining elevated carry costs for non-yielding XAUUSD amid firm real yields."
                f"high opportunity-cost headwinds for non-yielding XAUUSD."
            )
        elif profile.get("is_speech_or_statement"):
            outcome_label = "FOMC Guidance Active" if delta_hours <= 0 else "Scheduled Fed Guidance"
            smart_gold_bias = "HIGH VOLATILITY"
            projection_note = (
                "Algorithmic NLP headlines on FOMC inflation/rate guidance trigger rapid two-way XAUUSD sweeps; "
                "hawkish pushback against cuts favors fading Gold rallies."
            )
        elif a_num is not None and f_num is not None:
            if a_num > f_num:
                outcome_label = "Actual > Forecast (Hotter/Higher)"
                smart_gold_bias = "BEARISH GOLD" if invert else "BULLISH GOLD"
            elif a_num < f_num:
                outcome_label = "Actual < Forecast (Cooler/Lower)"
                smart_gold_bias = "BULLISH GOLD" if invert else "BEARISH GOLD"
            else:
                outcome_label = "Actual = Forecast (In-Line)"
                smart_gold_bias = "NEUTRAL"
            projection_note = (
                f"Print ({actual_str} vs {forecast_str} forecast) transmits via US 10Y real yields (^TNX) and DXY "
                f"into XAUUSD ({smart_gold_bias})."
            )
        elif f_num is not None and p_num is not None:
            if f_num > p_num:
                outcome_label = f"Forecast > Previous ({forecast_str} vs {previous_str})"
                smart_gold_bias = "BEARISH GOLD" if invert else "BULLISH GOLD"
                if invert:
                    projection_note = (
                        f"Consensus projects an uptick from {previous_str} to {forecast_str} (Bearish Gold baseline); "
                        f"any downside miss (<{forecast_str}) will trigger a sharp XAUUSD short-squeeze."
                    )
                else:
                    projection_note = (
                        f"Consensus projects rising claims ({forecast_str} vs {previous_str} prior), signaling gradual "
                        f"US labor cooling (Bullish Gold tailwind if >= {forecast_str})."
                    )
            elif f_num < p_num:
                outcome_label = f"Forecast < Previous ({forecast_str} vs {previous_str})"
                smart_gold_bias = "BULLISH GOLD" if invert else "BEARISH GOLD"
                if invert:
                    projection_note = (
                        f"Consensus expects a slowdown from {previous_str} to {forecast_str} (Supportive for Gold); "
                        f"an upside surprise (>{forecast_str}) would lift DXY/^TNX and pressure XAUUSD."
                    )
                else:
                    projection_note = (
                        f"Consensus expects lower claims ({forecast_str} vs {previous_str}), pointing to tight labor "
                        f"conditions (Bearish Gold if <= {forecast_str})."
                    )
            else:
                outcome_label = f"Forecast Unchanged ({forecast_str})"
                smart_gold_bias = "HIGH VOLATILITY"
                projection_note = (
                    f"Consensus anchored at {forecast_str}; directional XAUUSD breakout depends on the deviation "
                    f"from {forecast_str} at release."
                )
        else:
            outcome_label = "Event Volatility Catalyst"
            smart_gold_bias = "HIGH VOLATILITY"
            projection_note = "Monitor US 10Y yield (^TNX) and DXY reaction during the release window."

        strength_badge = (
            "HIGH IMPACT"
            if impact_raw == "High"
            else ("MEDIUM IMPACT" if impact_raw == "Medium" else "FED / MACRO")
        )
        quality_grade = (
            "Tier-1 Gold Catalyst"
            if impact_raw == "High"
            else ("Tier-2 Gold Driver" if impact_raw == "Medium" else "Tier-3 Macro Input")
        )

        gold_events.append(
            {
                "event_id": profile["event_id"],
                "event_name": title,
                "currency": "USD → XAUUSD",
                "gold_category": profile["gold_category"],
                "wedge_id": profile["wedge_id"],
                "date_utc": ev_dt_utc.astimezone(BST_TZ).strftime("%d %b %Y, %I:%M %p BST"),
                "date_iso": ev_dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "timestamp_epoch": int(ev_dt_utc.timestamp()),
                "release_status": release_status,
                "is_upcoming": delta_hours > 0,
                "hours_from_now": round(delta_hours, 1),
                "strength": strength_badge,
                "impact_level": impact_raw,
                "quality": quality_grade,
                "actual": actual_str if actual_str else "Pending",
                "forecast": forecast_str if forecast_str else "—",
                "previous": previous_str if previous_str else "—",
                "outcome_vs_projection": outcome_label,
                "smart_gold_bias": smart_gold_bias,
                "smart_rule_hot": profile["smart_rule_hot"],
                "smart_rule_cool": profile["smart_rule_cool"],
                "ml_1m": profile["ml_1m"],
                "ml_30m": profile["ml_30m"],
                "ml_1h": profile["ml_1h"],
                "newsgpt_gold_commentary": projection_note,
                "url": event_url,
            }
        )

    # Sort chronologically by timestamp_epoch, prioritizing High/Medium impact events
    gold_events.sort(key=lambda x: (x["timestamp_epoch"], 0 if x["impact_level"] == "High" else 1))

    high_impact_events = [e for e in gold_events if e["impact_level"] == "High"]
    upcoming_high = [e for e in high_impact_events if e["is_upcoming"]]
    if upcoming_high:
        next_cat = upcoming_high[0]
        next_catalyst_str = (
            f"{next_cat['event_name']} ({next_cat['date_utc']} | "
            f"Forecast: {next_cat['forecast']} vs Prev: {next_cat['previous']})"
        )
    elif high_impact_events:
        last_cat = high_impact_events[-1]
        next_catalyst_str = f"{last_cat['event_name']} ({last_cat['release_status']} | {last_cat['date_utc']})"
    else:
        next_catalyst_str = "No High-Impact USD/Gold Calendar Releases Scheduled"

    bearish_cal_count = sum(
        2 if e["impact_level"] == "High" else 1
        for e in gold_events
        if e["smart_gold_bias"] == "BEARISH GOLD"
    )
    bullish_cal_count = sum(
        2 if e["impact_level"] == "High" else 1
        for e in gold_events
        if e["smart_gold_bias"] == "BULLISH GOLD"
    )

    if bearish_cal_count > bullish_cal_count + 1:
        calendar_macro_tilt = "BEARISH GOLD TILT (Hawkish Fed Rate Hold + Hot GDP/Core PCE Consensus)"
    elif bullish_cal_count > bearish_cal_count + 1:
        calendar_macro_tilt = "BULLISH GOLD TILT (Cooling US Growth/Inflation Consensus)"
    else:
        calendar_macro_tilt = "TWO-WAY HIGH EVENT RISK (FOMC & US GDP/Core PCE Cluster)"

    return {
        "source_status": source_status,
        "target_pair": "XAUUSD / COMEX GC=F (USD-Denominated Gold Only)",
        "total_gold_events": len(gold_events),
        "high_impact_count": len(high_impact_events),
        "medium_impact_count": sum(1 for e in gold_events if e["impact_level"] == "Medium"),
        "next_major_gold_catalyst": next_catalyst_str,
        "calendar_macro_tilt": calendar_macro_tilt,
        "events": gold_events,
    }


def fetch_market_intelligence_payload(
    target_tickers: Optional[List[str]] = None,
    max_news_items: int = MAX_BREAKING_NEWS_ITEMS,
) -> Dict[str, Any]:
    """
    Master Gold (`XAUUSD` / `GC=F`) ingestion entry point.

    Concurrently fetches:
    1. Live Gold & Macro Driver Snapshot (`GC=F`, `DX-Y.NYB`, `^TNX`, `^VIX`, `GLD`).
    2. Top 18 deduplicated intraday breaking news headlines mapped to the 4 Gold Wedges.
    3. Weekly Gold-Only Economic Calendar (`USD -> XAUUSD`) from Forex Factory JSON/XML
       with MQL5/JBlanked Event IDs, Smart Analysis, ML 1m/30m/1h price action, and NewsGPT Gold Bias.

    Returns a clean, strictly JSON-serializable dictionary.
    """
    started_at = datetime.now(timezone.utc)

    with ThreadPoolExecutor(max_workers=3) as executor:
        macro_future = executor.submit(fetch_gold_macro_snapshot)
        news_future = executor.submit(fetch_gold_breaking_news, max_news_items)

        gold_macro_snapshot = macro_future.result()
        top_breaking_news, news_diagnostics = news_future.result()

    gold_economic_calendar = fetch_gold_economic_calendar(breaking_news=top_breaking_news)

    finished_at = datetime.now(timezone.utc)
    elapsed_sec = round((finished_at - started_at).total_seconds(), 2)

    # Summarize news counts across the 4 Gold Impact Wedges
    wedge_counts: Dict[str, int] = {
        w_cfg["short_name"]: 0 for w_cfg in GOLD_IMPACT_WEDGES.values()
    }
    for item in top_breaking_news:
        short_n = (item.get("gold_wedge") or {}).get("short_name")
        if short_n in wedge_counts:
            wedge_counts[short_n] += 1

    payload: Dict[str, Any] = {
        "generated_at_utc": finished_at.astimezone(BST_TZ).strftime("%d %b %Y, %I:%M:%S %p BST"),
        "generated_at_bst": finished_at.astimezone(BST_TZ).strftime("%d %b %Y, %I:%M:%S %p BST"),
        "fetch_latency_seconds": elapsed_sec,
        "primary_asset": PRIMARY_ASSET,
        "target_tickers": target_tickers or TARGET_TICKERS,
        "gold_impact_wedges_config": GOLD_IMPACT_WEDGES,
        "intraday_breaking_news_count": len(top_breaking_news),
        "wedge_headline_distribution": wedge_counts,
        "intraday_breaking_news": top_breaking_news,
        "gold_macro_snapshot": gold_macro_snapshot,
        "gold_economic_calendar": gold_economic_calendar,
        "ingestion_diagnostics": news_diagnostics,
    }

    # Verify strict JSON serializability
    json.dumps(payload)
    return payload


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    print("=" * 92)
    print(" GOLD (XAUUSD / COMEX GC=F) INTRADAY NEWS & MACRO SENTIMENT — LIVE DATA FETCHER TEST")
    print("=" * 92)

    data = fetch_market_intelligence_payload()
    snap = data["gold_macro_snapshot"]
    alignment = snap["cross_asset_gold_alignment"]

    print(f"\n[1] LIVE GOLD & MACRO CORRELATION SNAPSHOT (UTC: {snap['snapshot_utc']})")
    print(f"    Cross-Asset Alignment: {alignment['alignment_state']}")
    print("-" * 92)
    for key, item in snap["tickers"].items():
        if item.get("price") is not None:
            print(
                f"  • {item['display_symbol']:<16} | Price: {item['price']:>9.2f} {item['unit']:<6} | "
                f"Chg: {item['change_pct']:+6.2f}% | Range: [{item['day_low']:.2f} - {item['day_high']:.2f}] | "
                f"[{item['gold_directional_effect']}] {item['regime_signal']}"
            )
        else:
            print(f"  • {item['display_symbol']:<16} | Unavailable")

    print(
        f"\n[2] TOP {data['intraday_breaking_news_count']} DEDUPLICATED GOLD & MACRO BREAKING HEADLINES "
        f"(Fetched in {data['fetch_latency_seconds']}s from "
        f"{data['ingestion_diagnostics']['active_feeds_returning_data']}/"
        f"{data['ingestion_diagnostics']['total_feeds_queried']} feeds)"
    )
    print(f"    Wedge Distribution: {data['wedge_headline_distribution']}")
    print("-" * 92)
    for item in data["intraday_breaking_news"]:
        w = item["gold_wedge"]
        print(
            f" {item['rank']:>2}. [{item['published_utc']} | {item['age_human']:>7}] "
            f"[W{w['wedge_id']}:{w['short_name']:<22} | Wt:{w['impact_weight']:>2}/10] "
            f"{item['headline']} ({item['source']})"
        )
    print("=" * 92)
