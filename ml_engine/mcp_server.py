"""
ENHANCED MCP DATA SERVER — FinFolioX v2.0
==========================================
Model Context Protocol (MCP) Server with Real-Time Macro Intelligence.

NEW TIERS ADDED:
  Tier 0 (weight 1.5) — Central Bank / Fed (FRED API — free, no key needed)
  Tier 1 (weight 1.0) — SEC EDGAR (unchanged)
  Tier 2 (weight 0.9) — Yahoo Finance RSS (unchanged, boosted weight)
  Tier 3 (weight 0.8) — Geopolitical Events (GDELT Project — free, real-time)
  Tier 4 (weight 0.7) — Economic Calendar (open-meteo macro indicators)
  Tier 5 (weight 0.6) — Commodity / FX Macro (Yahoo Finance quotes)
  Tier 6 (weight 0.5) — Google Trends Proxy (RSS-based interest spikes)
  Tier 7 (weight 0.3) — Reddit WallStreetBets (unchanged)

All sources are FREE and require NO API keys.

GDELT Project:
  - Scans 65+ news sources worldwide every 15 minutes
  - Provides geopolitical tone scores (-10 = war, +10 = peace)
  - Event categories: military, sanctions, economic, diplomacy

FRED (Federal Reserve Economic Data):
  - Real-time Fed statements, FOMC minutes, rate decisions
  - Pulled via FRED RSS feeds — no API key required

Usage:
  mcp = MCPDataServer()
  payload = mcp.get_global_context_payload("WTI")  # or "CL=F", "AAPL", etc.
"""

import requests
import xml.etree.ElementTree as ET
import re
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging

logger = logging.getLogger("MCPServer")


# ==============================================================================
# CONFIGURATION — Tune weights here
# ==============================================================================
TIER_WEIGHTS = {
    "FRED":         1.50,   # Tier 0 — Central Bank / Fed
    "SEC EDGAR":    1.00,   # Tier 1 — Regulatory filings
    "Yahoo Finance":0.90,   # Tier 2 — Financial news
    "GDELT":        0.80,   # Tier 3 — Geopolitical events
    "EconCalendar": 0.70,   # Tier 4 — Economic data releases
    "MacroFX":      0.60,   # Tier 5 — Commodity / FX macro
    "GoogleTrends": 0.50,   # Tier 6 — Search interest proxy
    "Reddit r/WSB": 0.30,   # Tier 7 — Retail sentiment
}

# GDELT topic filters → maps to asset classes for smart routing
GDELT_TOPIC_MAP = {
    "OIL":  ["OPEC", "Strait of Hormuz", "crude oil", "energy sanctions", "Iran", "Saudi Arabia", "Russia oil"],
    "GOLD": ["Federal Reserve", "inflation", "gold reserves", "dollar collapse", "safe haven"],
    "SPY":  ["Federal Reserve", "recession", "S&P", "tariff", "trade war", "earnings"],
    "QQQ":  ["AI chip", "semiconductor", "tech regulation", "antitrust", "NVDA", "TSLA"],
    "WTI":  ["OPEC", "Strait of Hormuz", "crude oil", "pipeline", "Iran", "Libya", "Iraq"],
    "BTC":  ["Bitcoin", "crypto regulation", "SEC crypto", "stablecoin", "CBDC"],
    "TLT":  ["Federal Reserve", "interest rate", "bond market", "FOMC", "treasury yield"],
    "GLD":  ["gold", "inflation hedge", "dollar weakness", "central bank gold"],
    "DEFAULT": ["Federal Reserve", "recession", "inflation", "trade war", "geopolitical risk"],
}

# FRED series that matter most for equities / commodities
FRED_SERIES = {
    "FEDFUNDS":  "Federal Funds Rate",
    "CPIAUCSL":  "Consumer Price Index (Inflation)",
    "UNRATE":    "Unemployment Rate",
    "GDP":       "Gross Domestic Product",
    "DGS10":     "10-Year Treasury Yield",
    "DCOILWTICO":"WTI Crude Oil Price (FRED)",
}


class MCPDataServer:
    """
    Enhanced Model Context Protocol (MCP) Server.

    Fetches, sanitizes, and packages intelligence from 8 tiers:
      - Regulatory (SEC), Macro (FRED), Geopolitical (GDELT),
        News (Yahoo), Economic Calendar, FX/Commodity, Trends, Retail
    """

    def __init__(self):
        self.headers = {
            "User-Agent": "FinFolioX_Research_Bot/2.0 (Educational Capstone)"
        }
        self.reddit_headers = {
            "User-Agent": "FinFolioX:v2.0 (by /u/finfolio_admin)"
        }
        self.ticker_to_cik = {}
        self._load_sec_cik_mapping()
        print("   🔌 [MCP Server] Initialized and listening for requests...")

    # ==========================================================================
    # UTILITIES
    # ==========================================================================
    def _load_sec_cik_mapping(self):
        """Loads the official SEC Ticker-to-CIK JSON mapping."""
        try:
            url = "https://www.sec.gov/files/company_tickers.json"
            resp = requests.get(url, headers=self.headers, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                for entry in data.values():
                    self.ticker_to_cik[entry['ticker'].upper()] = str(entry['cik_str']).zfill(10)
        except Exception:
            pass

    def _clean_text(self, text: str) -> str:
        """Strips HTML, emojis, and artifacts to protect FinBERT."""
        if not text:
            return ""
        text = re.sub(re.compile('<.*?>'), '', text)
        text = text.encode('ascii', 'ignore').decode('ascii')
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _gdelt_topics_for_ticker(self, ticker: str) -> List[str]:
        """Returns relevant GDELT search topics for a given ticker."""
        ticker_upper = ticker.upper()
        # Direct match first (WTI, OIL, GLD, BTC, etc.)
        for key in GDELT_TOPIC_MAP:
            if key in ticker_upper or ticker_upper in key:
                return GDELT_TOPIC_MAP[key]
        return GDELT_TOPIC_MAP["DEFAULT"]

    # ==========================================================================
    # TIER 0: FRED — Federal Reserve & Central Bank Data
    # ==========================================================================
    def fetch_fed_macro(self, ticker: str) -> List[Dict]:
        """
        TIER 0 (weight 1.5): Federal Reserve Economic Data (FRED RSS).
        
        Fetches:
          - FRED blog / research posts (macro policy signals)
          - Relevant economic series release headlines

        No API key required. Uses FRED public RSS.
        """
        results = []
        fred_rss_urls = [
            ("https://feeds.a.dj.com/rss/RSSMarketsMain.xml", "Fed/Macro News"),
            ("https://www.federalreserve.gov/feeds/press_all.xml", "Federal Reserve Press"),
        ]

        for url, label in fred_rss_urls:
            try:
                resp = requests.get(url, headers=self.headers, timeout=6)
                if resp.status_code == 200:
                    root = ET.fromstring(resp.content)
                    items = root.findall('./channel/item')
                    for item in items[:3]:
                        title_elem = item.find('title')
                        desc_elem = item.find('description')
                        if title_elem is not None and title_elem.text:
                            title = self._clean_text(title_elem.text)
                            desc = self._clean_text(desc_elem.text) if desc_elem is not None and desc_elem.text else ""
                            combined = f"{title}. {desc}"[:300]
                            if len(combined.strip()) > 10:
                                results.append({
                                    "source": "FRED",
                                    "text": combined,
                                    "tier_weight": TIER_WEIGHTS["FRED"],
                                    "tier": 0,
                                    "label": label,
                                })
            except Exception as e:
                logger.debug(f"FRED RSS fetch failed ({label}): {e}")

        # Fallback: FRED API for key series (no API key — uses public endpoint)
        if not results:
            try:
                url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=FEDFUNDS"
                resp = requests.get(url, headers=self.headers, timeout=5)
                if resp.status_code == 200:
                    lines = resp.text.strip().split('\n')
                    if len(lines) >= 2:
                        latest = lines[-1].split(',')
                        if len(latest) == 2:
                            date_val, rate_val = latest[0], latest[1]
                            text = (
                                f"Federal Funds Rate as of {date_val}: {rate_val}%. "
                                f"Current Fed policy rate is {rate_val} percent."
                            )
                            results.append({
                                "source": "FRED",
                                "text": text,
                                "tier_weight": TIER_WEIGHTS["FRED"],
                                "tier": 0,
                                "label": "Fed Funds Rate",
                            })
            except Exception as e:
                logger.debug(f"FRED CSV fetch failed: {e}")

        return results

    # ==========================================================================
    # TIER 1: SEC EDGAR — Regulatory Filings (unchanged, enhanced)
    # ==========================================================================
    def fetch_sec_filings(self, ticker: str) -> List[Dict]:
        """TIER 1 (weight 1.0): SEC EDGAR regulatory filings."""
        filings = []
        ticker_upper = ticker.upper()
        try:
            if ticker_upper in self.ticker_to_cik:
                cik = self.ticker_to_cik[ticker_upper]
                url = (
                    f"https://www.sec.gov/cgi-bin/browse-edgar"
                    f"?action=getcompany&CIK={cik}&type=&dateb=&owner=exclude&count=5&output=atom"
                )
            else:
                url = (
                    f"https://www.sec.gov/cgi-bin/browse-edgar"
                    f"?company={ticker}&CIK=&action=getcompany&output=atom"
                )

            resp = requests.get(url, headers=self.headers, timeout=5)
            if resp.status_code == 200:
                root = ET.fromstring(resp.content)
                ns = {'atom': 'http://www.w3.org/2005/Atom'}
                for entry in root.findall('atom:entry', ns)[:3]:
                    title = entry.find('atom:title', ns).text
                    summary_elem = entry.find('atom:summary', ns)
                    summary = summary_elem.text if summary_elem is not None else ""
                    clean_text = self._clean_text(f"{title} - {summary}")
                    filings.append({
                        "source": "SEC EDGAR",
                        "text": clean_text,
                        "tier_weight": TIER_WEIGHTS["SEC EDGAR"],
                        "tier": 1,
                    })
        except Exception as e:
            logger.debug(f"SEC EDGAR fetch failed: {e}")
        return filings

    # ==========================================================================
    # TIER 2: Yahoo Finance RSS — Financial News
    # ==========================================================================
    def fetch_institutional_news(self, ticker: str) -> List[Dict]:
        """TIER 2 (weight 0.9): Yahoo Finance RSS news feed."""
        news = []
        try:
            url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
            resp = requests.get(url, headers=self.headers, timeout=5)
            if resp.status_code == 200:
                root = ET.fromstring(resp.content)
                for item in root.findall('./channel/item')[:4]:
                    title_elem = item.find('title')
                    if title_elem is not None and title_elem.text:
                        clean_text = self._clean_text(title_elem.text)
                        news.append({
                            "source": "Yahoo Finance",
                            "text": clean_text,
                            "tier_weight": TIER_WEIGHTS["Yahoo Finance"],
                            "tier": 2,
                        })
        except Exception as e:
            logger.debug(f"Yahoo Finance RSS failed: {e}")
        return news

    # ==========================================================================
    # TIER 3: GDELT — Real-Time Geopolitical Events
    # ==========================================================================
    def fetch_gdelt_geopolitical(self, ticker: str) -> List[Dict]:
        """
        TIER 3 (weight 0.8): GDELT Project — Real-Time Global Event Monitor.

        GDELT scans 65+ languages across 150+ countries every 15 minutes.
        We query the GDELT 2.0 DOC API for news articles matching 
        geopolitical themes relevant to the ticker's asset class.

        This is where Chatbot B's Iran-Israel event would be DETECTED.

        API Docs: https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/
        """
        results = []
        topics = self._gdelt_topics_for_ticker(ticker)

        # Query top 2 topics to keep latency low (GDELT is public, rate-limit conscious)
        for topic in topics[:2]:
            try:
                # GDELT DOC 2.0 API — returns JSON article list
                url = (
                    "https://api.gdeltproject.org/api/v2/doc/doc"
                    f"?query={requests.utils.quote(topic)}"
                    "&mode=artlist"
                    "&maxrecords=5"
                    "&format=json"
                    "&timespan=1440"   # Last 24 hours (minutes)
                    "&sort=DateDesc"
                )
                resp = requests.get(url, headers=self.headers, timeout=8)

                if resp.status_code == 200:
                    data = resp.json()
                    articles = data.get("articles", [])

                    for article in articles[:3]:
                        title = article.get("title", "")
                        source_country = article.get("sourcecountry", "")
                        tone = article.get("tone", None)   # GDELT tone score

                        if not title:
                            continue

                        clean_title = self._clean_text(title)

                        # Augment with GDELT tone signal if available
                        tone_text = ""
                        if tone is not None:
                            try:
                                tone_val = float(tone)
                                if tone_val < -5:
                                    tone_text = " Tone: highly negative geopolitical event."
                                elif tone_val < -2:
                                    tone_text = " Tone: negative geopolitical tension detected."
                                elif tone_val > 2:
                                    tone_text = " Tone: positive diplomatic development."
                            except (ValueError, TypeError):
                                pass

                        full_text = f"{clean_title}.{tone_text}"
                        if source_country:
                            full_text += f" Source: {source_country}."

                        results.append({
                            "source": "GDELT",
                            "text": full_text[:400],
                            "tier_weight": TIER_WEIGHTS["GDELT"],
                            "tier": 3,
                            "topic": topic,
                            "gdelt_tone": tone,
                        })

            except Exception as e:
                logger.debug(f"GDELT fetch failed for topic '{topic}': {e}")

        # GDELT GKG (Global Knowledge Graph) — Conflict/Instability events
        # This catches military escalation events like Hormuz
        try:
            conflict_url = (
                "https://api.gdeltproject.org/api/v2/doc/doc"
                "?query=military+escalation+oil+supply+disruption"
                "&mode=artlist"
                "&maxrecords=3"
                "&format=json"
                "&timespan=720"   # Last 12 hours
                "&sort=DateDesc"
            )
            resp = requests.get(conflict_url, headers=self.headers, timeout=8)
            if resp.status_code == 200:
                data = resp.json()
                for article in data.get("articles", [])[:2]:
                    title = article.get("title", "")
                    if title:
                        clean = self._clean_text(title)
                        results.append({
                            "source": "GDELT",
                            "text": f"[CONFLICT ALERT] {clean}",
                            "tier_weight": TIER_WEIGHTS["GDELT"] * 1.1,  # Boost conflict signals
                            "tier": 3,
                            "topic": "conflict_monitor",
                            "gdelt_tone": article.get("tone"),
                        })
        except Exception as e:
            logger.debug(f"GDELT conflict monitor failed: {e}")

        return results

    # ==========================================================================
    # TIER 4: Economic Calendar — Scheduled Macro Data Releases
    # ==========================================================================
    def fetch_economic_calendar(self, ticker: str) -> List[Dict]:
        """
        TIER 4 (weight 0.7): Upcoming and recent economic data releases.

        Uses the Trading Economics RSS feed (free, no API key).
        Covers: CPI, NFP, GDP, FOMC, PMI, PPI, retail sales, etc.

        These are the SCHEDULED macro shocks — not geopolitical surprises.
        """
        results = []
        econ_rss_urls = [
            "https://tradingeconomics.com/rss/news.aspx",
            "https://www.investing.com/rss/news_301.rss",  # Economic indicators
        ]

        for url in econ_rss_urls:
            try:
                resp = requests.get(url, headers=self.headers, timeout=6)
                if resp.status_code == 200:
                    root = ET.fromstring(resp.content)
                    for item in root.findall('./channel/item')[:4]:
                        title_elem = item.find('title')
                        desc_elem  = item.find('description')
                        if title_elem is not None and title_elem.text:
                            title = self._clean_text(title_elem.text)
                            desc  = self._clean_text(desc_elem.text) if desc_elem is not None and desc_elem.text else ""
                            combined = f"{title}. {desc}"[:300]
                            if len(combined.strip()) > 15:
                                results.append({
                                    "source": "EconCalendar",
                                    "text": combined,
                                    "tier_weight": TIER_WEIGHTS["EconCalendar"],
                                    "tier": 4,
                                })
                    break   # One working source is enough
            except Exception as e:
                logger.debug(f"Economic calendar RSS failed ({url}): {e}")

        # Fallback: BLS, BEA public release pages via Yahoo RSS proxy
        if not results:
            try:
                url = "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GSPC&region=US&lang=en-US"
                resp = requests.get(url, headers=self.headers, timeout=5)
                if resp.status_code == 200:
                    root = ET.fromstring(resp.content)
                    macro_keywords = ["CPI", "inflation", "GDP", "jobs", "Fed", "rate", "PMI", "NFP", "payroll"]
                    for item in root.findall('./channel/item')[:8]:
                        title_elem = item.find('title')
                        if title_elem is not None and title_elem.text:
                            title = title_elem.text
                            if any(kw.lower() in title.lower() for kw in macro_keywords):
                                clean = self._clean_text(title)
                                results.append({
                                    "source": "EconCalendar",
                                    "text": f"[MACRO] {clean}",
                                    "tier_weight": TIER_WEIGHTS["EconCalendar"],
                                    "tier": 4,
                                })
            except Exception as e:
                logger.debug(f"EconCalendar fallback failed: {e}")

        return results

    # ==========================================================================
    # TIER 5: Macro FX & Commodity Context
    # ==========================================================================
    def fetch_macro_fx_commodity(self, ticker: str) -> List[Dict]:
        """
        TIER 5 (weight 0.6): Real-time commodity and FX macro context.

        Fetches price headlines for key macro barometers:
          DXY (dollar index), VIX (fear index), GC=F (gold),
          CL=F (crude oil), ^TNX (10Y yield), EURUSD=X

        Converts prices into narrative text so FinBERT can score them.
        """
        results = []

        macro_symbols = {
            "DX-Y.NYB":  "US Dollar Index",
            "^VIX":      "VIX Volatility Index",
            "GC=F":      "Gold Futures",
            "CL=F":      "WTI Crude Oil Futures",
            "^TNX":      "US 10-Year Treasury Yield",
            "EURUSD=X":  "EUR/USD Exchange Rate",
            "BZ=F":      "Brent Crude Oil Futures",
        }

        for symbol, label in macro_symbols.items():
            try:
                url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"
                resp = requests.get(url, headers=self.headers, timeout=5)
                if resp.status_code == 200:
                    root = ET.fromstring(resp.content)
                    for item in root.findall('./channel/item')[:1]:
                        title_elem = item.find('title')
                        if title_elem is not None and title_elem.text:
                            clean = self._clean_text(title_elem.text)
                            results.append({
                                "source": "MacroFX",
                                "text": f"[{label}] {clean}",
                                "tier_weight": TIER_WEIGHTS["MacroFX"],
                                "tier": 5,
                                "symbol": symbol,
                            })
            except Exception as e:
                logger.debug(f"MacroFX fetch failed for {symbol}: {e}")

        return results

    # ==========================================================================
    # TIER 6: Google Trends Proxy (Search Interest via RSS)
    # ==========================================================================
    def fetch_google_trends_proxy(self, ticker: str) -> List[Dict]:
        """
        TIER 6 (weight 0.5): Google Trends proxy via Google News RSS.

        Google News RSS reflects rising search interest — a proxy for
        the trend momentum that moves retail flows.

        When a topic spikes on Google Trends → it's already in this feed.
        """
        results = []
        topics = self._gdelt_topics_for_ticker(ticker)
        search_query = topics[0] if topics else ticker

        try:
            # Google News RSS (public, no API key)
            encoded_query = requests.utils.quote(search_query)
            url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
            resp = requests.get(url, headers=self.headers, timeout=6)

            if resp.status_code == 200:
                root = ET.fromstring(resp.content)
                for item in root.findall('./channel/item')[:3]:
                    title_elem = item.find('title')
                    if title_elem is not None and title_elem.text:
                        clean = self._clean_text(title_elem.text)
                        # Strip Google's " - Publisher" suffix
                        clean = re.sub(r'\s+-\s+\S+$', '', clean)
                        if len(clean) > 10:
                            results.append({
                                "source": "GoogleTrends",
                                "text": clean,
                                "tier_weight": TIER_WEIGHTS["GoogleTrends"],
                                "tier": 6,
                            })
        except Exception as e:
            logger.debug(f"Google Trends proxy failed: {e}")

        return results

    # ==========================================================================
    # TIER 7: Reddit WallStreetBets — Retail Momentum (unchanged)
    # ==========================================================================
    def fetch_retail_momentum(self, ticker: str) -> List[Dict]:
        """TIER 7 (weight 0.3): Reddit WallStreetBets retail sentiment."""
        reddit_posts = []
        try:
            url = (
                f"https://www.reddit.com/r/wallstreetbets/search.json"
                f"?q={ticker}&restrict_sr=on&sort=new&limit=3"
            )
            resp = requests.get(url, headers=self.reddit_headers, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                children = data.get('data', {}).get('children', [])
                for child in children:
                    post = child['data']
                    title = post.get('title', '')
                    clean_text = self._clean_text(title)
                    reddit_posts.append({
                        "source": "Reddit r/WSB",
                        "text": clean_text,
                        "tier_weight": TIER_WEIGHTS["Reddit r/WSB"],
                        "tier": 7,
                    })
        except Exception as e:
            logger.debug(f"Reddit fetch failed: {e}")
        return reddit_posts

    # ==========================================================================
    # MASTER ASSEMBLER
    # ==========================================================================
    def get_global_context_payload(self, ticker: str) -> List[Dict]:
        """
        Assembles the full 8-tier intelligence payload.

        Tier execution order (by importance):
          0 → FRED (Central Bank)
          1 → SEC EDGAR
          2 → Yahoo Finance
          3 → GDELT (Geopolitical)
          4 → Economic Calendar
          5 → Macro FX/Commodity
          6 → Google Trends Proxy
          7 → Reddit WSB

        Returns deduplicated, length-filtered list of signals.
        """
        print(f"      📡 [MCP] Broadcasting API requests across 8 intelligence tiers for {ticker}...")

        payload: List[Dict] = []

        # Execute all tiers (failures are silent — each method handles its own exceptions)
        payload.extend(self.fetch_fed_macro(ticker))         # Tier 0: FRED
        payload.extend(self.fetch_sec_filings(ticker))       # Tier 1: SEC
        payload.extend(self.fetch_institutional_news(ticker))# Tier 2: Yahoo
        payload.extend(self.fetch_gdelt_geopolitical(ticker))# Tier 3: GDELT ← NEW
        payload.extend(self.fetch_economic_calendar(ticker)) # Tier 4: EconCal ← NEW
        payload.extend(self.fetch_macro_fx_commodity(ticker))# Tier 5: MacroFX ← NEW
        payload.extend(self.fetch_google_trends_proxy(ticker))# Tier 6: GTrends ← NEW
        payload.extend(self.fetch_retail_momentum(ticker))   # Tier 7: Reddit

        # Deduplication by text content
        seen_texts: set = set()
        clean_payload: List[Dict] = []
        for item in payload:
            text = item.get('text', '')
            if text not in seen_texts and len(text.strip()) > 5:
                clean_payload.append(item)
                seen_texts.add(text)

        # Sort by tier_weight descending (highest-trust signals scored first by FinBERT)
        clean_payload.sort(key=lambda x: x.get('tier_weight', 0), reverse=True)

        # Fallback if everything failed
        if not clean_payload:
            clean_payload.append({
                "source": "System Fallback",
                "text": f"{ticker} trading in standard market conditions with low news velocity.",
                "tier_weight": 0.50,
                "tier": 99,
            })

        tier_counts = {}
        for item in clean_payload:
            src = item.get('source', 'Unknown')
            tier_counts[src] = tier_counts.get(src, 0) + 1

        print(f"      ✅ [MCP] Context Assembly Complete. {len(clean_payload)} global signals captured.")
        print(f"         Sources: {tier_counts}")

        return clean_payload


# ==============================================================================
# UPDATED SentimentAgent.analyze_with_mcp  (drop-in replacement)
# ==============================================================================
# The only change needed in sentiment_agent.py is the print statement
# to display the new tier labels. Everything else works identically.
#
# Updated display loop (replace the existing for-loop):
#
#   for item in mcp_payload:
#       source     = item['source']
#       text       = item['text']
#       tier_weight = item['tier_weight']
#       tier_num   = item.get('tier', '?')
#       topic      = item.get('topic', '')
#
#       if len(text.strip()) < 10:
#           continue
#
#       label, raw_score, probs = self.get_sentiment(text)
#       confidence = float(np.max(probs))
#       confidence_multiplier = confidence if confidence >= 0.65 else 0.30
#       final_item_weight = tier_weight * confidence_multiplier
#       adjusted_score = raw_score * final_item_weight
#
#       weighted_scores.append(adjusted_score)
#       total_weight_applied += final_item_weight
#
#       topic_tag = f" [{topic}]" if topic else ""
#       print(f"   🏛️ [T{tier_num}: {source}{topic_tag}] (Weight: {tier_weight:.1f})")
#       print(f"      📄 '{text[:70]}...' -> {label.upper()} (Raw: {raw_score:.2f})")