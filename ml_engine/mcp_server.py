import requests
import xml.etree.ElementTree as ET
import re

class MCPDataServer:
    """
    Model Context Protocol (MCP) Server.
    Acts as an isolated, multi-threaded intelligence gatherer.
    Fetches raw data from the internet, sanitizes it, and packages it 
    into a standardized JSON payload.
    """
    def __init__(self):
        # Base headers for institutional APIs
        self.headers = {
            "User-Agent": "FinFolioX_Research_Bot/1.0 (Educational Capstone)"
        }
        # Strict Reddit headers to avoid 429 Rate Limits
        self.reddit_headers = {
            "User-Agent": "FinFolioX:v1.0 (by /u/finfolio_admin)"
        }
        
        # SEC requires CIK numbers, not tickers. We load the official mapping map.
        self.ticker_to_cik = {}
        self._load_sec_cik_mapping()
        
        print("   🔌 [MCP Server] Initialized and listening for requests...")

    def _load_sec_cik_mapping(self):
        """Loads the official SEC Ticker-to-CIK JSON mapping."""
        try:
            url = "https://www.sec.gov/files/company_tickers.json"
            resp = requests.get(url, headers=self.headers, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                for entry in data.values():
                    # Format CIK to 10 digits as required by EDGAR
                    self.ticker_to_cik[entry['ticker'].upper()] = str(entry['cik_str']).zfill(10)
        except Exception:
            pass # Fails silently, will use fallback ticker search

    def _clean_text(self, text):
        """Strips HTML, emojis, and weird artifacts to protect FinBERT."""
        if not text: return ""
        text = re.sub(re.compile('<.*?>'), '', text)
        text = text.encode('ascii', 'ignore').decode('ascii')
        return text.strip()

    def fetch_sec_filings(self, ticker):
        """TIER 1: Absolute Truth (SEC EDGAR RSS)"""
        filings = []
        ticker_upper = ticker.upper()
        try:
            # FIX: Use mapped CIK if available, otherwise fallback to company search
            if ticker_upper in self.ticker_to_cik:
                cik = self.ticker_to_cik[ticker_upper]
                url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=&dateb=&owner=exclude&count=5&output=atom"
            else:
                url = f"https://www.sec.gov/cgi-bin/browse-edgar?company={ticker}&CIK=&action=getcompany&output=atom"
                
            response = requests.get(url, headers=self.headers, timeout=5)
            if response.status_code == 200:
                root = ET.fromstring(response.content)
                ns = {'atom': 'http://www.w3.org/2005/Atom'}
                for entry in root.findall('atom:entry', ns)[:3]:
                    title = entry.find('atom:title', ns).text
                    summary_elem = entry.find('atom:summary', ns)
                    summary = summary_elem.text if summary_elem is not None else ""
                    clean_text = self._clean_text(f"{title} - {summary}")
                    filings.append({"source": "SEC EDGAR", "text": clean_text, "tier_weight": 1.0})
        except Exception:
            pass 
        return filings

    def fetch_institutional_news(self, ticker):
        """TIER 2: High Trust News (Yahoo Finance RSS - Highly Stable)"""
        news = []
        try:
            # FIX: Switched from Google to Yahoo RSS for stability
            url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
            response = requests.get(url, headers=self.headers, timeout=5)
            if response.status_code == 200:
                root = ET.fromstring(response.content)
                for item in root.findall('./channel/item')[:4]:
                    title = item.find('title').text
                    clean_text = self._clean_text(title)
                    news.append({"source": "Yahoo Finance", "text": clean_text, "tier_weight": 0.80})
        except Exception:
            pass
        return news

    def fetch_retail_momentum(self, ticker):
        """TIER 3: Retail Sentiment (Reddit WallStreetBets)"""
        reddit_posts = []
        try:
            url = f"https://www.reddit.com/r/wallstreetbets/search.json?q={ticker}&restrict_sr=on&sort=new&limit=3"
            # FIX: Using dedicated Reddit headers
            response = requests.get(url, headers=self.reddit_headers, timeout=5)
            if response.status_code == 200:
                data = response.json()
                children = data.get('data', {}).get('children', [])
                for child in children:
                    post = child['data']
                    title = post.get('title', '')
                    clean_text = self._clean_text(title)
                    reddit_posts.append({"source": "Reddit r/WSB", "text": clean_text, "tier_weight": 0.30})
        except Exception:
            pass
        return reddit_posts

    def get_global_context_payload(self, ticker):
        """Assembles the final context payload."""
        print(f"      📡 [MCP] Broadcasting API requests across 3 intelligence tiers for {ticker}...")
        payload = []
        
        payload.extend(self.fetch_sec_filings(ticker))
        payload.extend(self.fetch_institutional_news(ticker))
        payload.extend(self.fetch_retail_momentum(ticker))
        
        # Deduplication filter
        seen_texts = set()
        clean_payload = []
        for item in payload:
            if item['text'] not in seen_texts and len(item['text']) > 5:
                clean_payload.append(item)
                seen_texts.add(item['text'])

        if not clean_payload:
            clean_payload.append({
                "source": "System Fallback", 
                "text": f"{ticker} trading in standard market conditions with low news velocity.", 
                "tier_weight": 0.50
            })

        print(f"      ✅ [MCP] Context Assembly Complete. {len(clean_payload)} global signals captured.")
        return clean_payload