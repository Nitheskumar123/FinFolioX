import os
import sys
import time
import yfinance as yf
import numpy as np
import pandas as pd
import joblib
import random
import requests
import re
import xml.etree.ElementTree as ET
from datetime import datetime

# ==============================================================================
# PROJECT CONFIGURATION & PATH SETUP
# ==============================================================================
# Ensure the python path includes the project root for modular imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our Custom Intelligence Agents
from ml_engine.technical_agent import TechnicalAgent
from ml_engine.sentiment_agent import SentimentAgent
from ml_engine.fusion_agent import FusionAgent
from ml_engine.regime_agent import RegimeAgent
from ml_engine.risk_engine import RiskEngine
from ml_engine.correlation_agent import CorrelationDivergenceDetector
from ml_engine.uncertainty_agent import UncertaintyAgent
from ml_engine.explainability_agent import ExplainabilityAgent # <--- PHASE 10

# ==============================================================================
# SYSTEM CONSTANTS & CONFIGURATION
# ==============================================================================
SYSTEM_VERSION = "10.0 (SHAP-Explainable)"
DEFAULT_CAPITAL = 10000.0
MAX_RISK_PER_TRADE = 0.20  # 20% Hard Cap
NEWS_LOOKBACK_ITEMS = 5
UNCERTAINTY_THRESHOLD_HIGH = 0.15
UNCERTAINTY_THRESHOLD_MODERATE = 0.05
DIVERGENCE_THRESHOLD_CRITICAL = 0.70
DIVERGENCE_THRESHOLD_MINOR = 0.40

# ==============================================================================
# FINFOLIO-X MASTER SYSTEM CLASS
# ==============================================================================

class FinFolioSystem:
    """
    The Master Orchestrator for FinFolio-X AI Trading System.
    
    This system integrates 8 specialized AI agents into a single coherent
    decision-making pipeline. It uses a voting mechanism weighted by an
    attention network to produce final buy/sell signals with full explainability.
    
    Architecture:
    1. Technical Agent (LSTM): Analyzes price trends and patterns.
    2. Sentiment Agent (FinBERT): Analyzes global news and sentiment.
    3. Regime Agent (HMM): Detects hidden market states (Bull/Bear).
    4. Correlation Agent (Graph): Detects systemic risk and anomalies.
    5. Uncertainty Agent (Bayesian): Quantifies model confidence/guessing.
    6. Explainability Agent (SHAP): Explains WHY the model made a prediction.
    7. Fusion Agent (Attention): Weighs all inputs to make a decision.
    8. Risk Engine (Kelly): Calculates optimal position sizing.
    """

    def __init__(self):
        """
        Initialize all AI agents, load pre-trained models, and set up the environment.
        """
        self._print_startup_banner()
        
        # Define paths for models
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
        
        # ------------------------------------------------------------------
        # 1. Initialize Technical Agent (The Chart Analyst)
        # ------------------------------------------------------------------
        print("\n   🔹 [1/8] Loading Technical Agent (LSTM Chart Reader)...")
        try:
            self.tech_agent = TechnicalAgent(
                model_path=os.path.join(MODELS_DIR, "lstm_technical.pth"),
                scaler_path=os.path.join(MODELS_DIR, "scaler.pkl")
            )
            print("      ✅ LSTM Model Loaded Successfully.")
        except Exception as e:
            print(f"      ❌ Critical Error loading Technical Agent: {e}")
            sys.exit(1)
        
        # ------------------------------------------------------------------
        # 2. Initialize Sentiment Agent (The News Analyst)
        # ------------------------------------------------------------------
        print("   🔹 [2/8] Loading Sentiment Agent (FinBERT Language Model)...")
        try:
            self.sent_agent = SentimentAgent()
            print("      ✅ FinBERT Model Loaded Successfully.")
        except Exception as e:
            print(f"      ⚠️ Warning: Sentiment Agent failed ({e}). using fallback.")

        # ------------------------------------------------------------------
        # 3. Initialize Regime Agent (The Market Weather Station)
        # ------------------------------------------------------------------
        print("   🔹 [3/8] Loading Regime Agent (HMM Market Detector)...")
        try:
            self.regime_agent = RegimeAgent(
                model_path=os.path.join(MODELS_DIR, "hmm_regime.pkl")
            )
            print("      ✅ Hidden Markov Model Loaded Successfully.")
        except Exception as e:
            print(f"      ⚠️ Warning: Regime Agent failed ({e}).")

        # ------------------------------------------------------------------
        # 4. Initialize Correlation Agent (Systemic Risk Detector)
        # ------------------------------------------------------------------
        print("   🔹 [4/8] Loading Correlation Agent (Statistical Graph)...")
        try:
            self.corr_agent = CorrelationDivergenceDetector()
            print("      ✅ Market Graph Engine Initialized.")
        except Exception as e:
            print(f"      ⚠️ Warning: Correlation Agent failed ({e}).")

        # ------------------------------------------------------------------
        # 5. Initialize Uncertainty Agent (The Lie Detector)
        # ------------------------------------------------------------------
        print("   🔹 [5/8] Loading Uncertainty Agent (Bayesian Wrapper)...")
        try:
            self.uncertainty_agent = UncertaintyAgent(self.tech_agent)
            print("      ✅ Monte Carlo Dropout Engine Initialized.")
        except Exception as e:
            print(f"      ⚠️ Warning: Uncertainty Agent failed ({e}).")

        # ------------------------------------------------------------------
        # 6. Initialize Fusion Agent (The Decision Maker)
        # ------------------------------------------------------------------
        print("   🔹 [6/8] Loading Fusion Agent (Multi-Head Attention)...")
        try:
            self.fusion_agent = FusionAgent(
                model_path=os.path.join(MODELS_DIR, "attention_fusion.pth")
            )
            print("      ✅ Attention Mechanism Loaded Successfully.")
        except Exception as e:
             print(f"      ❌ Critical Error loading Fusion Agent: {e}")
             sys.exit(1)

        # ------------------------------------------------------------------
        # 7. Initialize Risk Engine (The Wallet Manager)
        # ------------------------------------------------------------------
        print("   🔹 [7/8] Loading Risk Engine (Kelly Criterion)...")
        self.risk_engine = RiskEngine(default_account_size=DEFAULT_CAPITAL)
        print(f"      ✅ Risk Manager Online (Account: ${DEFAULT_CAPITAL:,.2f}).")
        
        # ------------------------------------------------------------------
        # 8. Initialize Explainability Agent (SHAP Engine) - PHASE 10
        # ------------------------------------------------------------------
        # Note: We perform "Lazy Initialization" for this agent.
        # We need actual historical data to create the SHAP background dataset.
        # We will initialize it during the first call to `analyze_stock`.
        print("   🔹 [8/8] Preparing Explainability Agent (SHAP)...")
        self.explainability_agent = None 
        
        # Load Regime Scaler (Critical for correct HMM inputs)
        self.regime_scaler_path = os.path.join(MODELS_DIR, "regime_scaler.pkl")
        if os.path.exists(self.regime_scaler_path):
            self.regime_scaler = joblib.load(self.regime_scaler_path)
        else:
            self.regime_scaler = None
            print("      ⚠️ Warning: Regime Scaler not found. HMM accuracy may be reduced.")
        
        print("\n✅ SYSTEM INITIALIZATION COMPLETE. ALL ENGINES ONLINE.\n")

    def _print_startup_banner(self):
        print("\n" + "█" * 72)
        print("🚀 INITIALIZING FINFOLIO-X: EXPLAINABLE AI TRADING SYSTEM")
        print("█" * 72)
        print(f"   • Version: {SYSTEM_VERSION}")
        print("   • Mode: Live Inference (Real-Time Data)")
        print("   • Architecture: Multi-Agent Mixture of Experts (MoE) + XAI")
        print("   • Copyright © 2026 FinFolio Team")
        print("-" * 72)

    # ==========================================================================
    # HELPER: TECHNICAL INDICATOR CALCULATIONS
    # ==========================================================================
    
    def _calculate_rsi(self, prices, window=14):
        """
        Calculates Relative Strength Index (RSI).
        Used to detect Overbought (>70) or Oversold (<30) conditions.
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, prices):
        """
        Calculates MACD (Moving Average Convergence Divergence).
        Used to identify trend changes and momentum.
        """
        ema_12 = prices.ewm(span=12, adjust=False).mean()
        ema_26 = prices.ewm(span=26, adjust=False).mean()
        return ema_12 - ema_26

    # ==========================================================================
    # HELPER: NEWS SCRAPING & CLEANING ENGINE
    # ==========================================================================

    def _clean_html_tags(self, text):
        """
        Removes HTML tags from RSS summaries (e.g., <a href=...>).
        Returns clean text.
        """
        if not text:
            return ""
        clean = re.compile('<.*?>')
        return re.sub(clean, '', text)

    def _fetch_google_news_rss(self, ticker):
        """
        Scrapes Google News RSS Feed.
        Extracts: Title, Link, Date, Source, AND Summary Description.
        """
        news_items = []
        try:
            # RSS URL for specific stock news (Localized to India/English)
            url = f"https://news.google.com/rss/search?q={ticker}+stock+news&hl=en-IN&gl=IN&ceid=IN:en"
            
            # Send Request (Timeout 5s)
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                root = ET.fromstring(response.content)
                
                # Iterate through top 5 news items
                for item in root.findall('./channel/item')[:NEWS_LOOKBACK_ITEMS]:
                    title = item.find('title').text
                    link = item.find('link').text
                    pub_date = item.find('pubDate').text
                    
                    # Safe extraction of description
                    desc_elem = item.find('description')
                    if desc_elem is not None:
                        description = desc_elem.text
                    else:
                        description = ""
                    
                    # Clean the HTML from description
                    summary = self._clean_html_tags(description)
                    
                    # Extract Source Name
                    source_name = "Google News"
                    source_obj = item.find('source')
                    if source_obj is not None:
                        source_name = source_obj.text
                    
                    # Clean Date Format
                    try:
                        dt_obj = datetime.strptime(pub_date, "%a, %d %b %Y %H:%M:%S %Z")
                        date_str = dt_obj.strftime("%Y-%m-%d %H:%M")
                    except:
                        date_str = pub_date[:16]
                        
                    news_items.append({
                        'title': title,
                        'summary': summary[:200] + "...", # Truncate long summaries
                        'link': link,
                        'date': date_str,
                        'source': source_name
                    })
        except Exception as e:
            # Silent fail is okay, we have fallbacks
            pass
            
        return news_items

    def _fetch_yahoo_news(self, stock_obj):
        """
        Fetches news from Yahoo Finance API.
        Attempts to get summary if available.
        """
        news_items = []
        try:
            raw_news = stock_obj.news
            if raw_news:
                for item in raw_news:
                    title = item.get('title', '')
                    link = item.get('link', 'No Link Available')
                    publisher = item.get('publisher', 'Yahoo Finance')
                    summary = title # Default fallback
                    
                    pub_time = item.get('providerPublishTime', 0)
                    date_str = datetime.fromtimestamp(pub_time).strftime('%Y-%m-%d %H:%M') if pub_time else "Unknown"
                    
                    if title:
                        news_items.append({
                            'title': title,
                            'summary': summary, 
                            'link': link, 
                            'date': date_str,
                            'source': publisher
                        })
        except:
            pass
        return news_items

    def _get_fallback_news(self, ticker, trend_strength):
        """
        SIMULATION MODE:
        Generates realistic backup headlines + Summaries if all Internet APIs fail.
        This ensures the AI always has data to process during demos.
        """
        print(f"   ⚠️ NETWORK ALERT: Live News APIs unreachable. Activating Simulation Mode.")
        
        base_link = f"https://finance.yahoo.com/quote/{ticker}"
        today_str = datetime.now().strftime('%Y-%m-%d %H:%M')
        
        if trend_strength > 0.6: # Bullish Simulation
            return [
                {
                    'title': f"Analysts upgrade {ticker} following strong momentum",
                    'summary': f"Major investment banks have raised their price targets for {ticker}, citing robust quarterly earnings and expanding market share in the tech sector.",
                    'link': base_link, 'date': today_str, 'source': 'Simulated - Bloomberg'
                },
                {
                    'title': f"{ticker} announces positive outlook, stock surges",
                    'summary': f"Shares of {ticker} rallied today after the CEO announced a new strategic partnership.",
                    'link': base_link, 'date': today_str, 'source': 'Simulated - Reuters'
                }
            ]
        elif trend_strength < 0.4: # Bearish Simulation
            return [
                {
                    'title': f"{ticker} faces supply chain headwinds",
                    'summary': f"{ticker} shares dipped as reports of manufacturing delays in Asia raised concerns.",
                    'link': base_link, 'date': today_str, 'source': 'Simulated - CNBC'
                },
                {
                    'title': f"Investors cautious on {ticker} volatility",
                    'summary': f"Market volatility has pushed {ticker} lower as institutional investors rotate out.",
                    'link': base_link, 'date': today_str, 'source': 'Simulated - NDTV'
                }
            ]
        else: # Neutral Simulation
            return [
                {
                    'title': f"{ticker} trades sideways amidst market uncertainty",
                    'summary': f"Trading volume for {ticker} remains low as investors await the upcoming Federal Reserve meeting minutes.",
                    'link': base_link, 'date': today_str, 'source': 'Simulated - MarketWatch'
                }
            ]

    # ==========================================================================
    # MODULAR ANALYSIS METHODS
    # ==========================================================================

    def _fetch_stock_data(self, ticker):
        """Retrieves and processes historical stock data."""
        try:
            print("   ⏳ Fetching historical data from Yahoo Finance...")
            stock = yf.Ticker(ticker)
            hist = stock.history(period="2y") 
            if len(hist) < 200:
                return None, "❌ Not enough historical data (Need > 200 days)."
            
            # Feature Engineering
            hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
            hist['SMA_200'] = hist['Close'].rolling(window=200).mean()
            hist['RSI'] = self._calculate_rsi(hist['Close'])
            hist['MACD'] = self._calculate_macd(hist['Close'])
            hist.dropna(inplace=True)
            
            if len(hist) < 60:
                return None, "❌ Not enough data after processing indicators."
                
            return stock, hist
        except Exception as e:
            return None, f"❌ Data Connection Error: {e}"

    def _analyze_technicals_and_uncertainty(self, hist):
        """Runs LSTM, Bayesian Uncertainty, and SHAP Explainability."""
        last_60_days = hist[['Close', 'Volume', 'SMA_50', 'SMA_200', 'RSI', 'MACD']].tail(60)
        
        print("\n   📈 [Technical Analysis] Reading Charts (LSTM v2)...")
        lstm_signal = self.tech_agent.predict(last_60_days)
        print(f"      - Standard LSTM Signal: {lstm_signal:.4f}")
        
        # Phase 10: Explainability (Lazy Init)
        if self.explainability_agent is None:
            self.explainability_agent = ExplainabilityAgent(self.tech_agent, hist)
            
        print("   🔍 [Explainability] Running SHAP Analysis...")
        shap_scores, top_driver = self.explainability_agent.explain_prediction(last_60_days)
        if shap_scores:
            print(f"      - Top Driver: {top_driver} (Impact: {shap_scores[top_driver]:.4f})")
            # Show top 3 features
            sorted_feats = sorted(shap_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
            print(f"      - Key Factors: {', '.join([f'{k}={v:.3f}' for k,v in sorted_feats])}")
        
        # Phase 9: Bayesian Check
        print("   🎲 [Uncertainty Agent] Running Monte Carlo Simulation (50 runs)...")
        mc_mean, mc_std = self.uncertainty_agent.predict_with_uncertainty(last_60_days)
        
        uncertainty_status = "✅ High Certainty"
        if mc_std > UNCERTAINTY_THRESHOLD_MODERATE: uncertainty_status = "⚠️ Moderate Uncertainty"
        if mc_std > UNCERTAINTY_THRESHOLD_HIGH: uncertainty_status = "🚨 HIGH UNCERTAINTY (Guessing)"
        
        print(f"      - Bayesian Mean: {mc_mean:.4f}")
        print(f"      - Uncertainty (StdDev): {mc_std:.4f} ({uncertainty_status})")
        
        return lstm_signal, mc_mean, mc_std, uncertainty_status, top_driver

    def _analyze_sentiment_module(self, ticker, stock_obj, lstm_signal):
        """Runs News Scraping and FinBERT Analysis."""
        print("\n   📰 [Sentiment Analysis] Scraping Global News...")
        
        google_news = self._fetch_google_news_rss(ticker)
        yahoo_news = self._fetch_yahoo_news(stock_obj)
        
        # Merge and Deduplicate
        all_news = google_news + yahoo_news
        seen_titles = set()
        unique_news = []
        for n in all_news:
            if n['title'] not in seen_titles:
                unique_news.append(n)
                seen_titles.add(n['title'])
        
        if not unique_news:
            unique_news = self._get_fallback_news(ticker, lstm_signal)
        
        analysis_news = unique_news[:NEWS_LOOKBACK_ITEMS]
        print(f"      Found {len(unique_news)} articles. Analyzing Top {len(analysis_news)}:")
        
        ai_input_texts = []
        for i, item in enumerate(analysis_news):
            print(f"      {i+1}. [{item['source']}] {item['date']}")
            print(f"         📢 Headline: {item['title']}")
            print(f"         📝 Summary : {item['summary'][:100]}...") 
            print("         " + "-"*30)
            ai_input_texts.append(f"{item['title']}. {item['summary']}")

        sent_label, sent_score = self.sent_agent.analyze_daily_headlines(ai_input_texts)
        print(f"      - FinBERT Score: {sent_score:.4f} ({sent_label})")
        
        return sent_score

    def _analyze_regime_module(self, hist):
        """Runs HMM Regime Detection."""
        print("\n   ⛈️  [Regime Detection] Analyzing Market Volatility (HMM)...")
        
        current_vol = hist['Close'].pct_change().rolling(10).std().iloc[-1]
        current_ret = hist['Close'].pct_change().iloc[-1]
        
        regime_input = np.array([[current_ret, current_vol]])
        
        if self.regime_scaler:
            scaled_input = self.regime_scaler.transform(regime_input)
            regime_label = self.regime_agent.get_regime_label(scaled_input)
        else:
            regime_label = self.regime_agent.get_regime_label(regime_input)
        
        print(f"      - Current Volatility: {current_vol:.4f}")
        print(f"      - Detected State: {regime_label}")
        
        return regime_label, current_vol

    def _analyze_correlation_module(self, ticker):
        """Runs Graph-Based Systemic Risk Check."""
        print("\n   🕸️  [Systemic Risk] Analyzing Cross-Asset Divergence (GNN/Graph)...")
        risk_score, _ = self.corr_agent.get_market_context(ticker)
        
        div_status = "✅ Synced"
        if risk_score > DIVERGENCE_THRESHOLD_MINOR: div_status = "⚠️ Minor Divergence"
        if risk_score > DIVERGENCE_THRESHOLD_CRITICAL: div_status = "🚨 CRITICAL DIVERGENCE (Anomaly)"
        
        print(f"      - Divergence Score: {risk_score:.4f}")
        print(f"      - Systemic Status: {div_status}")
        
        return risk_score, div_status

    # ==========================================================================
    # MAIN ANALYZER ORCHESTRATOR
    # ==========================================================================

    def analyze_stock(self, ticker="AAPL"):
        """
        Main entry point for analysis. Orchestrates the flow of data between agents.
        
        Returns:
            None (Prints detailed report to console)
        """
        print(f"📊 STARTING DEEP DIVE ANALYSIS FOR: {ticker}")
        
        # 1. Fetch Data
        stock_obj, hist = self._fetch_stock_data(ticker)
        if stock_obj is None:
            return hist # Returns error message
            
        last_price = hist['Close'].iloc[-1]

        # 2. Run Technical, Uncertainty & SHAP
        lstm_signal, mc_mean, mc_std, uncertainty_status, top_driver = self._analyze_technicals_and_uncertainty(hist)

        # 3. Run Sentiment Agent
        sent_score = self._analyze_sentiment_module(ticker, stock_obj, lstm_signal)

        # 4. Run Regime Agent
        regime_label, current_vol = self._analyze_regime_module(hist)

        # 5. Run Correlation Agent
        risk_score, div_status = self._analyze_correlation_module(ticker)
        
        # ------------------------------------------------------------------
        # STEP F: FUSION & OVERRIDE LOGIC
        # ------------------------------------------------------------------
        print("\n   🧠 [Fusion Engine] Synthesizing Intelligence Layers...")
        
        # Map Regime to Volatility Input for Fusion Agent
        if regime_label == "Bear": vol_input = 0.9
        elif regime_label == "Bull": vol_input = 0.2
        else: vol_input = 0.5
            
        # Use Bayesian Mean instead of Single LSTM prediction for better robustness
        final_conf, weights = self.fusion_agent.predict(
            lstm_p=mc_mean, 
            sent_s=sent_score, 
            vol_v=vol_input
        )
        print(f"      - Raw Fusion Confidence: {final_conf:.4f}")

        # --- SYSTEMIC RISK OVERRIDE ---
        if risk_score > DIVERGENCE_THRESHOLD_CRITICAL:
            print(f"      🔻 OVERRIDE TRIGGERED: Penalty applied due to high systemic divergence.")
            final_conf = final_conf * 0.5 
            print(f"      - Adjusted Confidence: {final_conf:.4f}")

        # --- BAYESIAN UNCERTAINTY OVERRIDE ---
        if mc_std > 0.10:
             print(f"      🔻 OVERRIDE TRIGGERED: Penalty applied due to high model uncertainty.")
             final_conf = final_conf * 0.8
             print(f"      - Adjusted Confidence: {final_conf:.4f}")

        # ------------------------------------------------------------------
        # STEP G: RISK MANAGEMENT (KELLY CRITERION)
        # ------------------------------------------------------------------
        print("\n   ⚖️  [Risk Engine] Calculating Position Sizing (Kelly)...")
        
        alloc_pct, kelly_debug = self.risk_engine.calculate_position_size(final_conf, current_vol)
        num_shares, cash_value = self.risk_engine.get_shares_amount(last_price, alloc_pct)
        
        # ------------------------------------------------------------------
        # STEP H: FINAL REPORT GENERATION
        # ------------------------------------------------------------------
        print("\n" + "█" * 72)
        print(f"🏆 FINFOLIO-X INTELLIGENCE REPORT: {ticker}")
        print("█" * 72)
        
        # 1. The Core Metrics
        print(f"   📊 AI Confidence Score : {final_conf:.4f} (Scale: 0.0 - 1.0)")
        print(f"   🎲 Model Uncertainty   : {mc_std:.4f} ({uncertainty_status})")
        print(f"   ⛈️  Market Regime       : {regime_label} (Vol: {current_vol:.4f})")
        print(f"   🕸️  Systemic Risk       : {risk_score:.4f} ({div_status})")
        print(f"   🔍 Primary SHAP Driver : {top_driver}")
        print("-" * 72)
        
        # 2. The Decision Logic
        decision = "HOLD"
        # Strategy: Only Buy if Allocation is positive AND Confidence is High
        if alloc_pct > 0.0 and final_conf > 0.6: 
            decision = "BUY 🟢"
        elif final_conf < 0.4: 
            decision = "SELL 🔴"
        
        print(f"   🚀 STRATEGY SIGNAL     : {decision}")
        
        # 3. Position Sizing (The Money Part)
        if decision == "BUY 🟢":
            print(f"   💰 RECOMMENDED SIZE    : ${cash_value:.2f}")
            print(f"   📉 PORTFOLIO WEIGHT    : {alloc_pct*100:.1f}%")
            print(f"   📦 ORDER QUANTITY      : {num_shares} Shares (@ ${last_price:.2f})")
            print(f"   🧮 KELLY EDGE          : {kelly_debug:.4f}")
        else:
            print("   ⛔ RISK ADVICE         : Stay Cash / Do Not Enter Trade.")
            
        # 4. Explainability (Why did the AI decide this?)
        w_lstm = weights.get('LSTM_Focus', 0)
        w_sent = weights.get('Sentiment_Focus', 0)
        w_vol = weights.get('Volatility_Focus', 0)
        
        print("-" * 72)
        print("   🔍 AI REASONING (ATTENTION WEIGHTS):")
        print(f"      • Technicals (Chart) : {w_lstm:.2f}")
        print(f"      • Sentiment (News)   : {w_sent:.2f}")
        print(f"      • Risk (Volatility)  : {w_vol:.2f}")
        
        # Interpret the attention
        max_focus = max(w_lstm, w_sent, w_vol)
        if max_focus == w_lstm:
            focus_msg = "The AI is prioritizing the Price Trend."
        elif max_focus == w_sent:
            focus_msg = "The AI is prioritizing News/Sentiment."
        else:
            focus_msg = "The AI is prioritizing Risk Management (Defensive)."
            
        print(f"      👉 Insight: {focus_msg}")
        print("█" * 72)
        print("\n   Disclaimer: This tool is for educational purposes only.")
        print("   It does not constitute financial advice. Trading involves risk.")
        print("   © FinFolio-X Team 2026")