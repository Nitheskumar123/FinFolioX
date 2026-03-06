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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_engine.technical_agent import TechnicalAgent
from ml_engine.sentiment_agent import SentimentAgent
from ml_engine.fusion_agent import FusionAgent
from ml_engine.regime_agent import RegimeAgent
from ml_engine.risk_engine import RiskEngine
from ml_engine.correlation_agent import CorrelationDivergenceDetector
from ml_engine.uncertainty_agent import UncertaintyAgent
from ml_engine.explainability_agent import ExplainabilityAgent
from ml_engine.topology_agent import TopologyAgent

# ==============================================================================
# SYSTEM CONSTANTS
# ==============================================================================
SYSTEM_VERSION = "17.0 (MCP Embedded + Topology)"
DEFAULT_CAPITAL = 10_000.0
MAX_RISK_PER_TRADE = 0.20
NEWS_LOOKBACK_ITEMS = 5
UNCERTAINTY_THRESHOLD_HIGH = 0.15
UNCERTAINTY_THRESHOLD_MODERATE = 0.05
DIVERGENCE_THRESHOLD_CRITICAL = 0.70
DIVERGENCE_THRESHOLD_MINOR = 0.40

# FIX: Commodity ticker normalization map (placed at MODULE level, not inside class)
COMMODITY_MAP = {
    "GOLD": "GLD",
    "SILVER": "SLV",
    "OIL": "USO",
    "NATGAS": "UNG",
}

# ==============================================================================
# PHASE 11 IMPORT
# ==============================================================================
try:
    from adversarial_tester import AdversarialTester
    print("   ✅ Phase 11 (Red Team) Loaded via direct import.")
except ImportError:
    try:
        from ml_engine.adversarial_tester import AdversarialTester
        print("   ✅ Phase 11 (Red Team) Loaded via package import.")
    except ImportError:
        print("   ⚠️ Phase 11 Module missing. Red Team Disabled.")
        AdversarialTester = None

# ==============================================================================
# PHASE 13 IMPORT
# ==============================================================================
try:
    from conflict_resolver import ConflictResolver
    print("   ✅ Phase 13 (Conflict Arbitrator) Loaded via direct import.")
except ImportError:
    try:
        from ml_engine.conflict_resolver import ConflictResolver
        print("   ✅ Phase 13 (Conflict Arbitrator) Loaded via package import.")
    except ImportError:
        print("   ⚠️ Phase 13 Module missing. Conflict Resolution Disabled.")
        ConflictResolver = None

# ==============================================================================
# PHASE 14 IMPORT
# ==============================================================================
try:
    from meta_agent import MetaAgent
    print("   [+] Phase 14 (Meta-Agent) Loaded via direct import.")
except ImportError:
    try:
        from ml_engine.meta_agent import MetaAgent
        print("   [+] Phase 14 (Meta-Agent) Loaded via package import.")
    except ImportError:
        print("   [!] Phase 14 Module missing. Meta-Agent Disabled.")
        MetaAgent = None

# ==============================================================================
# PHASE 16 IMPORT
# ==============================================================================
try:
    from heatmap_agent import HeatmapAgent
    print("   [+] Phase 16 (Heatmap Agent) Loaded via direct import.")
except ImportError:
    try:
        from ml_engine.heatmap_agent import HeatmapAgent
        print("   [+] Phase 16 (Heatmap Agent) Loaded via package import.")
    except ImportError:
        print("   [!] Phase 16 Module missing. Heatmap Agent Disabled.")
        HeatmapAgent = None


# ==============================================================================
# FINFOLIO-X MASTER SYSTEM CLASS
# ==============================================================================
class FinFolioSystem:
    """
    The Master Orchestrator for FinFolio-X AI Trading System.

    Architecture:
    1. Technical Agent  (LSTM)       : Analyzes price trends and patterns.
    2. Sentiment Agent  (FinBERT)    : Analyzes global news and sentiment via MCP.
    3. Regime Agent     (HMM)        : Detects hidden market states.
    4. Correlation Agent(Graph)      : Detects systemic risk and anomalies.
    5. Uncertainty Agent(Bayesian)   : Quantifies model confidence.
    6. Explainability   (SHAP)       : Explains WHY the model decided.
    7. Topology Agent   (TDA)        : Phase 24 Geometric Market Shape.
    8. Fusion Agent     (Attention)  : Weighs all inputs to make a decision.
    9. Risk Engine      (Kelly)      : Calculates optimal position sizing.
    10. Conflict Resolver(Phase 13)  : Arbitrates agent disagreements.
    """

    def __init__(self):
        self._print_startup_banner()

        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        MODELS_DIR = os.path.join(BASE_DIR, "saved_models")

        # 1. Technical Agent
        print("\n   🔹 [1/9] Loading Technical Agent (LSTM Chart Reader)...")
        try:
            self.tech_agent = TechnicalAgent(
                model_path=os.path.join(MODELS_DIR, "lstm_technical.pth"),
                scaler_path=os.path.join(MODELS_DIR, "scaler.pkl"),
            )
            print("      ✅ LSTM Model Loaded Successfully.")
        except Exception as e:
            print(f"      ❌ Critical Error loading Technical Agent: {e}")
            sys.exit(1)

        # 2. Sentiment Agent
        print("   🔹 [2/9] Loading Sentiment Agent (FinBERT Language Model)...")
        try:
            self.sent_agent = SentimentAgent()
            print("      ✅ FinBERT Model Loaded Successfully.")
        except Exception as e:
            print(f"      Warning: Sentiment Agent failed ({e}). Using fallback.")
            self.sent_agent = None

        # 3. Regime Agent
        print("   🔹 [3/9] Loading Regime Agent (HMM Market Detector)...")
        try:
            self.regime_agent = RegimeAgent(
                model_path=os.path.join(MODELS_DIR, "hmm_regime.pkl")
            )
            print("      ✅ Hidden Markov Model Loaded Successfully.")
        except Exception as e:
            print(f"      ⚠️ Warning: Regime Agent failed ({e}).")
            self.regime_agent = None

        # 4. Correlation Agent
        print("   🔹 [4/9] Loading Correlation Agent (Statistical Graph)...")
        try:
            self.corr_agent = CorrelationDivergenceDetector()
            print("      ✅ Market Graph Engine Initialized.")
        except Exception as e:
            print(f"      ⚠️ Warning: Correlation Agent failed ({e}).")
            self.corr_agent = None

        # 5. Uncertainty Agent
        print("   🔹 [5/9] Loading Uncertainty Agent (Bayesian Wrapper)...")
        try:
            self.uncertainty_agent = UncertaintyAgent(self.tech_agent)
            print("      ✅ Monte Carlo Dropout Engine Initialized.")
        except Exception as e:
            print(f"      ⚠️ Warning: Uncertainty Agent failed ({e}).")
            self.uncertainty_agent = None

        # 6. Fusion Agent
        print("   🔹 [6/9] Loading Fusion Agent (Multi-Head Attention)...")
        try:
            self.fusion_agent = FusionAgent(
                model_path=os.path.join(MODELS_DIR, "attention_fusion.pth")
            )
            print("      ✅ Attention Mechanism Loaded Successfully.")
        except Exception as e:
            print(f"      ❌ Critical Error loading Fusion Agent: {e}")
            sys.exit(1)

        # 7. Risk Engine
        print("   🔹 [7/9] Loading Risk Engine (Kelly Criterion)...")
        self.risk_engine = RiskEngine(default_account_size=DEFAULT_CAPITAL)
        print(f"      ✅ Risk Manager Online (Account: ${DEFAULT_CAPITAL:,.2f}).")

        # 8. Explainability Agent (lazy init)
        print("   🔹 [8/9] Preparing Explainability Agent (SHAP)...")
        self.explainability_agent = None

        # 9. Topological Shape Agent (Phase 24)
        print("   🔹 [9/9] Loading Topological Shape Agent (Ripser)...")
        try:
            self.topology_agent = TopologyAgent(time_delay=5, dimension=3, lookback=60)
        except Exception:
            self.topology_agent = None

        # Regime Scaler
        self.regime_scaler_path = os.path.join(MODELS_DIR, "regime_scaler.pkl")
        if os.path.exists(self.regime_scaler_path):
            self.regime_scaler = joblib.load(self.regime_scaler_path)
        else:
            self.regime_scaler = None
            print("      ⚠️ Warning: Regime Scaler not found. HMM accuracy may be reduced.")

        print("\n✅ SYSTEM INITIALIZATION COMPLETE. ALL ENGINES ONLINE.\n")

        # Phase hooks
        self.red_team = AdversarialTester(self) if AdversarialTester else None
        self.conflict_resolver = ConflictResolver() if ConflictResolver else None
        self.meta_agent = MetaAgent() if MetaAgent else None
        self.heatmap_agent = HeatmapAgent() if HeatmapAgent else None

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
    # HELPER: TECHNICAL INDICATORS
    # ==========================================================================
    def _calculate_rsi(self, prices, window=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, prices):
        ema_12 = prices.ewm(span=12, adjust=False).mean()
        ema_26 = prices.ewm(span=26, adjust=False).mean()
        return ema_12 - ema_26


    # ==========================================================================
    # MODULAR ANALYSIS METHODS
    # ==========================================================================
    def _fetch_stock_data(self, ticker):
        """Retrieves and processes historical stock data."""
        ticker = COMMODITY_MAP.get(ticker.upper(), ticker)

        try:
            print("   ⏳ Fetching historical data from Yahoo Finance...")
            stock = yf.Ticker(ticker)
            hist = stock.history(period="2y")
            if len(hist) < 200:
                return None, "❌ Not enough historical data (Need > 200 days)."

            hist["SMA_50"] = hist["Close"].rolling(window=50).mean()
            hist["SMA_200"] = hist["Close"].rolling(window=200).mean()
            hist["RSI"] = self._calculate_rsi(hist["Close"])
            hist["MACD"] = self._calculate_macd(hist["Close"])
            hist.dropna(inplace=True)

            if len(hist) < 60:
                return None, "❌ Not enough data after processing indicators."

            return stock, hist
        except Exception as e:
            return None, f"❌ Data Connection Error: {e}"

    def _analyze_technicals_and_uncertainty(self, hist):
        """Runs LSTM, Bayesian Uncertainty, and SHAP Explainability."""
        last_60_days = hist[["Close", "Volume", "SMA_50", "SMA_200", "RSI", "MACD"]].tail(60)

        print("\n   📈 [Technical Analysis] Reading Charts (LSTM v2)...")
        lstm_signal = self.tech_agent.predict(last_60_days)
        print(f"      - Standard LSTM Signal: {lstm_signal:.4f}")

        if self.explainability_agent is None:
            self.explainability_agent = ExplainabilityAgent(self.tech_agent, hist)

        print("   🔍 [Explainability] Running SHAP Analysis...")
        shap_scores, top_driver = self.explainability_agent.explain_prediction(last_60_days)
        if shap_scores:
            print(f"      - Top Driver: {top_driver} (Impact: {shap_scores[top_driver]:.4f})")
            sorted_feats = sorted(shap_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
            print(f"      - Key Factors: {', '.join([f'{k}={v:.3f}' for k, v in sorted_feats])}")

        print("   🎲 [Uncertainty Agent] Running Monte Carlo Simulation (50 runs)...")
        mc_mean, mc_std = self.uncertainty_agent.predict_with_uncertainty(last_60_days)

        uncertainty_status = "✅ High Certainty"
        if mc_std > UNCERTAINTY_THRESHOLD_MODERATE:
            uncertainty_status = "⚠️ Moderate Uncertainty"
        if mc_std > UNCERTAINTY_THRESHOLD_HIGH:
            uncertainty_status = "🚨 HIGH UNCERTAINTY (Guessing)"

        print(f"      - Bayesian Mean: {mc_mean:.4f}")
        print(f"      - Uncertainty (StdDev): {mc_std:.4f} ({uncertainty_status})")

        return lstm_signal, mc_mean, mc_std, uncertainty_status, top_driver

    def _analyze_sentiment_module(self, ticker, stock_obj, lstm_signal):
        """
        Phase 22: Live News Ingestion via MCP.
        Delegates completely to the Sentiment Agent and MCP Server.
        """
        print("\n   📰 [Sentiment Analysis] Initiating MCP Protocol...")
        
        if self.sent_agent is None:
            print("      [!] Sentiment Agent unavailable. Using neutral score.")
            return 0.0

        try:
            # Ask FinBERT to process the MCP Payload
            result = self.sent_agent.analyze_with_mcp(ticker)
            
            # Guard against unexpected MCP/FinBERT None returns
            if not result:
                print("      ⚠️ MCP failed to return valid data. Defaulting to neutral.")
                return 0.0
                
            sent_label, sent_score = result
            print(f"      - Final Corroborated Sentiment Score: {sent_score:.4f} ({sent_label})")
            return sent_score
            
        except Exception as e:
            print(f"      ⚠️ MCP/FinBERT Pipeline Error: {e}. Defaulting to neutral.")
            return 0.0

    def _analyze_regime_module(self, hist):
        """
        Universal Regime Detection (Heuristic).
        """
        print("\n   ⛈️  [Regime Detection] Detecting Market State (Heuristic)...")

        current_vol = hist["Close"].pct_change().rolling(10).std().iloc[-1]
        if pd.isna(current_vol):
            current_vol = 0.015

        # --- Universal Mathematical Heuristic ---
        sma_50 = float(hist["SMA_50"].iloc[-1])
        sma_200 = float(hist["SMA_200"].iloc[-1])

        if sma_50 > sma_200 and current_vol < 0.025:
            regime_label = "Bull"
        elif sma_50 < sma_200 and current_vol > 0.015:
            regime_label = "Bear"
        else:
            regime_label = "Sideways"

        print(f"      - Current Volatility: {current_vol:.4f}")
        print(f"      - Detected State: {regime_label} (via Universal Heuristic)")

        return regime_label, current_vol

    def _analyze_correlation_module(self, ticker):
        """Runs Graph-Based Systemic Risk Check."""
        print("\n   🕸️  [Systemic Risk] Analyzing Cross-Asset Divergence (GNN/Graph)...")
        risk_score, _ = self.corr_agent.get_market_context(ticker)

        div_status = "✅ Synced"
        if risk_score > DIVERGENCE_THRESHOLD_MINOR:
            div_status = "⚠️ Minor Divergence"
        if risk_score > DIVERGENCE_THRESHOLD_CRITICAL:
            div_status = "🚨 CRITICAL DIVERGENCE (Anomaly)"

        print(f"      - Divergence Score: {risk_score:.4f}")
        print(f"      - Systemic Status: {div_status}")

        return risk_score, div_status

    # ==========================================================================
    # MAIN ANALYZER ORCHESTRATOR
    # ==========================================================================
    def analyze_stock(self, ticker="AAPL"):
        """Main entry point for analysis."""
        print(f"📊 STARTING DEEP DIVE ANALYSIS FOR: {ticker}")

        stock_obj, hist = self._fetch_stock_data(ticker)
        if stock_obj is None:
            return hist

        last_price = hist["Close"].iloc[-1]

        # Phase 14: Load trust scores 
        trust_scores = None
        if self.meta_agent:
            trust_scores = self.meta_agent.get_trust_scores(ticker=ticker)
            self.meta_agent.print_trust_report(trust_scores)

        lstm_signal, mc_mean, mc_std, uncertainty_status, top_driver = (
            self._analyze_technicals_and_uncertainty(hist)
        )
        sent_score = self._analyze_sentiment_module(ticker, stock_obj, lstm_signal)
        regime_label, current_vol = self._analyze_regime_module(hist)
        risk_score, div_status = self._analyze_correlation_module(ticker)

        # ── Phase 24: Topological Analysis ───────────────────────────────
        topo_modifier = 1.0
        topo_signal = "UNKNOWN"
        if hasattr(self, "topology_agent") and self.topology_agent:
            print("\n   🌀 [Phase 24] Computing Persistent Homology (Vietoris-Rips)...")
            topology_result = self.topology_agent.analyze(hist)
            topo_modifier = topology_result.get("topology_modifier", 1.0)
            topo_signal = topology_result.get("market_shape_signal", "UNKNOWN")

        # Phase 11: Red Team live check
        robustness_penalty = 0.0
        if self.red_team:
            print("\n   🛡️  [Red Team] Running Live Robustness Check...")
            try:
                crashed_df = self.red_team.generate_flash_crash(hist, drop_pct=0.20)
                input_crashed = self.red_team._prepare_data_for_ai(crashed_df)
                crashed_score = (
                    self.tech_agent.predict_signal(input_crashed)
                    if hasattr(self.tech_agent, "predict_signal")
                    else self.tech_agent.predict(input_crashed)
                )
                robustness_delta = lstm_signal - crashed_score
                if robustness_delta < 0.02:
                    print(f"      ❌ WARNING: Model is stubborn! (Delta: {robustness_delta:.4f})")
                    robustness_penalty = 0.2
                else:
                    print(f"      ✅ PASS: Model detected the crash. (Delta: {robustness_delta:.4f})")
            except Exception as e:
                print(f"      ⚠️ Red Team check failed: {e}")

        # Fusion - Modulated by Phase 24 Topology
        print("\n   🧠 [Fusion Engine] Synthesizing Intelligence Layers...")
        vol_input = 0.9 if regime_label == "Bear" else 0.2 if regime_label == "Bull" else 0.5
        
        # Apply Topology geometric modifier to Fusion Inputs
        final_conf, weights = self.fusion_agent.predict(
            lstm_p=mc_mean * topo_modifier,
            sent_s=sent_score * topo_modifier,
            vol_v=vol_input * topo_modifier,
            trust_scores=trust_scores,
        )
        print(f"      - Raw Fusion Confidence: {final_conf:.4f} (Topology Modified: {topo_modifier:.2f}x)")

        # Phase 13: Conflict Resolution
        if self.conflict_resolver:
            arbitration_result = self.conflict_resolver.arbitrate(
                tech_score=lstm_signal,
                sent_score=sent_score,
                mc_std=mc_std,
                regime_label=regime_label,
                risk_score=risk_score,
                fusion_confidence=final_conf,
                trust_scores=trust_scores,
            )
            final_conf = arbitration_result["adjusted_confidence"]
            self.conflict_resolver.print_report(arbitration_result)
        else:
            if risk_score > DIVERGENCE_THRESHOLD_CRITICAL:
                final_conf *= 0.5
            if mc_std > 0.10:
                final_conf *= 0.8

        # Phase 16: Disagreement Heatmap
        gdi_penalty = 1.0
        gdi_value = 0.0
        if self.heatmap_agent:
            heatmap_result = self.heatmap_agent.analyze(
                lstm_score=lstm_signal,
                sent_score=sent_score,
                regime_label=regime_label,
                regime_vol=current_vol,
            )
            self.heatmap_agent.print_heatmap(heatmap_result)
            gdi_penalty = heatmap_result["penalty"]
            gdi_value = heatmap_result["gdi"] * 100

        # Risk Management
        print("\n   [Risk Engine] Calculating Position Sizing (Kelly)...")
        alloc_pct, kelly_debug = self.risk_engine.calculate_position_size(
            final_conf, current_vol,
            disagreement_penalty=gdi_penalty,
            regime=regime_label,
        )
        num_shares, cash_value = self.risk_engine.get_shares_amount(last_price, alloc_pct)

        # Final Report
        print("\n" + "█" * 72)
        print(f"🏆 FINFOLIO-X INTELLIGENCE REPORT: {ticker}")
        print("█" * 72)
        print(f"   📊 AI Confidence Score : {final_conf:.4f} (Scale: 0.0 - 1.0)")
        print(f"   🎲 Model Uncertainty   : {mc_std:.4f} ({uncertainty_status})")
        print(f"   ⛈️  Market Regime       : {regime_label} (Vol: {current_vol:.4f})")
        print(f"   🕸️  Systemic Risk       : {risk_score:.4f} ({div_status})")
        print(f"   🌀 Topological Shape   : {topo_signal} (Mod: {topo_modifier:.2f}x)")
        print(f"   🔍 Primary SHAP Driver : {top_driver}")
        print("-" * 72)

        # ------------------------------------------------------------------
        # FINAL DECISION LOGIC (Relaxed for Prototype LSTM)
        # ------------------------------------------------------------------
        BUY_THRESHOLD = 0.50  
        BUY_GDI_MAX = 55.0  

        decision = "HOLD"
        
        # The ultimate entry gate
        if alloc_pct > 0.0 and final_conf >= BUY_THRESHOLD and regime_label != "Bear" and gdi_value < BUY_GDI_MAX:
            decision = "BUY 🟢"
        elif final_conf < 0.40:
            decision = "SELL 🔴"

        print(f"   🚀 STRATEGY SIGNAL     : {decision}")

        if decision == "BUY 🟢":
            print(f"   💰 RECOMMENDED SIZE    : ${cash_value:.2f}")
            print(f"   📉 PORTFOLIO WEIGHT    : {alloc_pct * 100:.1f}%")
            print(f"   📦 ORDER QUANTITY      : {num_shares} Shares (@ ${last_price:.2f})")
            print(f"   🧮 KELLY EDGE          : {kelly_debug:.4f}")
        else:
            print("   ⛔ RISK ADVICE         : Stay Cash / Do Not Enter Trade.")

        w_lstm = weights.get("LSTM_Focus", 0)
        w_sent = weights.get("Sentiment_Focus", 0)
        w_vol = weights.get("Volatility_Focus", 0)
        print("-" * 72)
        print("   🔍 AI REASONING (ATTENTION WEIGHTS):")
        print(f"      • Technicals (Chart) : {w_lstm:.2f}")
        print(f"      • Sentiment (News)   : {w_sent:.2f}")
        print(f"      • Risk (Volatility)  : {w_vol:.2f}")
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
        print("   (c) FinFolio-X Team 2026")

        # Phase 14: Log decision
        if self.meta_agent:
            try:
                self.meta_agent.log_decision(
                    ticker=ticker,
                    lstm_score=lstm_signal,
                    sent_score=sent_score,
                    regime_label=regime_label,
                    risk_score=risk_score,
                    fusion_confidence=final_conf,
                    final_decision=decision,
                    price_at_decision=last_price,
                )
            except Exception as e:
                print(f"   [!] Meta-Agent logging failed: {e}")

    def run_stress_test(self, ticker="AAPL"):
        """Manually triggers the Phase 11 stress test."""
        if self.red_team:
            self.red_team.run_robustness_test(ticker)
        else:
            print("❌ Cannot run stress test: Phase 11 module not loaded.")