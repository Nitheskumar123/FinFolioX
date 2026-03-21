import os
import sys
import time
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import timedelta

# Ensure Python can find the ml_engine module
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from ml_engine.master_system import FinFolioSystem
from ml_engine.langgraph_orchestrator import FinFolioGraphOrchestrator
import ml_engine.sentiment_agent as _sa  # <-- Import Sentiment Agent

# Store original yfinance history function to prevent price look-ahead bias
_original_history = yf.Ticker.history

def run_accuracy_test(test_date, outcome_date, csv_filename):
    tickers = [
        "AAPL", "MSFT", "NVDA", "TSLA", "META", "AMZN", "GOOGL", "AMD", "INTC", "NFLX",
        "JPM", "V", "WMT", "JNJ", "XOM", "CAT", "DIS", "BA", "MCD", "KO",
        "SPY", "QQQ", "TLT", "GLD", "SLV", "USO", "UNG", "DIA", "IWM", "EEM"
    ]

    print(f"\n🚀 STARTING COMPLETE AGENT EVALUATION")
    print(f"   Backtest Date: {test_date} -> Outcome Date: {outcome_date}")
    print(f"   Output File: {csv_filename}")

    # --- THE SMART PATCH ---
    # This securely overrides dates without causing keyword argument conflicts
    def _patched_history(self, **kwargs):
        kwargs.pop('period', None)
        kwargs.pop('start', None)  # Fixes the TypeError!
        kwargs.pop('end', None)    # Fixes the TypeError!
        
        start_val = (pd.to_datetime(test_date) - pd.Timedelta(days=800)).strftime('%Y-%m-%d')
        return _original_history(self, start=start_val, end=test_date, **kwargs)

    master_system = FinFolioSystem()
    orchestrator = FinFolioGraphOrchestrator(master_system)

    results = []
    
    # Buffer the end date slightly just for downloading the outcome evaluation data
    yf_end_date = (pd.to_datetime(outcome_date) + pd.Timedelta(days=2)).strftime('%Y-%m-%d')

    for i, ticker in enumerate(tickers):
        print(f"\n🧪 {i+1}/{len(tickers)} : {ticker}")
        
        try:
            # ==========================================
            # 1. RUN AI ANALYSIS (SANDBOXED)
            # ==========================================
            yf.Ticker.history = _patched_history # Turn patch ON
            try:
                state = orchestrator.run_analysis(ticker)
            finally:
                yf.Ticker.history = _original_history # Turn patch OFF immediately!

            if state.get("error"): 
                print(f"      ⚠️ Skipping {ticker}: {state['error']}")
                continue

            # ==========================================
            # 2. FETCH REAL-WORLD DATA (UNSANDBOXED)
            # ==========================================
            # Since the patch is off, this can safely look into the future for the outcome
            actual_data = yf.download(ticker, start=test_date, end=yf_end_date, progress=False)
            if actual_data.empty or len(actual_data) < 2:
                print(f"      ⚠️ No market data found for {ticker}")
                continue
            
            close_series = actual_data['Close'].squeeze()
            p_start = float(close_series.iloc[0]) # Price on test_date
            
            try:
                p_end = float(close_series.asof(outcome_date))
            except:
                p_end = float(close_series.iloc[-1]) # Fallback
                
            real_return = ((p_end - p_start) / p_start) * 100

            # 3. EXTRACT CONFIDENCE & REVERSE MATH
            adj_conf = state.get("fusion_confidence", 0.5)
            penalty = state.get("asc_penalty_multiplier", 1.0)
            
            # Use pre_asc_confidence directly from state (added in LangGraph orchestrator update)
            raw_conf = state.get("pre_asc_confidence", adj_conf)
            
            ai_decision = state.get("final_decision", "HOLD")

            # 4. CALCULATE ACCURACY STATUS
            status = "❌ Wrong"
            if "BUY" in ai_decision and real_return > 0.5: status = "✅ WIN (Up)"
            elif "SELL" in ai_decision and real_return < -0.5: status = "✅ WIN (Down)"
            elif "HOLD" in ai_decision and abs(real_return) <= 0.5: status = "✅ WIN (Flat)"
            elif "HOLD" in ai_decision and abs(real_return) > 0.5: status = "🟡 EDGE (Preserved Capital)"

            # 5. BUILD THE DATA ROW
            row = {
                "Ticker": ticker,
                "Accuracy_Status": status,
                "Real_Return": f"{real_return:+.2f}%",
                "Price_Start": round(p_start, 2),
                "Price_End": round(p_end, 2),
                "AI_Decision": ai_decision,
                "Decision_Flipped": state.get("decision_flipped", False),
                "Regime_Contradiction": state.get("regime_contradiction", False),
                "Market_Regime": state.get("regime_label"),
                "Volatility": round(state.get("current_vol", 0), 4),
                "Risk_Score": round(state.get("risk_score", 0), 4),
                "LSTM_Signal": round(state.get("lstm_signal", 0), 4),
                "FinBERT_Score": round(state.get("sent_score", 0), 4),
                "SHAP_Top_Driver": state.get("top_driver", "N/A"),
                "GDI_Boardroom_Tension": f"{round(state.get('gdi', 0)*100, 1)}%",
                "Topology_Mod": round(state.get("topology_modifier", 1.0), 4),
                "Causal_Mod": round(state.get("causal_modifier", 1.0), 4),
                "ASC_Sycophancy_Score": round(state.get("asc_score", 0), 4),
                "ASC_Penalty_Multiplier": round(penalty, 4),
                "RAW_CONFIDENCE": round(raw_conf, 4),
                "FINAL_CONFIDENCE": round(adj_conf, 4),
                "Alloc_Pct": f"{round(state.get('alloc_pct', 0)*100, 2)}%"
            }
            results.append(row)
            print(f"      Result: {status} | Return: {real_return:+.2f}%")
            
            time.sleep(1) # Rate limit protection

        except Exception as e:
            print(f"❌ Critical Error on {ticker}: {e}")

    # 6. EXPORT AND SUMMARY
    if results:
        df = pd.DataFrame(results)
        os.makedirs("data/meta", exist_ok=True)
        report_path = os.path.join(BASE_DIR, "data", "meta", csv_filename)
        df.to_csv(report_path, index=False)
        
        wins = len(df[df['Accuracy_Status'].str.contains("WIN")])
        edges = len(df[df['Accuracy_Status'].str.contains("EDGE")])
        losses = len(df[df['Accuracy_Status'].str.contains("Wrong")])
        
        accuracy = (wins / len(df)) * 100
        edge_rate = (edges / len(df)) * 100
        loss_rate = (losses / len(df)) * 100
        
        print("\n" + "█"*72)
        print(f"✅ EVALUATION COMPLETE FOR {test_date}!")
        print(f"✅ WINS: {wins} ({accuracy:.1f}%)")
        print(f"🟡 EDGES: {edges} ({edge_rate:.1f}%)")  
        print(f"❌ LOSSES: {losses} ({loss_rate:.1f}%)")
        print(f"📁 Full details saved to: {report_path}")
        print("█"*72)
    else:
        print(f"\n⚠️ No results generated for {test_date}.")


if __name__ == "__main__":
    
    # --- BLOCK SENTIMENT MCP TO PREVENT FUTURE NEWS BIAS ---
    _original_mcp = _sa.SentimentAgent.analyze_with_mcp
    
    try:
        # L4 FIX: Enable sentiment contribution validation using a synthetic momentum proxy
        # instead of a flat 0.0 which was completely nullifying sentiment impact!
        def mock_sentiment_proxy(self, ticker):
            # Because yf.Ticker.history is organically patched to halt at test_date above,
            # this avoids future news bias while providing valid variance for the Fusion Agent.
            hist = yf.Ticker(ticker).history(period="10d")
            if len(hist) >= 5:
                ret = float((hist['Close'].iloc[-1] / hist['Close'].iloc[-5]) - 1.0)
                score = min(max(ret * 15.0, -0.99), 0.99)
                label = "bullish" if score > 0.1 else "bearish" if score < -0.1 else "neutral"
                return (label, round(score, 4))
            return ("neutral", 0.0)
            
        _sa.SentimentAgent.analyze_with_mcp = mock_sentiment_proxy
        
        print("\n" + "*"*72)
        print("💡 SENTIMENT VALIDATION ON: Using dynamic momentum proxy to prevent future leak.")
        print("*"*72)
        
        # ==========================================
        # RUN 1: March 3 Analysis -> March 8 Outcome
        # ==========================================
        print("\n" + "="*72)
        print("▶️ INITIATING RUN 1: MAR 3 -> MAR 8")
        print("="*72)
        run_accuracy_test(
            test_date="2026-03-03", 
            outcome_date="2026-03-08", 
            csv_filename="accuracy_report_mar3_mar8.csv"
        )
        
        # ==========================================
        # 20-SECOND PAUSE BETWEEN RUNS
        # ==========================================
        print("\n" + "⏳"*36)
        print("Run 1 complete. Pausing for 20 seconds to cool down APIs...")
        print("⏳"*36)
        time.sleep(20)

        # ==========================================
        # RUN 2: March 9 Analysis -> March 16 Outcome
        # ==========================================
        print("\n" + "="*72)
        print("▶️ INITIATING RUN 2: MAR 9 -> MAR 16")
        print("="*72)
        run_accuracy_test(
            test_date="2026-03-09", 
            outcome_date="2026-03-16", 
            csv_filename="accuracy_report_mar9_mar16.csv"
        )
        
    finally:
        # Restore normal live MCP behavior
        _sa.SentimentAgent.analyze_with_mcp = _original_mcp
        
        print("\n" + "*"*72)
        print("🔥 SENTIMENT MCP RESTORED: Live headlines re-enabled.")
        print("*"*72)