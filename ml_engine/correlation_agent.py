import yfinance as yf
import pandas as pd
import numpy as np
import torch
from collections import deque
import logging

# Configure local logger for this agent
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CorrelationAgent")

class CorrelationDivergenceDetector:
    """
    detects Systemic Risk by analyzing the 'Graph' of market assets.
    
    The core hypothesis is that assets (like AAPL) generally move in sync with 
    their underlying market factors (SPY, QQQ, Rates, Volatility). 
    
    When an asset breaks this correlation significantly, it signals an 
    idiosyncratic anomaly or a potential trend reversal (Systemic Divergence).
    
    Logic:
    1. Nodes: Target Stock (AAPL) + Context (SPY, QQQ, TLT, VIXY).
    2. Edges: Dynamic Correlation (Rolling 30-day window).
    3. Anomaly: When Target moves significantly differently from the 
       correlation-weighted market consensus.
    """
    
    def __init__(self, lookback_window=60):
        """
        Initializes the Correlation Agent.
        
        Args:
            lookback_window (int): Size of the history buffer for Z-Score calculation.
                                   Higher values make the detector less sensitive to noise.
        """
        # We use VIXY (VIX ETF) instead of ^VIX because ^VIX often fails in API downloads
        # SPY: S&P 500 (Broad Market)
        # QQQ: Nasdaq 100 (Tech Sector)
        # TLT: 20+ Year Treasury Bond (Interest Rate/Safe Haven Proxy)
        # VIXY: Volatility Index Futures (Fear Gauge)
        self.assets = ["SPY", "QQQ", "TLT", "VIXY"] 
        self.lookback_window = lookback_window
        
        # History buffer to calculate Z-Scores (Standard Deviations)
        # This makes the model "learn" what normal divergence looks like over time.
        self.divergence_history = deque(maxlen=lookback_window)
        
        print("   ✅ Correlation Graph Engine Initialized.")

    def __repr__(self):
        return f"<CorrelationDivergenceDetector assets={self.assets} history={len(self.divergence_history)}>"
        
    def get_market_context(self, target_ticker="AAPL"):
        """
        Calculates the Systemic Risk Score (0.0 to 1.0) based on graph divergence.
        
        Steps:
        1. Fetch 6 months of daily OHLC data.
        2. Build a correlation matrix (The Graph Edges).
        3. Calculate 'Expected Move' based on neighbors (Graph Convolution).
        4. Compare 'Expected' vs 'Actual' to find Divergence.
        5. Normalize Divergence using rolling Z-Score.
        
        Returns:
            risk_score (float): 0.0 (Synced) -> 1.0 (Critical Divergence)
            corr_matrix (DataFrame): The adjacency matrix of the graph
        """
        # 1. Define the Graph Nodes
        tickers = [target_ticker] + self.assets
        print(f"   🕸️  [Correlation Agent] Building Market Graph: {target_ticker} vs {self.assets}...")
        
        try:
            # 2. Fetch Data (6 Months)
            # We need enough data to build valid rolling correlations (at least 30 days)
            data = yf.download(tickers, period="6mo", progress=False)['Close']
            
            # --- CRITICAL FIX: Column Normalization ---
            # yfinance sometimes returns MultiIndex columns or leaves tickers with '^'
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            # Clean column names (remove ^ if present and uppercase) to match input list
            data.columns = [col.replace('^', '').upper() for col in data.columns]
            clean_target = target_ticker.replace('^', '').upper()
            
            # Check for missing nodes (Data integrity check)
            missing_nodes = set([t.replace('^', '').upper() for t in tickers]) - set(data.columns)
            if missing_nodes:
                print(f"      ⚠️ Missing Graph Nodes: {missing_nodes}. Using partial graph.")
                # Verify Target is present
                if clean_target not in data.columns:
                    print(f"      ❌ Target {clean_target} data missing from graph. Aborting.")
                    return 0.5, None
            
            # 3. Calculate Daily Returns (Node Features)
            returns = data.pct_change().dropna()
            
            # Check data sufficiency
            if len(returns) < 30:
                print("      ⚠️ Insufficient data for graph analysis (Need > 30 days).")
                return 0.5, None

            # 4. Build Adjacency Matrix (Edges) using last 30 days
            # This represents the strength of connection between assets right now.
            recent_returns = returns.tail(30)
            corr_matrix = recent_returns.corr()
            
            # 5. Calculate "Market Consensus" Move
            # We want to know: "Based on SPY/QQQ, what SHOULD AAPL do today?"
            target_corr_vector = corr_matrix[clean_target].drop(clean_target)
            
            # Get today's moves of Others
            latest_moves = returns.iloc[-1]
            market_moves = latest_moves.drop(clean_target)
            
            # DEBUG: Print Key Correlations for Transparency
            if 'SPY' in target_corr_vector.index:
                print(f"      - Correlation with SPY: {target_corr_vector['SPY']:.3f}")
            if 'TLT' in target_corr_vector.index:
                print(f"      - Correlation with TLT: {target_corr_vector['TLT']:.3f}")
            
            # Weighted Sum: (Correlation * Move) / Sum of Correlations
            # This is mathematically equivalent to a Graph Convolution operation
            weights = target_corr_vector.abs()
            weight_sum = weights.sum()
            
            # --- CRITICAL FIX: Division by Zero Safety ---
            if weight_sum < 1e-6:
                print("      ⚠️ Weak correlations detected. Defaulting to market mean.")
                expected_move = market_moves.mean()
            else:
                expected_move = (target_corr_vector * market_moves).sum() / weight_sum
            
            actual_move = latest_moves[clean_target]
            
            # 6. Calculate Divergence (The Anomaly)
            raw_divergence = abs(actual_move - expected_move)
            
            # 7. Normalize using Z-Score (Statistical Robustness)
            self.divergence_history.append(raw_divergence)
            
            # --- CRITICAL FIX: Warm-Up Logic ---
            # We need at least 10 samples to calculate a meaningful Standard Deviation
            if len(self.divergence_history) >= 10:
                mean_div = np.mean(self.divergence_history)
                std_div = np.std(self.divergence_history)
                
                if std_div > 1e-6: # Avoid division by zero
                    z_score = (raw_divergence - mean_div) / std_div
                    # Sigmoid Function: Squash Z-score (-3 to +3) into Probability (0.0 to 1.0)
                    # A high Z-score means the current divergence is weirdly high compared to history.
                    risk_score = 1 / (1 + np.exp(-z_score))
                else:
                    risk_score = 0.5
            else:
                # Honest Warm-up Message
                print(f"      ℹ️ Warming up Divergence Model ({len(self.divergence_history)}/10 samples)...")
                risk_score = 0.5 # Neutral
            
            # Clamp result to ensure it stays between 0 and 1
            risk_score = max(0.0, min(1.0, risk_score))
            
            return risk_score, corr_matrix

        except Exception as e:
            print(f"      ⚠️ Graph Calculation Error: {e}")
            import traceback
            traceback.print_exc() # Print full error for debugging
            return 0.5, None