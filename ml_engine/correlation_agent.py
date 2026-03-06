import pickle
import os
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import logging
from collections import deque

logger = logging.getLogger("CorrelationAgent")


class CorrelationDivergenceDetector:
    """
    Detects Systemic Risk by analyzing the 'Graph' of market assets.

    Core hypothesis: assets generally move in sync with their underlying
    market factors (SPY, QQQ, Rates, Volatility). When an asset breaks
    this correlation significantly, it signals an idiosyncratic anomaly
    or a potential trend reversal (Systemic Divergence).

    FIX v2: Divergence history is now persisted to disk so the warm-up
    period survives Django restarts. The model is ready after the first
    10 unique analysis calls — not reset every time the server reloads.
    """

    def __init__(
        self,
        lookback_window=60,
        cache_path=None,
    ):
        # Use VIXY (VIX ETF) instead of ^VIX because ^VIX often fails in downloads
        self.assets = ["SPY", "QQQ", "TLT", "VIXY"]
        self.lookback_window = lookback_window

        # Resolve cache path relative to project root
        if cache_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            cache_path = os.path.join(base_dir, "data", "meta", "divergence_cache.pkl")
        self.cache_path = cache_path

        # Load history from disk (survives server restarts)
        self.divergence_history = self._load_history()

        print("   ✅ Correlation Graph Engine Initialized.")
        if len(self.divergence_history) > 0:
            print(
                f"      ✅ Divergence history restored "
                f"({len(self.divergence_history)}/{lookback_window} samples)"
            )

    # ------------------------------------------------------------------
    # PERSISTENCE HELPERS
    # ------------------------------------------------------------------
    def _load_history(self):
        """Loads divergence history deque from disk if it exists."""
        try:
            if os.path.exists(self.cache_path):
                with open(self.cache_path, "rb") as f:
                    loaded = pickle.load(f)
                # Ensure maxlen matches current config
                history = deque(loaded, maxlen=self.lookback_window)
                return history
        except Exception as e:
            logger.warning(f"Could not load divergence cache: {e}")
        return deque(maxlen=self.lookback_window)

    def _save_history(self):
        """Saves divergence history deque to disk."""
        try:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, "wb") as f:
                pickle.dump(list(self.divergence_history), f)
        except Exception as e:
            logger.warning(f"Could not save divergence cache: {e}")

    def __repr__(self):
        return (
            f"<CorrelationDivergenceDetector "
            f"assets={self.assets} "
            f"history={len(self.divergence_history)}/{self.lookback_window}>"
        )

    # ------------------------------------------------------------------
    # MAIN ENTRY POINT
    # ------------------------------------------------------------------
    def get_market_context(self, target_ticker="AAPL"):
        """
        Calculates the Systemic Risk Score (0.0 → 1.0) based on graph divergence.

        Steps:
        1. Fetch 6 months of daily OHLC data for target + context assets.
        2. Build a rolling 30-day correlation matrix (The Graph Edges).
        3. Calculate 'Expected Move' via graph convolution (weighted neighbor moves).
        4. Compare 'Expected' vs 'Actual' to find Divergence.
        5. Normalize Divergence using rolling Z-Score (persisted across restarts).

        Returns:
            risk_score  (float)      : 0.0 (Synced) → 1.0 (Critical Divergence)
            corr_matrix (DataFrame)  : Adjacency matrix of the graph
        """
        tickers = [target_ticker] + self.assets
        print(
            f"   🕸️  [Correlation Agent] Building Market Graph: "
            f"{target_ticker} vs {self.assets}..."
        )

        try:
            # 1. Fetch Data (6 months)
            data = yf.download(tickers, period="6mo", progress=False)["Close"]

            # Normalize column names
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            data.columns = [col.replace("^", "").upper() for col in data.columns]
            clean_target = target_ticker.replace("^", "").upper()

            # Check for missing nodes
            missing_nodes = (
                set([t.replace("^", "").upper() for t in tickers]) - set(data.columns)
            )
            if missing_nodes:
                print(f"      ⚠️ Missing Graph Nodes: {missing_nodes}. Using partial graph.")
                if clean_target not in data.columns:
                    print(f"      ❌ Target {clean_target} data missing. Aborting.")
                    return 0.5, None

            # 2. Calculate Daily Returns
            returns = data.pct_change().dropna()

            if len(returns) < 30:
                print("      ⚠️ Insufficient data for graph analysis (Need > 30 days).")
                return 0.5, None

            # 3. Build Adjacency Matrix (last 30 days)
            recent_returns = returns.tail(30)
            corr_matrix = recent_returns.corr()

            # 4. Calculate "Market Consensus" Move via Graph Convolution
            target_corr_vector = corr_matrix[clean_target].drop(clean_target)
            latest_moves = returns.iloc[-1]
            market_moves = latest_moves.drop(clean_target)

            # Print key correlations for transparency
            if "SPY" in target_corr_vector.index:
                print(f"      - Correlation with SPY: {target_corr_vector['SPY']:.3f}")
            if "TLT" in target_corr_vector.index:
                print(f"      - Correlation with TLT: {target_corr_vector['TLT']:.3f}")

            weights = target_corr_vector.abs()
            weight_sum = weights.sum()

            if weight_sum < 1e-6:
                print("      ⚠️ Weak correlations detected. Defaulting to market mean.")
                expected_move = market_moves.mean()
            else:
                expected_move = (target_corr_vector * market_moves).sum() / weight_sum

            actual_move = latest_moves[clean_target]
            raw_divergence = abs(actual_move - expected_move)

            # 5. Normalize via Z-Score (using persisted history)
            self.divergence_history.append(raw_divergence)
            self._save_history()  # ← persist to disk after every sample

            if len(self.divergence_history) >= 10:
                mean_div = np.mean(self.divergence_history)
                std_div = np.std(self.divergence_history)

                if std_div > 1e-6:
                    z_score = (raw_divergence - mean_div) / std_div
                    risk_score = 1.0 / (1.0 + np.exp(-z_score))
                else:
                    risk_score = 0.5
            else:
                warm_up = len(self.divergence_history)
                print(
                    f"      ℹ️ Warming up Divergence Model "
                    f"({warm_up}/10 samples)..."
                )
                risk_score = 0.5

            risk_score = float(max(0.0, min(1.0, risk_score)))
            return risk_score, corr_matrix

        except Exception as e:
            print(f"      ⚠️ Graph Calculation Error: {e}")
            import traceback
            traceback.print_exc()
            return 0.5, None