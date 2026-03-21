import numpy as np
import pandas as pd

class UncertaintyAgent:
    """
    Wraps the Technical Agent (LSTM) to estimate epistemic uncertainty.
    
    FIX: Replaced broken MC Dropout (training=True breaks BatchNorm) 
    and fake noise with probability-distance-from-center method.
    
    LOGIC: How far the LSTM probability is from 0.5 IS the uncertainty.
      - prob=0.99 → model is very confident → LOW uncertainty
      - prob=0.51 → model is barely leaning → HIGH uncertainty
    This is honest and architecturally correct for BiLSTM + BatchNorm models.
    """
    def __init__(self, technical_agent):
        self.tech_agent = technical_agent

    def predict_with_uncertainty(self, recent_data_df, n_iterations=10):
        from ml_engine.technical_agent import LSTM_COLS, build_lstm_features

        try:
            # Uses the exact same stretched probability as the decision engine
            raw_prob = float(self.tech_agent.predict(recent_data_df))

            # Distance from 0.5 = uncertainty (aligned with actual decision prob)
            distance_from_center = abs(raw_prob - 0.5)
            mc_std  = 0.5 - distance_from_center
            mc_mean = raw_prob

            return mc_mean, mc_std

        except Exception as e:
            print(f"      ⚠️ Uncertainty Agent Error: {e}")
            return 0.5, 0.15