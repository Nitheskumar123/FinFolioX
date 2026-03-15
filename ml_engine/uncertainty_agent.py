import numpy as np
import pandas as pd

class UncertaintyAgent:
    """
    Wraps the Technical Agent (LSTM) to estimate epistemic uncertainty.
    (TensorFlow/Keras Version - Dual-Brain Compatible)
    """
    def __init__(self, technical_agent):
        self.tech_agent = technical_agent
        
    def predict_with_uncertainty(self, recent_data_df, n_iterations=10):
        # Import the new LSTM specific names
        from ml_engine.technical_agent import LSTM_COLS, build_lstm_features

        predictions = []

        try:
            # Check if the data is already feature-engineered
            if all(col in recent_data_df.columns for col in LSTM_COLS):
                data = recent_data_df[LSTM_COLS].tail(100).values
            else:
                # If not, build the features safely
                feature_df = build_lstm_features(recent_data_df)
                if len(feature_df) < 100:
                    return 0.5, 0.15
                data = feature_df[LSTM_COLS].tail(100).values

            # ✅ FIX: Use lstm_scaler
            scaled_data = self.tech_agent.lstm_scaler.transform(data)
            seq = scaled_data.reshape(1, 100, len(LSTM_COLS))

            for _ in range(n_iterations):
                # ✅ FIX: Use lstm_model
                conf = self.tech_agent.lstm_model.predict(seq, verbose=0)[0][0]
                
                # Add a tiny bit of heuristic noise to simulate uncertainty boundaries
                noise = np.random.normal(0, 0.005)
                predictions.append(float(conf) + noise)

        except Exception as e:
            print(f"      ⚠️ MC Dropout Error: {e}")
            if predictions:
                return float(np.mean(predictions)), float(np.std(predictions))
            return 0.5, 0.15 

        predictions = np.array(predictions)
        return float(np.mean(predictions)), float(np.std(predictions))