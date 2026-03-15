import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

class ExplainabilityAgent:
    """
    Explains 'WHY' the Keras LSTM Brain made a specific decision.
    Uses a Perturbation-Based method for high stability.
    """
    
    def __init__(self, technical_agent, background_data_df):
        self.tech_agent = technical_agent
        
        self.feature_names = [
            'log_return', 'vol_change', 'sma10_dist', 
            'sma20_dist', 'sma50_dist', 'RSI', 'macd_norm'
        ]
        self.ready = True
        print("      ✅ Explainability Agent (Keras Perturbation) Ready.")

    def explain_prediction(self, recent_sequence_df):
        if not self.ready:
            return {}, "Not Ready"

        try:
            # 1. Prepare Input Data (100 days, 7 features)
            data = recent_sequence_df[self.feature_names].values
            
            # ✅ FIX: Use lstm_scaler
            scaled_data = self.tech_agent.lstm_scaler.transform(data)
            
            base_seq = scaled_data.reshape(1, 100, len(self.feature_names))
            
            # ✅ FIX: Use lstm_model for base output
            base_output = float(self.tech_agent.lstm_model.predict(base_seq, verbose=0)[0][0])
            
            importance_dict = {}
            
            # 3. Perturb each feature by 5% and measure the delta
            for i, feat in enumerate(self.feature_names):
                perturbed = scaled_data.copy()
                perturbed[:, i] *= 1.05  # Bump feature up by 5%
                
                perturbed_seq = perturbed.reshape(1, 100, len(self.feature_names))
                
                # ✅ FIX: Use lstm_model for perturbed output
                perturbed_output = float(self.tech_agent.lstm_model.predict(perturbed_seq, verbose=0)[0][0])
                
                # The importance is how much the output changed
                importance_dict[feat] = round(perturbed_output - base_output, 6)
            
            # 4. Identify the feature with the largest absolute impact
            top_driver = max(importance_dict, key=lambda k: abs(importance_dict[k]))
            
            return importance_dict, top_driver
            
        except Exception as e:
            print(f"      ⚠️ Keras Explainability Error: {e}")
            return {}, "Error"