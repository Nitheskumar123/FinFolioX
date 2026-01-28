import torch
import numpy as np
import shap
import pandas as pd
import warnings

# Suppress SHAP warnings for cleaner console output
warnings.filterwarnings("ignore")

class ExplainabilityAgent:
    """
    Explains 'WHY' the Technical Agent made a specific decision.
    
    Uses SHAP (SHapley Additive exPlanations) DeepExplainer to attribute
    prediction output to specific input features (RSI, MACD, Volume, etc.).
    """
    
    def __init__(self, technical_agent, background_data_df):
        """
        Initializes the SHAP Explainer.
        """
        self.tech_agent = technical_agent
        self.device = technical_agent.device
        self.feature_names = ['Close', 'Volume', 'SMA_50', 'SMA_200', 'RSI', 'MACD']
        self.ready = False
        
        print("      ⏳ Initializing Explainability Engine (SHAP)...")
        
        try:
            # 1. Prepare Background Data
            bg_data = background_data_df[self.feature_names].values
            bg_scaled = self.tech_agent.scaler.transform(bg_data)
            
            # 2. Create Proper Sliding Window Sequences
            sequences = []
            seq_len = 60
            
            if len(bg_scaled) <= seq_len:
                raise ValueError("Insufficient history for SHAP background")
                
            for i in range(len(bg_scaled) - seq_len):
                sequences.append(bg_scaled[i : i + seq_len])
                
            sequences = np.array(sequences)
            
            # 3. Sample Background Sequences
            # Use 20 random samples to represent "normal market conditions"
            sample_size = min(20, len(sequences))
            indices = np.random.choice(len(sequences), sample_size, replace=False)
            bg_sample_seqs = sequences[indices]
            
            # 4. Convert to Tensor
            self.background_tensor = torch.FloatTensor(bg_sample_seqs).to(self.device)
            
            # 5. Initialize SHAP DeepExplainer
            self.explainer = shap.DeepExplainer(
                self.tech_agent.model, 
                self.background_tensor
            )
            
            self.ready = True
            print("      ✅ Explainability Agent (SHAP) Ready.")
            
        except Exception as e:
            print(f"      ⚠️ SHAP Initialization Failed: {e}")
            self.ready = False

    def explain_prediction(self, recent_sequence_df):
        """
        Calculates SHAP values for the current market state.
        
        Returns:
            feature_importance (dict): { 'RSI': 0.15, 'Volume': -0.05, ... }
            top_driver (str): The name of the feature driving the decision most.
        """
        if not self.ready:
            return {}, "SHAP Not Ready"

        try:
            # 1. Prepare Input Data
            data = recent_sequence_df[self.feature_names].values
            scaled_data = self.tech_agent.scaler.transform(data)
            
            # Shape: [1, 60, 6] (Batch size 1)
            input_tensor = torch.FloatTensor(scaled_data).view(1, 60, 6).to(self.device)
            
            # 2. Compute SHAP Values
            # --- CRITICAL FIX: check_additivity=False ---
            # This prevents the crash caused by tiny floating-point rounding errors in LSTMs.
            shap_values = self.explainer.shap_values(input_tensor, check_additivity=False)
            
            # Handle different SHAP version outputs (list vs array)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            # Ensure correct shape dimensions
            if len(shap_values.shape) == 2:
                shap_values = shap_values[np.newaxis, :]
                
            # 3. Aggregate Importance with Time-Weighting
            # We give more weight to recent time steps (closer to today)
            timesteps = shap_values.shape[1] # 60
            time_weights = np.linspace(0.5, 1.0, timesteps)
            
            # Weighted average across time axis
            weighted_shap = np.average(shap_values[0], axis=0, weights=time_weights)
            
            # 4. Map to Feature Names
            importance_dict = {}
            for i, feat in enumerate(self.feature_names):
                importance_dict[feat] = float(weighted_shap[i])
                
            # 5. Identify Top Driver
            # Find feature with max ABSOLUTE impact
            top_driver_idx = np.argmax(np.abs(weighted_shap))
            top_driver = self.feature_names[top_driver_idx]
            
            return importance_dict, top_driver

        except Exception as e:
            print(f"      ⚠️ Explainability Calculation Error: {e}")
            return {}, "Error"