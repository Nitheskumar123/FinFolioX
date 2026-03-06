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
    
    Uses SHAP GradientExplainer — fully compatible with multi-layer LSTM 
    architectures (Titan 3-layer: 256 → 128 → 64).
    
    Fallback: If GradientExplainer also fails, uses a lightweight 
    Perturbation-Based method that works with ANY model architecture.
    """
    
    def __init__(self, technical_agent, background_data_df):
        """
        Initializes the SHAP Explainer.
        """
        self.tech_agent = technical_agent
        self.device = technical_agent.device
        self.feature_names = ['Close', 'Volume', 'SMA_50', 'SMA_200', 'RSI', 'MACD']
        self.ready = False
        self.use_perturbation = False
        
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
            sample_size = min(20, len(sequences))
            indices = np.random.choice(len(sequences), sample_size, replace=False)
            bg_sample_seqs = sequences[indices]
            
            # 4. Convert to Tensor
            self.background_tensor = torch.FloatTensor(bg_sample_seqs).to(self.device)
            
            # 5. Try GradientExplainer first (works with complex architectures)
            try:
                self.explainer = shap.GradientExplainer(
                    self.tech_agent.model, 
                    self.background_tensor
                )
                self.ready = True
                print("      ✅ Explainability Agent (SHAP GradientExplainer) Ready.")
            except Exception as ge:
                print(f"      ⚠️ GradientExplainer failed: {ge}")
                print("      🔄 Falling back to Perturbation-Based Explainability...")
                self.use_perturbation = True
                self.ready = True
                # Store background for perturbation method
                self.bg_scaled = bg_scaled
                print("      ✅ Perturbation Explainability Agent Ready.")
            
        except Exception as e:
            print(f"      ⚠️ SHAP Initialization Failed: {e}")
            self.ready = False

    def explain_prediction(self, recent_sequence_df):
        """
        Calculates feature importance for the current market state.
        
        Returns:
            feature_importance (dict): { 'RSI': 0.15, 'Volume': -0.05, ... }
            top_driver (str): The feature driving the decision most.
        """
        if not self.ready:
            return {}, "SHAP Not Ready"

        if self.use_perturbation:
            return self._perturbation_explain(recent_sequence_df)

        try:
            # 1. Prepare Input Data
            data = recent_sequence_df[self.feature_names].values
            scaled_data = self.tech_agent.scaler.transform(data)
            
            # Shape: [1, 60, 6] (Batch size 1)
            input_tensor = torch.FloatTensor(scaled_data).view(1, 60, 6).to(self.device)
            input_tensor.requires_grad = True  # Required for GradientExplainer
            
            # 2. Compute SHAP Values
            shap_values = self.explainer.shap_values(input_tensor)
            
            # Handle different SHAP version outputs (list vs array)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            # Convert to numpy if tensor
            if isinstance(shap_values, torch.Tensor):
                shap_values = shap_values.detach().cpu().numpy()

            # Ensure correct shape dimensions
            if len(shap_values.shape) == 2:
                shap_values = shap_values[np.newaxis, :]
                
            # 3. Aggregate Importance with Time-Weighting
            timesteps = shap_values.shape[1]  # 60
            time_weights = np.linspace(0.5, 1.0, timesteps)
            
            # Weighted average across time axis
            weighted_shap = np.average(shap_values[0], axis=0, weights=time_weights)
            
            # 4. Map to Feature Names
            importance_dict = {}
            for i, feat in enumerate(self.feature_names):
                importance_dict[feat] = float(np.array(weighted_shap[i]).item())
                
            # 5. Identify Top Driver
            top_driver_idx = np.argmax(np.abs(weighted_shap))
            top_driver = self.feature_names[top_driver_idx]
            
            return importance_dict, top_driver

        except Exception as e:
            print(f"      ⚠️ SHAP GradientExplainer Error: {e}")
            print(f"      🔄 Falling back to perturbation method...")
            return self._perturbation_explain(recent_sequence_df)

    def _perturbation_explain(self, recent_sequence_df):
        """
        Lightweight fallback: perturb each feature by +/-5% and measure
        how much the model output changes. Higher delta = more important.
        Works with ANY PyTorch model architecture.
        """
        try:
            data = recent_sequence_df[self.feature_names].values
            scaled_data = self.tech_agent.scaler.transform(data)
            
            base_seq = torch.FloatTensor(scaled_data).view(1, 60, 6).to(self.device)
            
            with torch.no_grad():
                base_output = self.tech_agent.model(base_seq).item()
            
            importance_dict = {}
            for i, feat in enumerate(self.feature_names):
                perturbed = scaled_data.copy()
                perturbed[:, i] *= 1.05  # Perturb feature up by 5%
                
                perturbed_seq = torch.FloatTensor(perturbed).view(1, 60, 6).to(self.device)
                
                with torch.no_grad():
                    perturbed_output = self.tech_agent.model(perturbed_seq).item()
                
                importance_dict[feat] = round(perturbed_output - base_output, 6)
            
            # Identify top driver
            top_driver_idx = max(importance_dict, key=lambda k: abs(importance_dict[k]))
            
            return importance_dict, top_driver_idx
            
        except Exception as e:
            print(f"      ⚠️ Perturbation Explainability Error: {e}")
            return {}, "Error"