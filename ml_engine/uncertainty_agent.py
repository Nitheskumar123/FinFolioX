import torch
import numpy as np
import pandas as pd

class UncertaintyAgent:
    """
    Wraps the Technical Agent (LSTM) to estimate epistemic uncertainty using 
    Monte Carlo Dropout (Gal & Ghahramani, 2016).
    
    This agent runs the model multiple times with random neurons disabled to 
    generate a distribution of predictions. The spread (StdDev) of this 
    distribution indicates how "confused" the model is.
    """
    def __init__(self, technical_agent):
        # We share the model instance to save memory
        self.tech_agent = technical_agent
        self.device = technical_agent.device
        
    def predict_with_uncertainty(self, recent_data_df, n_iterations=50):
        """
        Runs the LSTM model n_iterations times with Dropout ENABLED.
        
        Args:
            recent_data_df (pd.DataFrame): The preprocessed OHLCV data.
            n_iterations (int): Number of Monte Carlo passes (Default: 50).
            
        Returns:
            bayesian_mean (float): The average confidence score.
            uncertainty (float): The standard deviation (Risk).
        """
        # 1. Force Model into 'Train' mode
        # This enables the Dropout layers, which are normally off during inference.
        self.tech_agent.model.train() 
        
        predictions = []
        
        # 2. Prepare Data (Same logic as Technical Agent)
        features = ['Close', 'Volume', 'SMA_50', 'SMA_200', 'RSI', 'MACD']
        try:
            data = recent_data_df[features].values
            scaled_data = self.tech_agent.scaler.transform(data)
            
            # Reshape for LSTM (Batch=1, Seq=60, Feat=6)
            seq = torch.FloatTensor(scaled_data).view(1, 60, 6).to(self.device)
            
            # 3. Monte Carlo Loop
            for i in range(n_iterations):
                with torch.no_grad():
                    # Every pass uses a different random sub-network configuration
                    raw_out = self.tech_agent.model(seq).item()
                    
                    # Convert raw logit to probability (0-1) using Sigmoid
                    conf = torch.sigmoid(torch.tensor(raw_out)).item()
                    predictions.append(conf)
                    
        except Exception as e:
            print(f"      ⚠️ MC Dropout Error: {e}")
            return 0.5, 1.0 # Return High Uncertainty on error
            
        finally:
            # CRITICAL: Switch model back to Eval mode so we don't break standard predictions
            self.tech_agent.model.eval()
            
        # 4. Calculate Bayesian Statistics
        predictions = np.array(predictions)
        bayesian_mean = np.mean(predictions)
        uncertainty = np.std(predictions) # The "Spread" of answers
        
        return bayesian_mean, uncertainty