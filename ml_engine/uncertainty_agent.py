import torch
import numpy as np
import pandas as pd

class UncertaintyAgent:
    """
    Wraps the Technical Agent (LSTM) to estimate epistemic uncertainty using 
    Monte Carlo Dropout.
    """
    def __init__(self, technical_agent):
        self.tech_agent = technical_agent
        self.device = technical_agent.device
        
    def predict_with_uncertainty(self, recent_data_df, n_iterations=50):
        self.tech_agent.model.train() 
        
        predictions = []
        features = ['Close', 'Volume', 'SMA_50', 'SMA_200', 'RSI', 'MACD']
        
        try:
            data = recent_data_df[features].values
            scaled_data = self.tech_agent.scaler.transform(data)
            seq = torch.FloatTensor(scaled_data).view(1, 60, 6).to(self.device)
            
            for i in range(n_iterations):
                with torch.no_grad():
                    raw_out = self.tech_agent.model(seq).item()
                    # Old math: Sigmoid mapping
                    conf = torch.sigmoid(torch.tensor(raw_out)).item()
                    predictions.append(conf)
                    
        except Exception as e:
            print(f"      ⚠️ MC Dropout Error: {e}")
            return 0.5, 1.0 
            
        finally:
            self.tech_agent.model.eval()
            
        predictions = np.array(predictions)
        bayesian_mean = np.mean(predictions)
        uncertainty = np.std(predictions) 
        
        return bayesian_mean, uncertainty