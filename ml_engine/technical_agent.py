import torch
import torch.nn as nn
import joblib
import numpy as np
import pandas as pd
import os

# --- DEFINE THE LSTM ARCHITECTURE ---
# This must match EXACTLY what you trained in Phase 3.
# Standard Architecture: 6 Inputs -> 64 Hidden -> 2 Layers -> 1 Output
class LSTMModel(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM Layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=0.2)
        
        # Fully Connected Layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        # Take the output of the last time step
        out = self.fc(out[:, -1, :])
        return out

class TechnicalAgent:
    def __init__(self, model_path, scaler_path):
        """
        The Technical Agent reads charts using the LSTM and Scaler.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Load the Scaler
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            print(f"✅ Scaler loaded from {scaler_path}")
        else:
            raise FileNotFoundError(f"❌ Scaler not found at {scaler_path}")

        # 2. Load the Model
        self.model = LSTMModel(input_size=6, hidden_size=64, num_layers=2).to(self.device)
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print(f"✅ Technical Agent (LSTM) Loaded from {model_path}")
        else:
            raise FileNotFoundError(f"❌ Model not found at {model_path}")

    def predict(self, recent_data_df):
        """
        Input: DataFrame with last 60 rows containing:
               ['Close', 'Volume', 'SMA_50', 'SMA_200', 'RSI', 'MACD']
        Output: A prediction confidence score (0.0 to 1.0)
        """
        # Ensure we have exactly 60 rows
        if len(recent_data_df) != 60:
            print(f"⚠️ Warning: Expected 60 rows, got {len(recent_data_df)}. Prediction might be inaccurate.")
        
        # Select features in the exact order used during training
        features = ['Close', 'Volume', 'SMA_50', 'SMA_200', 'RSI', 'MACD']
        try:
            data = recent_data_df[features].values
        except KeyError as e:
            raise KeyError(f"❌ Missing columns in input data: {e}")

        # 1. Scale the data
        scaled_data = self.scaler.transform(data)
        
        # 2. Convert to Tensor (Batch Size=1, Sequence Length=60, Features=6)
        seq = torch.FloatTensor(scaled_data).view(1, 60, 6).to(self.device)
        
        # 3. Predict
        with torch.no_grad():
            raw_output = self.model(seq).item()
            
            # If your model outputs raw prices, we map it to a confidence score 
            # (Simplification for inference: Sigmoid squashes it to 0-1)
            confidence = torch.sigmoid(torch.tensor(raw_output)).item()
            
        return confidence