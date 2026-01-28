import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MultiHeadFusion(nn.Module):
    def __init__(self, d_model=16, nhead=4):
        """
        Hierarchical Multi-Head Attention Fusion Engine.
        
        Args:
            d_model (int): Dimension of the internal representation.
            nhead (int): Number of 'heads' (parallel attention mechanisms).
        """
        # --- VALIDATION CHECK ---
        # Ensures that the dimensions align, preventing crashes during initialization.
        assert d_model % nhead == 0, f"d_model ({d_model}) must be divisible by nhead ({nhead})"
        
        super(MultiHeadFusion, self).__init__()
        
        # 1. EMBEDDING LAYERS
        # Projects scalar inputs (1 dim) into vector space (d_model dim)
        self.lstm_embed = nn.Linear(1, d_model)
        self.sent_embed = nn.Linear(1, d_model)
        self.vol_embed = nn.Linear(1, d_model)
        
        # 2. MULTI-HEAD ATTENTION
        # The core brain that learns relationships between the inputs
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        
        # 3. DECISION LAYERS
        self.fc1 = nn.Linear(d_model * 3, 32) 
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(32, 1) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, lstm_pred, sentiment_score, volatility):
        # Step A: Embed inputs
        e_lstm = F.relu(self.lstm_embed(lstm_pred)).unsqueeze(1)
        e_sent = F.relu(self.sent_embed(sentiment_score)).unsqueeze(1)
        e_vol = F.relu(self.vol_embed(volatility)).unsqueeze(1)
        
        # Step B: Stack sequence [LSTM, Sentiment, Volatility]
        sequence = torch.cat((e_lstm, e_sent, e_vol), dim=1)
        
        # Step C: Attention
        attn_output, attn_weights = self.attention(sequence, sequence, sequence)
        
        # Step D: Decision
        flattened = attn_output.reshape(attn_output.size(0), -1)
        x = F.relu(self.fc1(flattened))
        x = self.dropout(x)
        final_confidence = self.sigmoid(self.fc2(x))
        
        return final_confidence, attn_weights

# --- WRAPPER CLASS FOR DJANGO ---
class FusionAgent:
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MultiHeadFusion().to(self.device)
        
        if model_path:
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"✅ Advanced Fusion Agent Loaded from {model_path}")
            except FileNotFoundError:
                print("⚠️ No trained fusion model found. Using random weights.")
        
        self.model.eval()

    def interpret_weights(self, attn_weights):
        """
        Helper to make sense of the attention matrix.
        Returns dictionary showing how much focus is on each component.
        """
        # Average across heads and batch
        avg_weights = attn_weights.mean(dim=0).cpu().numpy() # Shape (3,3)
        
        # Simplify: How much did the model look at LSTM vs Sentiment?
        # Row 0 is LSTM, Row 1 is Sentiment, Row 2 is Volatility
        focus_map = {
            "LSTM_Focus": float(np.mean(avg_weights[:, 0])),
            "Sentiment_Focus": float(np.mean(avg_weights[:, 1])),
            "Volatility_Focus": float(np.mean(avg_weights[:, 2]))
        }
        return focus_map

    def predict(self, lstm_p, sent_s, vol_v):
        t_lstm = torch.tensor([[lstm_p]], dtype=torch.float32).to(self.device)
        t_sent = torch.tensor([[sent_s]], dtype=torch.float32).to(self.device)
        t_vol = torch.tensor([[vol_v]], dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            conf, weights = self.model(t_lstm, t_sent, t_vol)
            
        interpretation = self.interpret_weights(weights)
            
        return conf.item(), interpretation