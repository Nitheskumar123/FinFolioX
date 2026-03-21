import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ==============================================================================
# ARCHITECTURE A — Kaggle P100 model (legacy)
# ==============================================================================
class KaggleFusion(nn.Module):
    def __init__(self, d_model=64, nhead=8, dropout=0.17):
        super().__init__()
        assert d_model % nhead == 0

        self.lstm_proj = nn.Linear(1, d_model)
        self.sent_proj = nn.Linear(1, d_model)
        self.vol_proj  = nn.Linear(1, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, 3, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2), 
            nn.ReLU(), 
            nn.Dropout(dropout), 
            nn.Linear(d_model // 2, 1), 
            nn.Sigmoid(), 
        )

    def forward(self, x_lstm, x_sent, x_vol):
        t_lstm  = self.lstm_proj(x_lstm).unsqueeze(1)
        t_sent  = self.sent_proj(x_sent).unsqueeze(1)
        t_vol   = self.vol_proj(x_vol).unsqueeze(1)
        tokens  = torch.cat([t_lstm, t_sent, t_vol], dim=1) + self.pos_embed
        enc     = self.transformer(tokens)
        pooled  = enc.mean(dim=1) 
        conf    = self.decoder(pooled)
        dummy   = torch.zeros(enc.size(0), 3, 3, device=x_lstm.device)
        return conf, dummy


# ==============================================================================
# ARCHITECTURE B — Local synthetic model (New Advanced Version)
# Keys: lstm_embed, sent_embed, vol_embed, attention.*, fc1, fc2
# ==============================================================================
class MultiHeadFusion(nn.Module):
    def __init__(self, d_model=16, nhead=4):
        assert d_model % nhead == 0
        super().__init__()
        self.lstm_embed = nn.Linear(1, d_model)
        self.sent_embed = nn.Linear(1, d_model)
        self.vol_embed  = nn.Linear(1, d_model)
        self.attention  = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        self.fc1        = nn.Linear(d_model * 3, 32)
        self.dropout    = nn.Dropout(0.2)
        self.fc2        = nn.Linear(32, 1)
        self.sigmoid    = nn.Sigmoid()

    def forward(self, lstm_pred, sentiment_score, volatility):
        e_lstm    = F.relu(self.lstm_embed(lstm_pred)).unsqueeze(1)
        e_sent    = F.relu(self.sent_embed(sentiment_score)).unsqueeze(1)
        e_vol     = F.relu(self.vol_embed(volatility)).unsqueeze(1)
        sequence  = torch.cat((e_lstm, e_sent, e_vol), dim=1)
        attn_out, attn_w = self.attention(sequence, sequence, sequence)
        x         = F.relu(self.fc1(attn_out.reshape(attn_out.size(0), -1)))
        x         = self.dropout(x)
        return self.sigmoid(self.fc2(x)), attn_w


# ==============================================================================
# FUSION AGENT — auto-detects architecture from checkpoint keys
# ==============================================================================
class FusionAgent:
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = None
        self._norm_stats = None

        if model_path:
            self._load(model_path)
        else:
            self.model = KaggleFusion().to(self.device)

        if self.model is not None:
            self.model.eval()

    def _load(self, model_path):
        try:
            checkpoint = torch.load(
                model_path, map_location=self.device, weights_only=False
            )

            # Unwrap Kaggle-style dict
            if isinstance(checkpoint, dict) and "model_state" in checkpoint:
                state_dict         = checkpoint["model_state"]
                hp                 = checkpoint.get("hyperparameters", {})
                self._norm_stats   = checkpoint.get("normalization_stats", None)
            else:
                state_dict = checkpoint
                hp         = {}

            keys = set(state_dict.keys())

            if "lstm_proj.weight" in keys:
                # ── Kaggle architecture ───────────────────────────────
                d_model = state_dict["lstm_proj.weight"].shape[0]
                nhead   = hp.get("nhead",   8)
                dropout = hp.get("dropout", 0.17)
                self.model = KaggleFusion(
                    d_model=d_model, nhead=nhead, dropout=dropout
                ).to(self.device)
                print(
                    f"      ℹ️  Kaggle architecture detected "
                    f"(d_model={d_model}, nhead={nhead})"
                )

            elif "lstm_embed.weight" in keys:
                # ── Local MultiHeadFusion architecture ────────────────
                d_model = state_dict["lstm_embed.weight"].shape[0]
                
                # ✅ FIX: Hardcoded to match your training script!
                nhead = 4 
                
                self.model = MultiHeadFusion(
                    d_model=d_model, nhead=nhead
                ).to(self.device)
                print(
                    f"      ℹ️  Local architecture detected "
                    f"(d_model={d_model}, nhead={nhead})"
                )

            else:
                raise ValueError(
                    f"Unknown checkpoint format. First 5 keys: {list(keys)[:5]}"
                )

            self.model.load_state_dict(state_dict)
            norm_note = "  (norm stats loaded)" if self._norm_stats else ""
            print(f"✅ Advanced Fusion Agent Loaded from {model_path}{norm_note}")

        except FileNotFoundError:
            print("⚠️ No trained fusion model found. Using default weights.")
            self.model = KaggleFusion().to(self.device)
        except Exception as e:
            print(f"❌ Critical Error loading Fusion Agent: {e}")
            raise

    def _normalize_input(self, val, key):
        """Apply Kaggle z-score normalization if stats are available."""
        if self._norm_stats is None:
            return val
        mu    = self._norm_stats.get(f"{key}_mean", 0.0)
        sigma = self._norm_stats.get(f"{key}_std",  1.0)
        return (val - mu) / (sigma + 1e-8)

    def interpret_weights(self, attn_weights):
        w = attn_weights.mean(dim=0).cpu().numpy()
        return {
            "LSTM_Focus":       float(np.mean(w[:, 0])),
            "Sentiment_Focus":  float(np.mean(w[:, 1])),
            "Volatility_Focus": float(np.mean(w[:, 2])),
        }

    def predict(self, lstm_p, sent_s, vol_v, trust_scores=None):
        if trust_scores:
            lstm_p = lstm_p * trust_scores.get("technical", 1.0)
            sent_s = sent_s * trust_scores.get("sentiment", 1.0)
            vol_v  = vol_v  * trust_scores.get("regime",    1.0)

        # Normalize inputs if Kaggle norm stats present
        lstm_n = self._normalize_input(lstm_p, "lstm")
        sent_n = self._normalize_input(sent_s, "sent")
        vol_n  = self._normalize_input(vol_v,  "vol")

        t_lstm = torch.tensor([[lstm_n]], dtype=torch.float32).to(self.device)
        t_sent = torch.tensor([[sent_n]], dtype=torch.float32).to(self.device)
        t_vol  = torch.tensor([[vol_n]],  dtype=torch.float32).to(self.device)

        with torch.no_grad():
            conf, weights = self.model(t_lstm, t_sent, t_vol)

        focus_map  = self.interpret_weights(weights)
        final_conf = conf.item()

        if final_conf < 0.35:
            print(
                f"      🔴 [Fusion Agent] Neural weight collapse "
                f"({final_conf:.4f}). Clamping to 0.35."
            )
            final_conf = 0.35

        return final_conf, focus_map