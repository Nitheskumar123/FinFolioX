import os
import joblib
import numpy as np
import pandas as pd
import ta
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers

# ============================================================
# CUSTOM TRANSFORMER LAYERS (Required for loading the model)
# ============================================================
@tf.keras.utils.register_keras_serializable(package='Custom')
class PositionalEncoding(layers.Layer):
    def __init__(self, max_len=100, d_model=64, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model
        
    def build(self, input_shape):
        position = np.arange(self.max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model))
        pe = np.zeros((self.max_len, self.d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = tf.constant(pe[np.newaxis, :, :], dtype=tf.float32)
        super().build(input_shape)
        
    def call(self, x):
        return x + self.pe[:, :tf.shape(x)[1], :]
        
    def get_config(self):
        config = super().get_config()
        config.update({'max_len': self.max_len, 'd_model': self.d_model})
        return config

@tf.keras.utils.register_keras_serializable(package='Custom')
class TransformerBlock(layers.Layer):
    def __init__(self, d_model=64, num_heads=4, ff_dim=128, dropout_rate=0.3, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout_rate)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation='gelu'),
            layers.Dropout(dropout_rate),
            layers.Dense(d_model),
            layers.Dropout(dropout_rate)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        
    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1, training=training)
        return self.layernorm2(out1 + ffn_output)
        
    def get_config(self):
        config = super().get_config()
        config.update({'d_model': self.d_model, 'num_heads': self.num_heads, 'ff_dim': self.ff_dim, 'dropout_rate': self.dropout_rate})
        return config

# ============================================================
# LSTM FEATURE BUILDER (7 Features)
# ============================================================
LSTM_COLS = ['log_return', 'vol_change', 'sma10_dist', 'sma20_dist', 'sma50_dist', 'RSI', 'macd_norm']

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(com=period - 1, min_periods=period).mean()
    loss = -delta.clip(upper=0).ewm(com=period - 1, min_periods=period).mean()
    return 100 - (100 / (1 + gain / (loss + 1e-9)))

def compute_macd(series, fast=12, slow=26, signal=9):
    macd_line = series.ewm(span=fast, adjust=False).mean() - series.ewm(span=slow, adjust=False).mean()
    return macd_line - macd_line.ewm(span=signal, adjust=False).mean()

def build_lstm_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    out["vol_change"] = df["Volume"].pct_change().clip(-5.0, 5.0)
    out["sma10_dist"] = (df["Close"] - df["Close"].rolling(10).mean()) / df["Close"].rolling(10).mean()
    out["sma20_dist"] = (df["Close"] - df["Close"].rolling(20).mean()) / df["Close"].rolling(20).mean()
    out["sma50_dist"] = (df["Close"] - df["Close"].rolling(50).mean()) / df["Close"].rolling(50).mean()
    out["RSI"] = compute_rsi(df["Close"])
    out["macd_norm"] = compute_macd(df["Close"]) / df["Close"]
    return out.replace([np.inf, -np.inf], np.nan).dropna()

# ============================================================
# TRANSFORMER FEATURE BUILDER (15 Features)
# ============================================================
TRANSFORMER_COLS = [
    'log_return', 'log_return_5d', 'log_return_20d',
    'sma20_dist', 'sma50_dist', 'sma200_dist', 'bb_position',
    'rsi_norm', 'macd_hist_norm', 'stoch_norm', 'williams_norm',
    'atr_pct', 'vol_ratio', 'vol_change', 'obv_roc'
]

def build_transformer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    out['log_return_5d'] = np.log(df['Close'] / df['Close'].shift(5))
    out['log_return_20d'] = np.log(df['Close'] / df['Close'].shift(20))
    
    out['sma20_dist'] = (df['Close'] - df['Close'].rolling(20).mean()) / df['Close'].rolling(20).mean()
    out['sma50_dist'] = (df['Close'] - df['Close'].rolling(50).mean()) / df['Close'].rolling(50).mean()
    out['sma200_dist'] = (df['Close'] - df['Close'].rolling(200).mean()) / df['Close'].rolling(200).mean()
    
    bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    out['bb_position'] = (df['Close'] - bb.bollinger_mavg()) / (bb.bollinger_hband() - bb.bollinger_lband() + 1e-8)
    
    rsi = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    out['rsi_norm'] = (rsi - 50) / 50
    
    macd = ta.trend.MACD(df['Close'])
    out['macd_hist_norm'] = macd.macd_diff() / df['Close']
    
    stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
    out['stoch_norm'] = (stoch.stoch() - 50) / 50
    
    williams = ta.momentum.WilliamsRIndicator(df['High'], df['Low'], df['Close'])
    out['williams_norm'] = (williams.williams_r() + 50) / 50
    
    atr = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=14)
    out['atr_pct'] = atr.average_true_range() / df['Close']
    
    vol_10 = df['Close'].pct_change().rolling(10).std()
    vol_60 = df['Close'].pct_change().rolling(60).std()
    out['vol_ratio'] = vol_10 / (vol_60 + 1e-8)
    
    out['vol_change'] = np.log(df['Volume'] / df['Volume'].rolling(20).mean() + 1e-8)
    
    obv = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
    out['obv_roc'] = obv.pct_change(periods=10).clip(-1, 1) 
    
    return out.replace([np.inf, -np.inf], np.nan).dropna()


# ============================================================
# DUAL-BRAIN TECHNICAL AGENT
# ============================================================
class TechnicalAgent:
    def __init__(self, lstm_model_path: str, lstm_scaler_path: str, trans_model_path: str, trans_scaler_path: str):
        # 1. Load LSTM
        self.lstm_scaler = joblib.load(lstm_scaler_path)
        self.lstm_model = load_model(lstm_model_path)
        print("      ✅ Brain 1: Keras LSTM Loaded")

        # 2. Load Transformer
        self.trans_scaler = joblib.load(trans_scaler_path)
        
        try:
            # Pass custom objects to ensure perfect loading
            custom_objs = {
                'PositionalEncoding': PositionalEncoding,
                'TransformerBlock': TransformerBlock
            }
            self.trans_model = load_model(trans_model_path, custom_objects=custom_objs, compile=False)
            print("      ✅ Brain 2: CNN-Transformer Loaded")
        except Exception as e:
            print(f"      ❌ Transformer Load Error: {e}")
            self.trans_model = None

    def predict(self, recent_data_df: pd.DataFrame) -> float:
        # --- BRAIN 1: LSTM PREDICTION ---
        lstm_df = build_lstm_features(recent_data_df)
        if len(lstm_df) < 100:
            return 0.5000
            
        lstm_data = self.lstm_scaler.transform(lstm_df[LSTM_COLS].tail(100).values)
        lstm_seq = lstm_data.reshape(1, 100, len(LSTM_COLS))
        lstm_prob = float(self.lstm_model.predict(lstm_seq, verbose=0)[0][0])
        
        # --- BRAIN 2: TRANSFORMER PREDICTION ---
        if self.trans_model is None:
            return lstm_prob
            
        trans_df = build_transformer_features(recent_data_df)
        if len(trans_df) < 100:
            return lstm_prob
            
        trans_data = self.trans_scaler.transform(trans_df[TRANSFORMER_COLS].tail(100).values)
        trans_data = np.clip(np.nan_to_num(trans_data, nan=0.0), -5, 5) 
        trans_seq = trans_data.reshape(1, 100, len(TRANSFORMER_COLS))
        trans_prob = float(self.trans_model.predict(trans_seq, verbose=0)[0][0])
        
        print(f"      - LSTM Brain        : {lstm_prob:.4f}")
        print(f"      - Transformer Brain : {trans_prob:.4f}")
        
        # --- FUSION: THE DUAL BRAIN ---
        return (lstm_prob + trans_prob) / 2.0

    def predict_signal(self, recent_data_df: pd.DataFrame) -> float:
        return self.predict(recent_data_df)