import pandas as pd
import numpy as np
import os
import sys
import joblib
from sklearn.preprocessing import StandardScaler

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_engine.regime_agent import RegimeAgent

# CONFIG
DATA_PATH = os.path.join("data", "processed", "AAPL_features.csv")
SAVE_PATH = os.path.join("saved_models", "hmm_regime.pkl")
SCALER_PATH = os.path.join("saved_models", "regime_scaler.pkl")

def prepare_data():
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: Processed data not found. Run Phase 2 scripts first.")
        return None, None
        
    df = pd.read_csv(DATA_PATH, index_col=0)
    
    # 1. Feature Engineering
    df['Returns'] = df['Close'].pct_change()
    df['Volatility'] = df['Returns'].rolling(window=10).std()
    df.dropna(inplace=True)
    
    # 2. Extract Raw Values
    features = df[['Returns', 'Volatility']].values
    
    # 3. Normalize Data (CRITICAL IMPROVEMENT)
    # HMMs work best when data is centered around 0 with std dev of 1
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    
    # Save the scaler so we can use it on Live Data later
    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    print(f"💾 Regime Scaler saved to {SCALER_PATH}")
    
    return X_scaled, df

def train_hmm():
    print("🚀 STARTING PHASE 6: REGIME DETECTION TRAINING")
    print("-" * 50)
    
    X, df = prepare_data()
    if X is None: return

    # Initialize and Train
    agent = RegimeAgent()
    agent.train(X)
    
    # Save
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    agent.save(SAVE_PATH)
    
    # --- VERIFICATION ---
    print("\n🧐 TESTING ON RECENT DATA (Last 5 Days)")
    last_5_X = X[-5:]
    last_5_Dates = df.index[-5:]
    
    for i, (day_data, date) in enumerate(zip(last_5_X, last_5_Dates)):
        # Get the human-readable label (Bull/Bear)
        label = agent.get_regime_label(day_data)
        print(f"   {date}: Detected Regime -> {label}")

    print("-" * 50)
    print("✅ PHASE 6 COMPLETE. The Weather Station is Online.")

if __name__ == "__main__":
    train_hmm()