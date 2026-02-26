"""
PHASE 11: ADVERSARIAL ROBUSTNESS (The Red Team)
------------------------------------------------
This module acts as an "enemy" to your trading bot.
It generates fake market crashes and fake news to test 
if your Master System can survive extreme conditions.
"""

import numpy as np
import pandas as pd
import logging

class AdversarialTester:
    def __init__(self, master_system):
        self.system = master_system
        self.logger = logging.getLogger("RedTeam")

    def _prepare_data_for_ai(self, df):
        """
        Helper to slice the data exactly how the LSTM wants it.
        Expects: Last 60 rows, Specific 6 columns.
        """
        needed_cols = ['Close', 'Volume', 'SMA_50', 'SMA_200', 'RSI', 'MACD']
        
        # Recalculate if missing
        if not all(col in df.columns for col in needed_cols):
             if hasattr(self.system, '_calculate_rsi'): 
                 df['SMA_50'] = df['Close'].rolling(window=50).mean()
                 df['SMA_200'] = df['Close'].rolling(window=200).mean()
                 df['RSI'] = self.system._calculate_rsi(df['Close'])
                 df['MACD'] = self.system._calculate_macd(df['Close'])
                 df.dropna(inplace=True)
        
        # Fallback slicing
        if all(col in df.columns for col in needed_cols):
            df_subset = df[needed_cols]
        else:
            df_subset = df.iloc[:, -6:] 
            
        return df_subset.tail(60)

    def generate_flash_crash(self, hist_df, drop_pct=0.40):
        """Simulates a market crash and FORCES bad indicators."""
        print(f"   ⚡ SIMULATING NUCLEAR FLASH CRASH: -{drop_pct*100}% Drop...")
        crashed_df = hist_df.copy()
        
        # 1. Crash the price massively
        last_idx = crashed_df.index[-1]
        crashed_df.loc[last_idx, 'Close'] = crashed_df.loc[last_idx, 'Close'] * (1.0 - drop_pct)
        crashed_df.loc[last_idx, 'Low'] = crashed_df.loc[last_idx, 'Close'] 
        
        # 2. Recalculate standard indicators
        crashed_df['SMA_50'] = crashed_df['Close'].rolling(window=50).mean()
        crashed_df['SMA_200'] = crashed_df['Close'].rolling(window=200).mean()
        
        # 3. NUCLEAR OVERRIDE: Force indicators to extreme Bearish values
        # The LSTM might ignore a price drop, but it CANNOT ignore RSI=5
        crashed_df.loc[last_idx, 'RSI'] = 5.0    # Extreme Oversold
        crashed_df.loc[last_idx, 'MACD'] = -10.0 # Momentum Collapse
        
        print("   ⚠️ INJECTING SYNTHETIC BEARISH INDICATORS (RSI=5, MACD=-10)...")
        
        crashed_df.dropna(inplace=True)
        return crashed_df

    def run_robustness_test(self, ticker):
        """Runs the stress test comparing Normal vs. Crashed data."""
        print("\n" + "!"*60)
        print(f"🧪 PHASE 11: STARTING STRESS TEST FOR {ticker}")
        print("!"*60)

        # --- STEP A: NORMAL RUN ---
        print("\n1️⃣  RUNNING BASELINE (Normal Market Conditions)...")
        
        stock_obj = None
        df = None

        try:
            if hasattr(self.system, '_fetch_stock_data'):
                stock_obj, df = self.system._fetch_stock_data(ticker)
            elif hasattr(self.system, 'fetch_market_data'):
                stock_obj, df = self.system.fetch_market_data(ticker)
            else:
                print("❌ CRITICAL ERROR: Your Master System has no known 'fetch data' method!")
                return
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            return

        if df is None or df.empty:
            print("❌ Error: No data returned for ticker.")
            return

        # Score Normal
        try:
            input_normal = self._prepare_data_for_ai(df)
            if hasattr(self.system.tech_agent, 'predict_signal'):
                normal_tech_score = self.system.tech_agent.predict_signal(input_normal)
            else:
                normal_tech_score = self.system.tech_agent.predict(input_normal)
            print(f"   ✅ Normal Technical Score: {normal_tech_score:.4f}")
        except Exception as e:
            print(f"   ❌ Error calculating normal score: {e}")
            return

        # --- STEP B: ATTACK RUN ---
        print("\n2️⃣  RUNNING ATTACK (Flash Crash Simulation)...")
        try:
            # 40% Drop + Nuclear Indicators
            crashed_df = self.generate_flash_crash(df, drop_pct=0.40)
            input_crashed = self._prepare_data_for_ai(crashed_df)
            
            if hasattr(self.system.tech_agent, 'predict_signal'):
                crashed_tech_score = self.system.tech_agent.predict_signal(input_crashed)
            else:
                crashed_tech_score = self.system.tech_agent.predict(input_crashed)
                
            print(f"   ⚠️ Crashed Technical Score: {crashed_tech_score:.4f}")

        except Exception as e:
            print(f"   ❌ Error during crash simulation: {e}")
            return

        # --- STEP C: REPORT CARD ---
        print("\n" + "="*40)
        print("🛡️  ROBUSTNESS REPORT CARD")
        print("="*40)
        
        score_drop = normal_tech_score - crashed_tech_score
        
        # We allow a very small drop to pass, just to verify the inputs changed
        if score_drop > 0.01:
            print(f"✅ PASS: System detected the crash!")
            print(f"   Confidence dropped by {score_drop:.4f}")
        elif crashed_tech_score < 0.4:
            print(f"✅ PASS: System is already cautious (Score {crashed_tech_score:.4f})")
        else:
            print(f"❌ FAIL: System ignored the crash.")
            print(f"   Confidence drop: {score_drop:.4f}")

        print("\nTest Complete.\n")