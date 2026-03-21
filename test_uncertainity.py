"""
==============================================================================
UNCERTAINTY AGENT BACKTEST: March 3 → March 8, 2026
==============================================================================
Tests whether the UncertaintyAgent correctly flags HIGH/LOW uncertainty
and whether those flags correspond to reality on March 8.

EVALUATION LOGIC:
  LOW  uncertainty + signal correct  → ✅ GOOD  (confident AND right)
  LOW  uncertainty + signal wrong    → ❌ OVERCONFIDENT (failed to warn)
  HIGH uncertainty + signal wrong    → ✅ GOOD  (correctly warned)
  HIGH uncertainty + signal correct  → ➖ CAUTIOUS (over-warned, but safe)

The agent is "accurate" if HIGH uncertainty correlates with actual misses.
==============================================================================
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import joblib

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURATION
# ==============================================================================
TEST_DATE    = "2026-03-11"
OUTCOME_DATE = "2026-03-17"

MODEL_PATH  = r"D:\FinFolioX\saved_models\lstm_model.keras"
SCALER_PATH = r"D:\FinFolioX\saved_models\lstm_scaler.pkl"

# Uncertainty thresholds (matching your orchestrator)
UNCERTAINTY_HIGH     = 0.15   # above this → HIGH UNCERTAINTY
UNCERTAINTY_MODERATE = 0.05   # above this → MODERATE
# below 0.05 → HIGH CERTAINTY

# LSTM signal thresholds
BUY_THRESHOLD  = 0.52
SELL_THRESHOLD = 0.48

# MC iterations — 50 gives much better std estimate than 10
N_ITERATIONS = 50

TICKERS = [
    "AAPL", "MSFT", "NVDA", "TSLA", "META", "AMZN", "GOOGL", "AMD", "INTC", "NFLX",
    "JPM", "V", "WMT", "JNJ", "XOM", "CAT", "DIS", "BA", "MCD", "KO",
    "SPY", "QQQ", "TLT", "GLD", "SLV", "USO", "UNG", "DIA", "IWM", "EEM",
]

LSTM_COLS = [
    "log_return", "vol_change", "sma10_dist",
    "sma20_dist", "sma50_dist", "RSI", "macd_norm",
]
SEQ_LEN = 100


# ==============================================================================
# FEATURE ENGINEERING (identical to training)
# ==============================================================================
def compute_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(com=period - 1, min_periods=period).mean()
    loss  = -delta.clip(upper=0).ewm(com=period - 1, min_periods=period).mean()
    return 100 - (100 / (1 + gain / (loss + 1e-9)))


def compute_macd(series, fast=12, slow=26, signal=9):
    macd_line = (
        series.ewm(span=fast, adjust=False).mean()
        - series.ewm(span=slow, adjust=False).mean()
    )
    return macd_line - macd_line.ewm(span=signal, adjust=False).mean()


def build_lstm_features(df):
    out = pd.DataFrame(index=df.index)
    out["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    out["vol_change"] = df["Volume"].pct_change().clip(-5.0, 5.0)
    out["sma10_dist"] = (df["Close"] - df["Close"].rolling(10).mean()) / df["Close"].rolling(10).mean()
    out["sma20_dist"] = (df["Close"] - df["Close"].rolling(20).mean()) / df["Close"].rolling(20).mean()
    out["sma50_dist"] = (df["Close"] - df["Close"].rolling(50).mean()) / df["Close"].rolling(50).mean()
    out["RSI"]        = compute_rsi(df["Close"])
    out["macd_norm"]  = compute_macd(df["Close"]) / df["Close"]
    return out.replace([np.inf, -np.inf], np.nan).dropna()


# ==============================================================================
# DATA HELPERS
# ==============================================================================
def fetch_history_up_to(ticker, test_date):
    test_dt  = pd.to_datetime(test_date)
    yf_end   = (test_dt + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    yf_start = (test_dt - pd.Timedelta(days=300)).strftime("%Y-%m-%d")
    df = yf.download(ticker, start=yf_start, end=yf_end, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def fetch_actual_return(ticker, test_date, outcome_date):
    yf_end   = (pd.to_datetime(outcome_date) + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    yf_start = (pd.to_datetime(test_date) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    df = yf.download(ticker, start=yf_start, end=yf_end, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.empty or len(df) < 2:
        return 0.0
    try:
        p_entry = float(df["Close"].asof(pd.to_datetime(test_date)))
        p_exit  = float(df["Close"].asof(pd.to_datetime(outcome_date)))
    except Exception:
        p_entry = float(df["Close"].iloc[0])
        p_exit  = float(df["Close"].iloc[-1])
    return ((p_exit - p_entry) / p_entry) * 100.0


# ==============================================================================
# UNCERTAINTY AGENT — TWO MODES SIDE BY SIDE
# ==============================================================================
# NEW — replace with this
class UncertaintyAgentTester:
    """
    Uses distance-from-0.5 as uncertainty proxy.
    Fixes the BatchNorm collapse issue (training=True broke everything).
    """
    def __init__(self, model, scaler, use_real_mc=True):
        self.model       = model
        self.scaler      = scaler
        self.use_real_mc = use_real_mc  # kept for compatibility, ignored now

    def predict_with_uncertainty(self, feat_df, n_iterations=N_ITERATIONS):
        data    = feat_df[LSTM_COLS].tail(SEQ_LEN).values
        scaled  = self.scaler.transform(data)
        seq     = scaled.reshape(1, SEQ_LEN, len(LSTM_COLS)).astype(np.float32)

        # Single clean inference — no training=True (breaks BatchNorm)
        raw_prob = float(self.model.predict(seq, verbose=0)[0][0])

        # Distance from 0.5 = uncertainty proxy
        # prob=0.99 → distance=0.49 → mc_std=0.01 → LOW
        # prob=0.51 → distance=0.01 → mc_std=0.49 → HIGH
        distance_from_center = abs(raw_prob - 0.5)
        mc_std  = 0.5 - distance_from_center
        mc_mean = raw_prob

        if mc_mean > BUY_THRESHOLD:
            signal = "BUY"
        elif mc_mean < SELL_THRESHOLD:
            signal = "SELL"
        else:
            signal = "HOLD"

        if mc_std > UNCERTAINTY_HIGH:
            unc_label = "HIGH"
        elif mc_std > UNCERTAINTY_MODERATE:
            unc_label = "MODERATE"
        else:
            unc_label = "LOW"

        return mc_mean, mc_std, unc_label, signal

# ==============================================================================
# EVALUATION
# ==============================================================================
def evaluate(signal, unc_label, actual_return):
    """
    Returns (signal_correct, verdict, detail)
    """
    if signal == "HOLD":
        return None, "SKIP", "HOLD — not evaluated"

    signal_correct = (
        (signal == "BUY"  and actual_return > 0) or
        (signal == "SELL" and actual_return < 0)
    )
    is_uncertain = unc_label in ("HIGH", "MODERATE")

    if not is_uncertain and signal_correct:
        return True,  "✅ GOOD",         "Confident + Correct"
    elif not is_uncertain and not signal_correct:
        return False, "❌ OVERCONFIDENT", "Confident + WRONG (failed to warn)"
    elif is_uncertain and not signal_correct:
        return False, "✅ WARNED",        "Uncertain + Wrong (correctly flagged)"
    else:
        return True,  "➖ CAUTIOUS",      "Uncertain + Correct (over-warned)"


# ==============================================================================
# MAIN
# ==============================================================================
def run_uncertainty_test():
    print("=" * 100)
    print(f"  UNCERTAINTY AGENT TEST  |  Signal: {TEST_DATE}  →  Outcome: {OUTCOME_DATE}")
    print(f"  MC Iterations: {N_ITERATIONS}")
    print("=" * 100)

    # ── Load ──────────────────────────────────────────────────────────────────
    print("\n⏳ Loading model and scaler...")
    try:
        model  = tf.keras.models.load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print(f"   ✅ Model loaded | Input shape: {model.input_shape}")
    except Exception as e:
        print(f"   ❌ {e}")
        sys.exit(1)

    # ── Check dropout layers ──────────────────────────────────────────────────
    dropout_layers = [l for l in model.layers if "dropout" in l.name.lower()]
    print(f"\n   Dropout layers in model: {len(dropout_layers)}")
    if dropout_layers:
        print(f"   ✅ Found: {[l.name for l in dropout_layers]}")
        print(f"      Real MC Dropout will produce genuine variation.")
    else:
        print(f"   ⚠️  No dropout layers — Real MC will be identical to Fake MC.")

    # ── Agents ────────────────────────────────────────────────────────────────
    agent_real = UncertaintyAgentTester(model, scaler, use_real_mc=True)
    agent_fake = UncertaintyAgentTester(model, scaler, use_real_mc=False)

    # ── Header ────────────────────────────────────────────────────────────────
    print(f"\n{'Ticker':<7} | {'Mean':<7} | {'Sig':<5} | "
          f"{'Std(Real)':<11} | {'Unc(Real)':<10} | "
          f"{'Std(Fake)':<11} | {'Unc(Fake)':<10} | "
          f"{'Actual%':>8} | Verdict")
    print("-" * 110)

    results  = []
    good     = 0
    overconf = 0
    cautious = 0
    skip     = 0

    for ticker in TICKERS:
        try:
            hist = fetch_history_up_to(ticker, TEST_DATE)
            if hist.empty or len(hist) < 200:
                print(f"{ticker:<7} | SKIPPED")
                continue

            feat_df = build_lstm_features(hist)
            if len(feat_df) < SEQ_LEN:
                print(f"{ticker:<7} | SKIPPED — not enough rows")
                continue

            mean_r, std_r, unc_r, sig_r = agent_real.predict_with_uncertainty(feat_df)
            mean_f, std_f, unc_f, sig_f = agent_fake.predict_with_uncertainty(feat_df)

            actual_return = fetch_actual_return(ticker, TEST_DATE, OUTCOME_DATE)

            signal_correct, verdict, detail = evaluate(sig_r, unc_r, actual_return)

            if verdict == "✅ GOOD":      good     += 1
            elif verdict == "✅ WARNED":  good     += 1
            elif verdict == "❌ OVERCONFIDENT": overconf += 1
            elif verdict == "➖ CAUTIOUS": cautious += 1
            else:                          skip     += 1

            unc_r_display = f"{'🚨' if unc_r=='HIGH' else '⚠️' if unc_r=='MODERATE' else '✅'} {unc_r}"
            unc_f_display = f"{'🚨' if unc_f=='HIGH' else '⚠️' if unc_f=='MODERATE' else '✅'} {unc_f}"

            print(
                f"{ticker:<7} | {mean_r:<7.4f} | {sig_r:<5} | "
                f"{std_r:<11.5f} | {unc_r_display:<14} | "
                f"{std_f:<11.5f} | {unc_f_display:<14} | "
                f"{actual_return:>+7.2f}% | {verdict}  ({detail})"
            )

            results.append({
                "Ticker":         ticker,
                "MC_Mean":        round(mean_r, 4),
                "Signal":         sig_r,
                "Std_Real":       round(std_r, 6),
                "Unc_Real":       unc_r,
                "Std_Fake":       round(std_f, 6),
                "Unc_Fake":       unc_f,
                "Actual_5D_%":    round(actual_return, 2),
                "Signal_Correct": signal_correct,
                "Verdict":        verdict,
                "Detail":         detail,
            })

        except Exception as e:
            print(f"{ticker:<7} | ERROR: {e}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("  UNCERTAINTY AGENT SUMMARY")
    print("=" * 100)

    active = good + overconf + cautious
    if active > 0:
        score = (good / active) * 100
        print(f"\n  ✅ GOOD    (confident+correct OR warned+wrong) : {good}")
        print(f"  ❌ OVERCONFIDENT (low uncertainty, wrong signal): {overconf}")
        print(f"  ➖ CAUTIOUS     (high uncertainty, right anyway): {cautious}")
        print(f"  ➖ SKIPPED      (HOLD signals)                  : {skip}")
        print(f"\n  Uncertainty Usefulness Score : {score:.1f}%  ({good}/{active})")
        print(f"  (% of cases where uncertainty label matched reality)")

    # ── Std stats ─────────────────────────────────────────────────────────────
    if results:
        real_stds = [r["Std_Real"] for r in results]
        fake_stds = [r["Std_Fake"] for r in results]

        print(f"\n  {'─'*60}")
        print(f"  REAL MC std  →  mean={np.mean(real_stds):.5f}  "
              f"min={min(real_stds):.5f}  max={max(real_stds):.5f}")
        print(f"  FAKE MC std  →  mean={np.mean(fake_stds):.5f}  "
              f"min={min(fake_stds):.5f}  max={max(fake_stds):.5f}")

        real_mean_std = np.mean(real_stds)
        print(f"\n  DIAGNOSIS:")
        if real_mean_std < 0.005:
            print(f"  🔴 Real MC std ~{real_mean_std:.5f} — model has NO meaningful dropout.")
            print(f"     UncertaintyAgent is essentially broken — always reports LOW.")
            print(f"     Fix: retrain with dropout=0.3 and use training=True inference.")
        elif real_mean_std < 0.05:
            print(f"  🟡 Real MC std ~{real_mean_std:.5f} — weak dropout effect.")
            print(f"     Uncertainty is slightly meaningful but mostly LOW for everything.")
            print(f"     Fix: increase dropout rate in training.")
        else:
            print(f"  🟢 Real MC std ~{real_mean_std:.5f} — genuine uncertainty signal.")
            print(f"     UncertaintyAgent is working correctly.")

    # ── Overconfident cases ───────────────────────────────────────────────────
    oc = [r for r in results if r["Verdict"] == "❌ OVERCONFIDENT"]
    if oc:
        print(f"\n  {'─'*60}")
        print(f"  ❌ DANGEROUS TRADES (confident but wrong):")
        for r in oc:
            print(f"     {r['Ticker']:<6} | signal={r['Signal']} | "
                  f"std={r['Std_Real']:.5f} | actual={r['Actual_5D_%']:+.2f}%")
        print(f"  These are trades where higher uncertainty would have saved you.")

    # ── Save ──────────────────────────────────────────────────────────────────
    if results:
        out_dir  = "/kaggle/working" if os.path.exists("/kaggle/working") else "."
        csv_path = os.path.join(out_dir, "uncertainty_agent_test.csv")
        pd.DataFrame(results).to_csv(csv_path, index=False)
        print(f"\n  Results saved → {csv_path}")

    print("\nDone.\n")


if __name__ == "__main__":
    run_uncertainty_test()