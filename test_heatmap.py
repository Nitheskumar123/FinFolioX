"""
==============================================================================
HEATMAP AGENT (Phase 16) BACKTEST: March 3 → March 8, 2026
==============================================================================
Tests whether the HeatmapAgent's GDI (Global Disagreement Index) correctly
flags HIGH boardroom tension BEFORE bad trades happen.

EVALUATION LOGIC:
  LOW  GDI (HARMONY)  + signal correct → ✅ GOOD     (agents agreed AND were right)
  LOW  GDI (HARMONY)  + signal wrong   → ❌ BLIND     (agents agreed but were WRONG)
  HIGH GDI (TENSION)  + signal wrong   → ✅ WARNED    (tension correctly flagged bad trade)
  HIGH GDI (TENSION)  + signal correct → ➖ CAUTIOUS  (tension over-warned, signal worked)

NOTE: Two separate DataFrames are used:
  - hist_raw   (300 cal days) → LSTM feature building (same as lstm backtest)
  - hist_regime (600 cal days) → Regime detection (needs SMA_200 warmup)
  This prevents the SMA_200 dropna from eating into LSTM feature rows.
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
TEST_DATE    = "2026-03-03"
OUTCOME_DATE = "2026-03-08"

MODEL_PATH  = r"D:\FinFolioX\saved_models\lstm_model.keras"
SCALER_PATH = r"D:\FinFolioX\saved_models\lstm_scaler.pkl"

# GDI thresholds (matching heatmap_agent.py)
GDI_LOW_THRESHOLD  = 0.20
GDI_MED_THRESHOLD  = 0.40
GDI_HIGH_THRESHOLD = 0.60

# LSTM signal thresholds
BUY_THRESHOLD  = 0.52
SELL_THRESHOLD = 0.48

# Penalty multipliers (matching heatmap_agent.py)
PENALTY_NONE     = 1.00
PENALTY_MODERATE = 0.75
PENALTY_HIGH     = 0.50
PENALTY_EXTREME  = 0.25

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
# FEATURE ENGINEERING
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
# DATA HELPERS — TWO SEPARATE FETCHES
# ==============================================================================
def fetch_history_up_to(ticker, test_date):
    """
    Returns (hist_raw, hist_regime):
      hist_raw    — 300 calendar days, raw OHLCV, used for LSTM feature building
      hist_regime — 600 calendar days with SMA_50/200/RSI added, used for regime detection

    Keeping them separate prevents SMA_200 dropna from eating LSTM feature rows.
    """
    test_dt = pd.to_datetime(test_date)
    yf_end  = (test_dt + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    # ── Raw fetch for LSTM (300 cal days — same as lstm backtest) ─────────────
    yf_start_short = (test_dt - pd.Timedelta(days=300)).strftime("%Y-%m-%d")
    df_raw = yf.download(ticker, start=yf_start_short, end=yf_end, progress=False)
    if isinstance(df_raw.columns, pd.MultiIndex):
        df_raw.columns = df_raw.columns.get_level_values(0)

    # ── Long fetch for regime (600 cal days — SMA_200 needs 200 rows warmup) ──
    yf_start_long = (test_dt - pd.Timedelta(days=600)).strftime("%Y-%m-%d")
    df_long = yf.download(ticker, start=yf_start_long, end=yf_end, progress=False)
    if isinstance(df_long.columns, pd.MultiIndex):
        df_long.columns = df_long.columns.get_level_values(0)

    if not df_long.empty:
        df_long["SMA_50"]  = df_long["Close"].rolling(50).mean()
        df_long["SMA_200"] = df_long["Close"].rolling(200).mean()
        df_long["RSI"]     = compute_rsi(df_long["Close"])
        df_long.dropna(inplace=True)   # removes first ~200 warmup rows — OK since we have 600 days

    return df_raw, df_long


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
# LSTM SIGNAL (matching TechnicalAgent with logit stretch factor=3.5)
# ==============================================================================
def get_lstm_signal(model, scaler, feat_df):
    data      = feat_df[LSTM_COLS].tail(SEQ_LEN).values
    scaled    = scaler.transform(data)
    seq       = scaled.reshape(1, SEQ_LEN, len(LSTM_COLS)).astype(np.float32)
    raw       = float(model.predict(seq, verbose=0)[0][0])
    p         = np.clip(raw, 1e-5, 1.0 - 1e-5)
    logit     = np.log(p / (1.0 - p))
    stretched = float(1.0 / (1.0 + np.exp(-logit * 3.5)))
    return raw, stretched


# ==============================================================================
# REGIME DETECTION (matching production _analyze_regime_module exactly)
# ==============================================================================
def get_regime(hist, ticker=""):
    current_vol = hist["Close"].pct_change().rolling(10).std().iloc[-1]
    if pd.isna(current_vol):
        current_vol = 0.015

    sma_50  = float(hist["SMA_50"].iloc[-1])
    sma_200 = float(hist["SMA_200"].iloc[-1])
    ret_5d  = (hist["Close"].iloc[-1] / hist["Close"].iloc[-5] - 1.0) if len(hist) >= 5 else 0.0
    rsi_now = float(hist["RSI"].iloc[-1]) if "RSI" in hist.columns else 50.0

    # Base regime
    if sma_50 > sma_200 and current_vol < 0.025:
        regime = "Bull"
    elif sma_50 < sma_200 and current_vol > 0.015:
        regime = "Bear"
    else:
        regime = "Sideways"

    # Bull exhaustion
    if regime == "Bull" and len(hist) >= 6:
        sma_50_prev = float(hist["SMA_50"].iloc[-6])
        sma_slope   = (sma_50 - sma_50_prev) / sma_50_prev
        if ret_5d < -0.015 or rsi_now < 45:
            regime = "Sideways"
        elif ret_5d < 0 and rsi_now < 55 and sma_slope < 0:
            regime = "Sideways"

    # Bear corrections
    if regime == "Bear" and rsi_now < 35:
        regime = "Sideways"
    if regime == "Bear" and rsi_now > 60:
        regime = "Sideways"

    return regime, float(current_vol)


# ==============================================================================
# HEATMAP AGENT (inline — exact copy of production heatmap_agent.py logic)
# ==============================================================================
def run_heatmap(lstm_score, sent_score, regime_label):
    norm_lstm   = max(0.0, min(1.0, lstm_score))
    norm_sent   = max(0.0, min(1.0, (sent_score + 1.0) / 2.0))
    norm_regime = {"bull": 0.65, "bear": 0.35}.get(regime_label.lower(), 0.50)

    spread_lf = abs(norm_lstm - norm_sent)
    spread_lr = abs(norm_lstm - norm_regime)
    spread_fr = abs(norm_sent - norm_regime)

    # H5 FIX: sentiment frozen → use only LSTM vs Regime
    sentiment_frozen = abs(sent_score) < 0.001
    if sentiment_frozen:
        gdi = spread_lr * 1.5
    else:
        gdi = np.mean([spread_lf, spread_lr, spread_fr]) * 1.5

    gdi = max(0.0, min(1.0, gdi))

    if gdi < GDI_LOW_THRESHOLD:
        tension, penalty = "HARMONY",  PENALTY_NONE
    elif gdi < GDI_MED_THRESHOLD:
        tension, penalty = "MODERATE", PENALTY_MODERATE
    elif gdi < GDI_HIGH_THRESHOLD:
        tension, penalty = "HIGH",     PENALTY_HIGH
    else:
        tension, penalty = "EXTREME",  PENALTY_EXTREME

    return {
        "gdi":     round(gdi, 4),
        "tension": tension,
        "penalty": penalty,
        "agents":  {
            "LSTM":    round(norm_lstm, 4),
            "FinBERT": round(norm_sent, 4),
            "Regime":  round(norm_regime, 4),
        },
        "pairs": {
            "LSTM_vs_Regime":   round(spread_lr, 4),
            "LSTM_vs_FinBERT":  round(spread_lf, 4),
            "FinBERT_vs_Regime": round(spread_fr, 4),
        },
    }


# ==============================================================================
# EVALUATION
# ==============================================================================
def evaluate(lstm_signal, tension, actual_return):
    if lstm_signal > BUY_THRESHOLD:
        direction = "BUY"
    elif lstm_signal < SELL_THRESHOLD:
        direction = "SELL"
    else:
        return "HOLD", None, "➖ SKIP", "HOLD signal"

    signal_correct = (
        (direction == "BUY"  and actual_return > 0) or
        (direction == "SELL" and actual_return < 0)
    )
    is_tense = tension in ("MODERATE", "HIGH", "EXTREME")

    if not is_tense and signal_correct:
        return direction, True,  "✅ GOOD",     "Harmony + Correct"
    elif not is_tense and not signal_correct:
        return direction, False, "❌ BLIND",    "Harmony + WRONG (dangerous)"
    elif is_tense and not signal_correct:
        return direction, False, "✅ WARNED",   "Tension + Wrong (GDI saved you)"
    else:
        return direction, True,  "➖ CAUTIOUS", "Tension + Correct (over-warned)"


# ==============================================================================
# MAIN
# ==============================================================================
def run_heatmap_test():
    print("=" * 105)
    print(f"  HEATMAP AGENT TEST  |  Signal: {TEST_DATE}  →  Outcome: {OUTCOME_DATE}")
    print(f"  Sentiment frozen at 0.0 (no live MCP in backtest)")
    print(f"  LSTM uses 300-day raw fetch | Regime uses 600-day enriched fetch")
    print("=" * 105)

    # ── Load LSTM ──────────────────────────────────────────────────────────────
    print("\n⏳ Loading LSTM model and scaler...")
    try:
        model  = tf.keras.models.load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print(f"   ✅ Model loaded | Input shape: {model.input_shape}")
    except Exception as e:
        print(f"   ❌ {e}")
        sys.exit(1)

    # ── Header ─────────────────────────────────────────────────────────────────
    print(f"\n{'Ticker':<7} | {'LSTM':^7} | {'Dir':<5} | {'Regime':<9} | "
          f"{'GDI':^7} | {'Tension':<11} | {'Penalty':^7} | "
          f"{'Actual%':>8} | Verdict")
    print("-" * 105)

    results  = []
    good     = 0
    blind    = 0
    cautious = 0
    skip     = 0

    for ticker in TICKERS:
        try:
            # ── Fetch two separate DataFrames ─────────────────────────────────
            hist_raw, hist_regime = fetch_history_up_to(ticker, TEST_DATE)

            # Validate LSTM data
            if hist_raw.empty or len(hist_raw) < 150:
                print(f"{ticker:<7} | SKIPPED — raw data too short ({len(hist_raw)} rows)")
                continue

            feat_df = build_lstm_features(hist_raw)
            if len(feat_df) < SEQ_LEN:
                print(f"{ticker:<7} | SKIPPED — only {len(feat_df)} feature rows (need {SEQ_LEN})")
                continue

            # Validate regime data
            if hist_regime.empty or len(hist_regime) < 10:
                print(f"{ticker:<7} | SKIPPED — regime data too short ({len(hist_regime)} rows)")
                continue

            # ── Get signals ───────────────────────────────────────────────────
            raw_prob, lstm_stretched  = get_lstm_signal(model, scaler, feat_df)
            regime_label, current_vol = get_regime(hist_regime, ticker)
            sent_score                = 0.0   # frozen in backtest

            # ── Run HeatmapAgent ──────────────────────────────────────────────
            heatmap = run_heatmap(lstm_stretched, sent_score, regime_label)
            gdi     = heatmap["gdi"]
            tension = heatmap["tension"]
            penalty = heatmap["penalty"]

            # ── Actual return ─────────────────────────────────────────────────
            actual_return = fetch_actual_return(ticker, TEST_DATE, OUTCOME_DATE)

            # ── Evaluate ──────────────────────────────────────────────────────
            direction, signal_correct, verdict, detail = evaluate(
                lstm_stretched, tension, actual_return
            )

            if verdict == "✅ GOOD":       good     += 1
            elif verdict == "✅ WARNED":   good     += 1
            elif verdict == "❌ BLIND":    blind    += 1
            elif verdict == "➖ CAUTIOUS": cautious += 1
            else:                           skip     += 1

            t_icon = {
                "HARMONY":  "✅",
                "MODERATE": "⚠️ ",
                "HIGH":     "🚨 ",
                "EXTREME":  "💥",
            }.get(tension, "??")

            print(
                f"{ticker:<7} | {lstm_stretched:<7.4f} | {direction:<5} | {regime_label:<9} | "
                f"{gdi*100:>5.1f}%  | {t_icon}{tension:<9} | {penalty:.2f}x   | "
                f"{actual_return:>+7.2f}% | {verdict}"
            )

            results.append({
                "Ticker":         ticker,
                "LSTM_Stretched": round(lstm_stretched, 4),
                "LSTM_Raw":       round(raw_prob, 4),
                "Direction":      direction,
                "Regime":         regime_label,
                "Vol":            round(current_vol, 4),
                "GDI_%":          round(gdi * 100, 2),
                "Tension":        tension,
                "Penalty":        penalty,
                "Actual_5D_%":    round(actual_return, 2),
                "Signal_Correct": signal_correct,
                "Verdict":        verdict,
                "Detail":         detail,
                "LSTM_vs_Regime": round(heatmap["pairs"]["LSTM_vs_Regime"], 4),
            })

        except Exception as e:
            print(f"{ticker:<7} | ERROR: {e}")

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 105)
    print("  HEATMAP AGENT SUMMARY")
    print("=" * 105)

    active = good + blind + cautious
    if active > 0:
        score = (good / active) * 100
        print(f"\n  ✅ GOOD    (harmony+correct OR tension+wrong) : {good}")
        print(f"  ❌ BLIND   (harmony but signal WRONG)          : {blind}")
        print(f"  ➖ CAUTIOUS (tension but signal worked anyway)  : {cautious}")
        print(f"  ➖ SKIPPED  (HOLD signals)                     : {skip}")
        print(f"\n  GDI Usefulness Score : {score:.1f}%  ({good}/{active})")
        print(f"  (% of cases where GDI tension correctly matched reality)")

    # ── GDI distribution ───────────────────────────────────────────────────────
    if results:
        gdis = [r["GDI_%"] for r in results]
        print(f"\n  {'─'*60}")
        print(f"  GDI Distribution:")
        print(f"    Mean={np.mean(gdis):.1f}%  Min={min(gdis):.1f}%  Max={max(gdis):.1f}%")
        print(f"\n  Tension breakdown:")
        for t in ["HARMONY", "MODERATE", "HIGH", "EXTREME"]:
            count = sum(1 for r in results if r["Tension"] == t)
            icon  = {"HARMONY": "✅", "MODERATE": "⚠️ ", "HIGH": "🚨 ", "EXTREME": "💥"}.get(t, "")
            print(f"    {icon} {t:<10}: {count:2d}  {'#' * count}")

    # ── BLIND cases ────────────────────────────────────────────────────────────
    blind_cases = [r for r in results if r["Verdict"] == "❌ BLIND"]
    if blind_cases:
        print(f"\n  {'─'*60}")
        print(f"  ❌ BLIND CASES — GDI said HARMONY but signal was WRONG:")
        print(f"  These are the most dangerous — LSTM + Regime agreed but market disagreed.")
        for r in blind_cases:
            print(f"     {r['Ticker']:<6} | dir={r['Direction']} | "
                  f"GDI={r['GDI_%']:.1f}% | regime={r['Regime']} | "
                  f"LSTM={r['LSTM_Stretched']:.4f} | actual={r['Actual_5D_%']:+.2f}%")
        print(f"\n  DIAGNOSIS: BLIND cases = LSTM and Regime both pointed same direction")
        print(f"  but market went the other way. No heatmap can fix this — it's a model")
        print(f"  limitation. Only more diverse agents (volume, options flow) would help.")

    # ── WARNED cases ───────────────────────────────────────────────────────────
    warned_cases = [r for r in results if r["Verdict"] == "✅ WARNED"]
    if warned_cases:
        print(f"\n  {'─'*60}")
        print(f"  ✅ WARNED CASES — GDI correctly flagged bad trades:")
        for r in warned_cases:
            print(f"     {r['Ticker']:<6} | dir={r['Direction']} | "
                  f"GDI={r['GDI_%']:.1f}% ({r['Tension']}) | "
                  f"penalty={r['Penalty']:.2f}x | actual={r['Actual_5D_%']:+.2f}%")

    # ── Capital protection estimate ─────────────────────────────────────────────
    print(f"\n  {'─'*60}")
    print(f"  CAPITAL PROTECTION (assuming $10,000 account, 10% position per trade):")
    saved_total = 0.0
    for r in (warned_cases if warned_cases else []):
        base        = 1000.0
        reduced     = base * r["Penalty"]
        loss_full   = base    * abs(r["Actual_5D_%"]) / 100
        loss_reduced = reduced * abs(r["Actual_5D_%"]) / 100
        saved_total += (loss_full - loss_reduced)
    print(f"  Estimated loss avoided by GDI position sizing: ${saved_total:.2f}")

    # ── Save ───────────────────────────────────────────────────────────────────
    if results:
        out_dir  = "/kaggle/working" if os.path.exists("/kaggle/working") else "."
        csv_path = os.path.join(out_dir, "heatmap_agent_test.csv")
        pd.DataFrame(results).to_csv(csv_path, index=False)
        print(f"\n  Results saved → {csv_path}")

    print("\nDone.\n")


if __name__ == "__main__":
    run_heatmap_test()