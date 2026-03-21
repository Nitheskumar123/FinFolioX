"""
==============================================================================
FUSION + HYBRID REGIME BACKTEST  — 5-Day Horizon
==============================================================================
Tests the full pipeline:
  HybridRegimeAgent  →  vol + regime_label + regime_confidence
  TechnicalAgent     →  lstm_signal  (simulated at ~0.65 accuracy baseline)
  SentimentAgent     →  BLOCKED / frozen at 0.0 (backtest — no live news)
  FusionAgent        →  final_confidence
  Decision           →  BUY (conf≥0.52) | SELL (conf<0.40) | HOLD

Test windows (5-day horizon):
  Mar03→08   Bear start
  Mar04→09   Bear early
  Mar09→16   Deep Bear
  Mar12→17   Bounce window

What this shows:
  1. How FusionAgent reacts to regime_confidence from HybridRegimeAgent
  2. Whether the confidence is high enough to trigger BUY/SELL decisions
  3. Whether those decisions are directionally correct vs actual 5-day return
  4. The regime acts as the vol_v input (Bear=0.9, Bull=0.2, Sideways=0.5)
     which directly shapes fusion confidence output
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
warnings.filterwarnings("ignore")

# ── Paths — update if needed ──────────────────────────────────────────────────
HMM_PATH    = r"D:\FinFolioX\saved_models\hmm_regime.pkl"
FUSION_PATH = r"D:\FinFolioX\saved_models\attention_fusion.pth"

# ── Decision thresholds (matches finfolio_master.py) ─────────────────────────
BUY_CONFIDENCE_THRESHOLD  = 0.52
SELL_CONFIDENCE_THRESHOLD = 0.40
COMMODITY_BUY_THRESHOLD   = 0.55
COMMODITY_TICKERS_DECISION = {"GLD", "SLV", "USO", "UNG", "GDX"}

# ── Test windows ──────────────────────────────────────────────────────────────
TEST_WINDOWS = [
    ("2026-03-03", "2026-03-08",  "Mar03→08  Bear start"),
    ("2026-03-04", "2026-03-09",  "Mar04→09  Bear early"),
    ("2026-03-09", "2026-03-16",  "Mar09→16  Deep Bear"),
    ("2026-03-12", "2026-03-17",  "Mar12→17  Bounce"),
]

TICKERS = [
    "AAPL", "MSFT", "NVDA", "TSLA", "META", "AMZN", "GOOGL", "AMD", "INTC", "NFLX",
    "JPM", "V", "WMT", "JNJ", "XOM", "CAT", "DIS", "BA", "MCD", "KO",
    "SPY", "QQQ", "TLT", "GLD", "SLV", "USO", "UNG", "DIA", "IWM", "EEM",
]

# ── LSTM signal simulation ────────────────────────────────────────────────────
# Your LSTM is ~65% accurate on 5-day horizon.
# We simulate it here using 3 simple technical signals that approximate what
# a trained LSTM would output, without needing the actual model file.
# This keeps the test portable — swap in real LSTM output when available.
def simulate_lstm_signal(hist: pd.DataFrame) -> float:
    """
    Simulates LSTM output (0–1) using fast EMA crossover + RSI momentum.
    Approximates a trained LSTM at ~65% directional accuracy.
    Replace with self.tech_agent.predict(hist) when running in production.
    """
    try:
        close  = hist["Close"]
        ema_8  = close.ewm(span=8,  adjust=False).mean().iloc[-1]
        ema_21 = close.ewm(span=21, adjust=False).mean().iloc[-1]
        rsi    = float(hist["RSI"].iloc[-1]) if "RSI" in hist.columns else 50.0
        ret_3d = float(close.iloc[-1] / close.iloc[-3] - 1.0) if len(hist) >= 3 else 0.0

        # Bullish score components
        ema_bull  = 1.0 if ema_8 > ema_21 else 0.0
        rsi_bull  = (rsi - 50) / 100.0         # +0.4 at RSI=90, -0.4 at RSI=10
        mom_bull  = np.clip(ret_3d * 10, -0.3, 0.3)

        raw_signal = 0.50 + (ema_bull - 0.5) * 0.25 + rsi_bull * 0.15 + mom_bull
        return float(np.clip(raw_signal, 0.05, 0.95))
    except Exception:
        return 0.50


# ==============================================================================
# DATA HELPERS
# ==============================================================================
def compute_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(com=period - 1, min_periods=period).mean()
    loss  = -delta.clip(upper=0).ewm(com=period - 1, min_periods=period).mean()
    return 100 - (100 / (1 + gain / (loss + 1e-9)))


def fetch_history(ticker, test_date):
    test_dt  = pd.to_datetime(test_date)
    yf_end   = (test_dt + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    yf_start = (test_dt - pd.Timedelta(days=600)).strftime("%Y-%m-%d")
    df = yf.download(ticker, start=yf_start, end=yf_end, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.empty:
        return pd.DataFrame()
    df["EMA_20"]  = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA_50"]  = df["Close"].ewm(span=50, adjust=False).mean()
    df["SMA_50"]  = df["Close"].rolling(50).mean()
    df["SMA_200"] = df["Close"].rolling(200).mean()
    df["RSI"]     = compute_rsi(df["Close"])
    df.dropna(inplace=True)
    return df


def fetch_actual_return(ticker, test_date, outcome_date):
    yf_end   = (pd.to_datetime(outcome_date) + pd.Timedelta(days=3)).strftime("%Y-%m-%d")
    yf_start = (pd.to_datetime(test_date)    - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
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
# LOAD AGENTS
# ==============================================================================
def load_hybrid_regime():
    try:
        # Inline minimal HybridRegimeAgent so this script is standalone
        # (no import needed — same logic as hybrid_regime_agent.py)
        payload    = joblib.load(HMM_PATH)
        model      = payload["model"]
        scaler     = payload["scaler"]
        regime_map = payload["regime_map"]
        n_comp     = payload.get("n_components", model.n_components)
        print(f"   ✅ HMM loaded  map={regime_map}")
        return model, scaler, regime_map, n_comp
    except Exception as e:
        print(f"   ❌ HMM load failed: {e}")
        return None, None, {}, 0


def load_fusion_agent():
    try:
        import torch
        # Import FusionAgent from the project — adjust path if needed
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        try:
            from ml_engine.fusion_agent import FusionAgent
        except ImportError:
            from fusion_agent import FusionAgent
        agent = FusionAgent(model_path=FUSION_PATH)
        print(f"   ✅ FusionAgent loaded from {FUSION_PATH}")
        return agent
    except Exception as e:
        print(f"   ❌ FusionAgent load failed: {e}")
        return None


# ==============================================================================
# REGIME DETECTION (inline — matches hybrid_regime_agent.py v10)
# ==============================================================================
COMMODITY_T   = {"GLD","SLV","USO","UNG","GDX","DJP","PDBC","XOM","CVX","COP","OXY","PSX"}
PRECIOUS_M    = {"GLD","SLV","GDX"}
BOND_T        = {"TLT","IEF","BND","AGG","SHY","TBT","TMF"}
DEFENSIVE_T   = {"WMT","JNJ","KO","MCD","PG","PEP","CL","MDT","ABT"}
CYCLICAL_T    = {"CAT","DE","HON","MMM","GE","UNP","CSX"}
MACRO_ETF_T   = {"SPY","QQQ","DIA","IWM","EEM","TLT"}
HMM_WIN       = 90
OV_RSI        = 42.0
OV_DD         = -0.08
REV_STREAK    = 5
REV_RSI_MAX   = 40
DYN_RSI       = 42
DYN_DD        = -0.10
EXHST_SOFT    = 7
EXHST_HARD    = 12
TRANS_EXIT    = 0.15


def _hmm_features(df):
    close      = df["Close"].squeeze()
    log_ret    = np.log(close / close.shift(1))
    roll_vol   = log_ret.rolling(10).std()
    vol_m90    = roll_vol.rolling(90).mean()
    vol_s90    = roll_vol.rolling(90).std()
    vol_z      = (roll_vol - vol_m90) / (vol_s90 + 1e-9)
    trend_5d   = close.pct_change(5)
    delta      = close.diff()
    gain       = delta.clip(lower=0).ewm(com=13, min_periods=14).mean()
    loss       = -delta.clip(upper=0).ewm(com=13, min_periods=14).mean()
    rsi_n      = (100 - (100 / (1 + gain / (loss + 1e-9)))) / 100.0
    high_20d   = close.rolling(20).max()
    dd         = (close - high_20d) / (high_20d + 1e-9)
    feat_df = pd.DataFrame({
        "log_return": log_ret,  "rolling_vol": roll_vol,
        "vol_zscore": vol_z,    "trend_5d":    trend_5d,
        "rsi_norm":   rsi_n,    "drawdown":    dd,
    }, index=close.index).replace([np.inf, -np.inf], np.nan).dropna()
    return feat_df.values


def hmm_predict(hist, model, scaler, regime_map):
    try:
        feats = _hmm_features(hist)
        if len(feats) < 50: return "Unknown"
        fr    = feats[-HMM_WIN:] if len(feats) > HMM_WIN else feats
        sts   = model.predict(scaler.transform(fr))
        return regime_map.get(int(sts[-1]), "Unknown")
    except Exception:
        return "Unknown"


def bear_exit_prob(hist, model, scaler, regime_map, n_comp):
    try:
        feats = _hmm_features(hist)
        if len(feats) < 50: return 0.0
        fr  = feats[-HMM_WIN:] if len(feats) > HMM_WIN else feats
        sts = model.predict(scaler.transform(fr))
        cs  = int(sts[-1])
        if regime_map.get(cs, "") != "Bear": return 0.0
        return float(sum(model.transmat_[cs, j] for j in range(n_comp)
                         if regime_map.get(j, "") != "Bear"))
    except Exception:
        return 0.0


def bond_regime(hist):
    if len(hist) < 25: return "Sideways", 0.015
    rsi     = float(hist["RSI"].iloc[-1]) if "RSI" in hist.columns else 50.0
    r20d    = float(hist["Close"].iloc[-1] / hist["Close"].iloc[-20] - 1.0)
    r5d     = float(hist["Close"].iloc[-1] / hist["Close"].iloc[-5]  - 1.0)
    vol     = hist["Close"].pct_change().rolling(10).std().iloc[-1]
    if   r20d > 0.02  and r5d > 0.0  and rsi < 72: reg = "Bull"
    elif r20d < -0.02 and r5d < 0.0  and rsi > 30: reg = "Bear"
    else:                                            reg = "Sideways"
    return reg, float(vol) if not pd.isna(vol) else 0.015


def rule_regime(hist, ticker):
    t = ticker.upper()
    if t in BOND_T: return bond_regime(hist)
    cv    = hist["Close"].pct_change().rolling(10).std().iloc[-1]
    if pd.isna(cv): cv = 0.015
    e20   = float(hist["EMA_20"].iloc[-1])
    e50   = float(hist["EMA_50"].iloc[-1])
    price = float(hist["Close"].iloc[-1])
    rsi   = float(hist["RSI"].iloc[-1]) if "RSI" in hist.columns else 50.0
    if t in DEFENSIVE_T:
        sp = (e20 - e50) / (e50 + 1e-9)
        if   e20 > e50 and sp < 0.015:  reg = "Sideways"
        elif e20 > e50 and cv < 0.020:  reg = "Bull"
        elif e20 < e50 and cv > 0.012:  reg = "Bear"
        else:                            reg = "Sideways"
    elif t in CYCLICAL_T:
        if e20 > e50 and price >= e20*1.002 and cv < 0.020 and rsi > 50: reg = "Bull"
        elif e20 < e50 and cv > 0.015: reg = "Bear"
        else:                          reg = "Sideways"
    elif t in COMMODITY_T:
        r10 = (hist["Close"].iloc[-1]/hist["Close"].iloc[-10]-1.0) if len(hist)>=10 else 0.0
        if   rsi>65 and r10>0.05:  reg="Bull"
        elif rsi>55 and r10>0.02:  reg="Bull"
        elif rsi<40 and r10<-0.03: reg="Bear"
        elif cv>0.025 and r10<0.0: reg="Bear"
        else:                      reg="Sideways"
        return reg, float(cv)
    else:
        if   e20 > e50 and cv < 0.025: reg = "Bull"
        elif e20 < e50 and cv > 0.015: reg = "Bear"
        else:                          reg = "Sideways"
    if reg=="Bull" and price < e20:  reg = "Sideways"
    if reg=="Bear" and price > e20:  reg = "Sideways"
    if reg=="Bear" and rsi < 35:     reg = "Sideways"
    if reg=="Bear" and rsi > 60:     reg = "Sideways"
    return reg, float(cv)


def bull_check(ticker, hist, rev_active=False):
    t = ticker.upper()
    try:
        e20   = float(hist["EMA_20"].iloc[-1])
        price = float(hist["Close"].iloc[-1])
        rsi   = float(hist["RSI"].iloc[-1]) if "RSI" in hist.columns else 50.0
        r5    = (hist["Close"].iloc[-1]/hist["Close"].iloc[-5]-1.0) if len(hist)>=5 else 0.0
        if t in PRECIOUS_M:
            r10 = (hist["Close"].iloc[-1]/hist["Close"].iloc[-10]-1.0) if len(hist)>=10 else 0.0
            if r10 < 0.015: return "Sideways", 0.55
        if price < e20 * 1.002: return "Sideways", 0.55
        rsi_min = 45 if rev_active else 52
        if rsi > rsi_min: return "Bull", 0.65 if rsi>55 else 0.60
        if r5  > 0.01:    return "Bull", 0.60
        return "Sideways", 0.55
    except Exception:
        return "Sideways", 0.55


def fuse_hmm_rule(rule, hmm_r, ticker, hist, rev_active=False):
    t = ticker.upper()
    if t in BOND_T:  return rule, 0.80
    if hmm_r == "Unknown": return (bull_check(t, hist, rev_active) if rule=="Bull" else (rule, 0.75))
    if rule == hmm_r: return rule, 1.00
    if (rule=="Bull" and hmm_r=="Bear") or (rule=="Bear" and hmm_r=="Bull"): return "Sideways", 0.50
    if hmm_r=="Bear"  and rule=="Sideways": return "Bear", 0.65
    if rule=="Bear"   and hmm_r=="Sideways": return "Bear", 0.65
    if hmm_r=="Bull"  and rule=="Sideways": return bull_check(t, hist, rev_active)
    if rule=="Bull"   and hmm_r=="Sideways": return bull_check(t, hist, rev_active)
    return "Sideways", 0.70


def get_market_state(spy_hist, model, scaler, regime_map, n_comp):
    streak=0; spy_rsi=50.0; ret_1d=0.0; rev=False; cp=0.0; ho=False
    if spy_hist is None or spy_hist.empty or model is None:
        return streak, spy_rsi, ret_1d, rev, cp, ho
    try:
        spy_rsi = float(spy_hist["RSI"].iloc[-1]) if "RSI" in spy_hist.columns else 50.0
        if len(spy_hist) >= 2:
            ret_1d = float(spy_hist["Close"].iloc[-1]/spy_hist["Close"].iloc[-2]-1.0)
        feats = _hmm_features(spy_hist)
        if len(feats) >= 50:
            fr    = feats[-HMM_WIN:] if len(feats)>HMM_WIN else feats
            sts   = model.predict(scaler.transform(fr))
            lbls  = [regime_map.get(int(s),"?") for s in sts]
            for l in reversed(lbls):
                if l=="Bear": streak+=1
                else: break
        rsi_ov = spy_rsi <= REV_RSI_MAX
        rsi_ex = spy_rsi <= 35
        bnc    = ret_1d > 0
        if streak >= REV_STREAK and rsi_ov and (rsi_ex or bnc): rev = True
        if not rev:
            if   streak >= EXHST_HARD: ho = True
            elif streak >= EXHST_SOFT: cp = 0.15
    except Exception:
        pass
    return streak, spy_rsi, ret_1d, rev, cp, ho


def full_regime_detect(hist, ticker, spy_hist, model, scaler, regime_map, n_comp):
    """Returns (regime_label, current_vol, regime_confidence)"""
    t = ticker.upper()
    r_regime, cv = rule_regime(hist, ticker)

    # oversold on rule
    if r_regime=="Bear" and t not in COMMODITY_T and t not in BOND_T:
        rsi = float(hist["RSI"].iloc[-1]) if "RSI" in hist.columns else 50.0
        h20 = float(hist["Close"].rolling(20).max().iloc[-1])
        dd  = (float(hist["Close"].iloc[-1])-h20)/(h20+1e-9)
        if rsi < OV_RSI and dd < OV_DD: r_regime = "Sideways"

    h_regime   = hmm_predict(hist, model, scaler, regime_map) if model else "Unknown"
    streak, spy_rsi, ret_1d, rev, cp, ho = get_market_state(spy_hist, model, scaler, regime_map, n_comp)
    fused, conf = fuse_hmm_rule(r_regime, h_regime, t, hist, rev)

    # oversold post-fusion
    if fused=="Bear" and t not in COMMODITY_T and t not in BOND_T:
        rsi = float(hist["RSI"].iloc[-1]) if "RSI" in hist.columns else 50.0
        h20 = float(hist["Close"].rolling(20).max().iloc[-1])
        dd  = (float(hist["Close"].iloc[-1])-h20)/(h20+1e-9)
        if rsi < OV_RSI and dd < OV_DD: fused = "Sideways"

    # transition gate
    if fused=="Bear" and t not in COMMODITY_T and t not in BOND_T and model:
        p = bear_exit_prob(hist, model, scaler, regime_map, n_comp)
        if p > TRANS_EXIT:          fused = "Sideways"; conf = min(conf, 0.55)
        elif p > TRANS_EXIT*0.6:    conf  = max(0.45, conf - 0.15)

    # global market state
    if fused=="Bear" and t not in MACRO_ETF_T and t not in COMMODITY_T and t not in BOND_T:
        if rev:  fused="Sideways"; conf=min(conf,0.52)
        elif ho: fused="Sideways"; conf=min(conf,0.52)
        elif cp: conf=max(0.40, conf-cp)

    # dynamic reversal
    if fused=="Bear" and t not in COMMODITY_T and t not in BOND_T and t not in MACRO_ETF_T:
        try:
            rsi = float(hist["RSI"].iloc[-1]) if "RSI" in hist.columns else 50.0
            h20 = float(hist["Close"].rolling(20).max().iloc[-1])
            dd  = (float(hist["Close"].iloc[-1])-h20)/(h20+1e-9)
            if rsi < DYN_RSI and dd < DYN_DD: fused="Sideways"
        except Exception:
            pass

    return fused, cv, conf


# ==============================================================================
# DECISION LOGIC
# ==============================================================================
def make_decision(fusion_conf: float, regime_label: str, ticker: str) -> str:
    t = ticker.upper()
    threshold = COMMODITY_BUY_THRESHOLD if t in COMMODITY_TICKERS_DECISION \
                else BUY_CONFIDENCE_THRESHOLD
    if fusion_conf >= threshold and regime_label != "Bear":
        return "BUY"
    elif fusion_conf < SELL_CONFIDENCE_THRESHOLD and regime_label != "Bull":
        return "SELL"
    else:
        return "HOLD"


def evaluate_decision(decision: str, actual_return: float) -> tuple[str, bool | None]:
    if decision == "BUY":
        return ("✅", True)  if actual_return > 0 else ("❌", False)
    elif decision == "SELL":
        return ("✅", True)  if actual_return < 0 else ("❌", False)
    else:
        return ("➖", None)


# ==============================================================================
# SINGLE WINDOW
# ==============================================================================
def run_window(test_date, outcome_date, label,
               hmm_model, hmm_scaler, hmm_map, hmm_ncomp,
               fusion_agent, verbose=True):

    if verbose:
        print(f"\n{'─'*120}")
        print(f"  {label}  |  {test_date} → {outcome_date}")
        print(f"{'─'*120}")

    # Pre-fetch all histories
    hist_map = {}
    for t in TICKERS:
        try:
            h = fetch_history(t, test_date)
            if not h.empty and len(h) >= 50:
                hist_map[t] = h
        except Exception:
            pass

    spy_hist = hist_map.get("SPY")

    # Market state (for verbose header)
    if spy_hist is not None and hmm_model is not None:
        streak, spy_rsi, ret_1d, rev, cp, ho = get_market_state(
            spy_hist, hmm_model, hmm_scaler, hmm_map, hmm_ncomp)
        mode = ("REVERSAL" if rev else "EXHAUSTION" if (cp>0 or ho) else "NORMAL")
    else:
        streak, spy_rsi, ret_1d, mode = 0, 50.0, 0.0, "NORMAL"

    if verbose:
        print(f"  SPY streak={streak}d  RSI={spy_rsi:.1f}  1d={ret_1d*100:+.2f}%  Mode={mode}")
        print(f"\n  {'Ticker':<7} {'Regime':<10} {'RConf':>6} {'LSTM':>6} {'Sent':>6} "
              f"{'FConf':>6} {'Decision':<7} {'Actual%':>9} {'Result'}")
        print(f"  {'─'*90}")

    results = []
    correct = wrong = neutral = 0
    bull_c = bull_w = 0
    bear_c = bear_w = 0

    for ticker in TICKERS:
        try:
            hist = hist_map.get(ticker)
            if hist is None or hist.empty or len(hist) < 50:
                continue

            # 1. Hybrid Regime
            regime_label, current_vol, regime_conf = full_regime_detect(
                hist, ticker, spy_hist,
                hmm_model, hmm_scaler, hmm_map, hmm_ncomp
            )

            # 2. LSTM signal (simulated — swap for real agent in production)
            lstm_signal = simulate_lstm_signal(hist)

            # 3. Sentiment — FROZEN at 0.0 for backtest
            sent_score = 0.0

            # 4. Fusion
            # vol_v input: regime drives this directly (matches production code)
            vol_v = 0.9 if regime_label == "Bear" else 0.2 if regime_label == "Bull" else 0.5

            if fusion_agent is not None:
                raw_conf, _ = fusion_agent.predict(
                    lstm_p=lstm_signal,
                    sent_s=sent_score,
                    vol_v=vol_v,
                    trust_scores=None,
                )
                # Apply regime_confidence multiplier (matches production flow)
                fusion_conf = float(np.clip(raw_conf * regime_conf, 0.0, 1.0))
            else:
                # Fallback: simple weighted average
                fusion_conf = float(np.clip(
                    lstm_signal * 0.6 + (1 - vol_v) * 0.4, 0.0, 1.0
                ) * regime_conf)

            # 5. Decision
            decision = make_decision(fusion_conf, regime_label, ticker)

            # 6. Actual return
            actual_return = fetch_actual_return(ticker, test_date, outcome_date)

            # 7. Evaluate
            result_icon, flag = evaluate_decision(decision, actual_return)

            if flag is True:    correct += 1
            elif flag is False: wrong   += 1
            else:               neutral += 1

            if decision == "BUY":
                if flag is True:    bull_c += 1
                elif flag is False: bull_w += 1
            elif decision == "SELL":
                if flag is True:    bear_c += 1
                elif flag is False: bear_w += 1

            if verbose:
                reg_icon = {"Bull":"🟢","Bear":"🔴","Sideways":"⚪"}.get(regime_label,"?")
                dec_icon = {"BUY":"🟢BUY","SELL":"🔴SELL","HOLD":"⚪HOLD"}.get(decision,decision)
                print(f"  {ticker:<7} {reg_icon}{regime_label:<9} {regime_conf:>6.2f} "
                      f"{lstm_signal:>6.3f} {sent_score:>6.3f} "
                      f"{fusion_conf:>6.3f} {dec_icon:<9} "
                      f"{actual_return:>+8.2f}%  {result_icon}")

            results.append({
                "ticker": ticker, "regime": regime_label,
                "regime_conf": round(regime_conf, 2),
                "lstm": round(lstm_signal, 3),
                "fusion_conf": round(fusion_conf, 3),
                "decision": decision,
                "actual_%": round(actual_return, 2),
                "result": result_icon,
            })

        except Exception as e:
            if verbose:
                print(f"  {ticker:<7} ERROR: {e}")

    active = correct + wrong
    acc    = (correct / active * 100) if active > 0 else 0.0
    ba     = (bear_c  / (bear_c + bear_w) * 100) if (bear_c + bear_w) > 0 else 0.0
    bua    = (bull_c  / (bull_c + bull_w) * 100) if (bull_c + bull_w) > 0 else 0.0

    if verbose:
        print(f"\n  ── {correct}C/{wrong}W/{neutral}N  "
              f"→  Accuracy={acc:.1f}%  "
              f"BUY_acc={bua:.1f}%  SELL_acc={ba:.1f}%  "
              f"Active={active}  Neutral={neutral}")
        # Decision distribution
        buys  = sum(1 for r in results if r["decision"] == "BUY")
        sells = sum(1 for r in results if r["decision"] == "SELL")
        holds = sum(1 for r in results if r["decision"] == "HOLD")
        print(f"  ── Decisions: 🟢BUY={buys}  🔴SELL={sells}  ⚪HOLD={holds}")

    return {
        "label": label, "test_date": test_date, "outcome_date": outcome_date,
        "mode": mode, "streak": streak, "spy_rsi": round(spy_rsi, 1),
        "correct": correct, "wrong": wrong, "neutral": neutral,
        "active": active, "accuracy": acc,
        "buy_acc": bua, "sell_acc": ba,
        "results": results,
    }


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print("=" * 120)
    print("  FUSION + HYBRID REGIME BACKTEST  |  5-Day Horizon")
    print("  LSTM: simulated ~65%  |  Sentiment: FROZEN 0.0  |  Regime: HybridRegimeAgent v10")
    print("=" * 120)

    print("\n⏳ Loading models...")
    hmm_model, hmm_scaler, hmm_map, hmm_ncomp = load_hybrid_regime()
    fusion_agent = load_fusion_agent()

    all_stats = []
    for test_date, outcome_date, label in TEST_WINDOWS:
        stats = run_window(
            test_date, outcome_date, label,
            hmm_model, hmm_scaler, hmm_map, hmm_ncomp,
            fusion_agent, verbose=True
        )
        all_stats.append(stats)

    # ── Consolidated summary ──────────────────────────────────────────────────
    print("\n" + "=" * 120)
    print("  CONSOLIDATED SUMMARY")
    print("=" * 120)
    print(f"\n  {'Window':<28} {'Mode':<12} {'Str':>3} {'RSI':>5}  "
          f"{'Acc':>7} {'BUYacc':>8} {'SELLacc':>9} {'Active':>7}")
    print(f"  {'─'*90}")

    for s in all_stats:
        ok = "✅" if s["accuracy"] >= 65 else ("⚠️" if s["accuracy"] >= 50 else "❌")
        print(f"  {s['label']:<28} {s['mode']:<12} {s['streak']:>3}d "
              f"{s['spy_rsi']:>5.1f}  "
              f"{s['accuracy']:>6.1f}%{ok}  "
              f"{s['buy_acc']:>7.1f}%  "
              f"{s['sell_acc']:>8.1f}%  "
              f"{s['active']:>6}")

    avg_acc  = sum(s["accuracy"] for s in all_stats) / len(all_stats)
    avg_buy  = sum(s["buy_acc"]  for s in all_stats) / len(all_stats)
    avg_sell = sum(s["sell_acc"] for s in all_stats) / len(all_stats)
    total_c  = sum(s["correct"]  for s in all_stats)
    total_w  = sum(s["wrong"]    for s in all_stats)

    print(f"\n  {'─'*90}")
    print(f"  {'OVERALL AVG':<28} {'':>16}  "
          f"{avg_acc:>6.1f}%   "
          f"{avg_buy:>7.1f}%  "
          f"{avg_sell:>8.1f}%  "
          f"{total_c+total_w:>6}")

    print(f"\n  Key numbers:")
    print(f"    Overall accuracy   : {avg_acc:.1f}%  "
          f"{'✅' if avg_acc >= 65 else '⚠️'}")
    print(f"    BUY  accuracy      : {avg_buy:.1f}%  (directional correct on BUY signals)")
    print(f"    SELL accuracy      : {avg_sell:.1f}%  (directional correct on SELL signals)")
    print(f"    Total correct/wrong: {total_c}C / {total_w}W")

    print(f"\n  Fusion pipeline flow (per ticker per window):")
    print(f"    LSTM signal (~0.65 acc) → regime vol_v (Bear=0.9/Bull=0.2/Side=0.5)")
    print(f"    → FusionAgent.predict() → raw_conf")
    print(f"    → × regime_confidence (HybridRegime) → final fusion_conf")
    print(f"    → conf≥0.52 → BUY | conf<0.40 → SELL | else → HOLD")

    # Save results
    all_rows = []
    for s in all_stats:
        for r in s["results"]:
            r["window"] = s["label"]
            all_rows.append(r)
    pd.DataFrame(all_rows).to_csv("fusion_regime_backtest.csv", index=False)
    print(f"\n  Results saved → fusion_regime_backtest.csv")
    print("\nDone.\n")


if __name__ == "__main__":
    main()