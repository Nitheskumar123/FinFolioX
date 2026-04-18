"""
ml_engine/meta_agent.py  HOLD  Self-Correcting Meta-Agent (Phase 14 + Phase 26)
=============================================================================
Maintains a CSV decision ledger, evaluates past predictions T+5 days later,
and dynamically adjusts per-agent trust multipliers via EMA. Phase 26 extends
the ledger with two new columns: asc_score (the Agent Sycophancy Coefficient
at decision time) and asc_reliable (whether the buffer had enough data). The
evaluate_past_decisions() method now produces an ASC accuracy correlation
table showing decision accuracy bucketed by ASC range (low/medium/high) HOLD
the core empirical validation for the IEEE paper hypothesis that high-ASC
decisions have systematically lower outcome accuracy than low-ASC decisions.
"""

import os
import json
import csv
import numpy as np
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging

logger = logging.getLogger("MetaAgent")

try:
    from counterfactual_engine import CounterfactualEngine
except ImportError:
    try:
        from ml_engine.counterfactual_engine import CounterfactualEngine
    except ImportError:
        CounterfactualEngine = None

# ==============================================================================
# CONFIGURATION
# ==============================================================================

TRUST_MIN         = 0.50
TRUST_MAX         = 1.50
TRUST_DEFAULT     = 1.0
EMA_ALPHA         = 0.15
EVAL_LOOKBACK_DAYS = 5
MOVEMENT_THRESHOLD = 0.01

# ASC accuracy bucket thresholds (for paper validation table)
ASC_LOW_THRESHOLD  = 0.30
ASC_HIGH_THRESHOLD = 0.70


class MetaAgent:
    """
    The Self-Correcting Meta-Agent.
    Extends Phase 14 with Phase 26 ASC tracking for empirical validation.
    """

    def __init__(self):
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.meta_dir    = os.path.join(BASE_DIR, "data", "meta")
        self.ledger_path = os.path.join(self.meta_dir, "decision_ledger.csv")
        self.trust_path  = os.path.join(self.meta_dir, "trust_scores.json")

        os.makedirs(self.meta_dir, exist_ok=True)

        if not os.path.exists(self.ledger_path):
            self._create_ledger()

        if not os.path.exists(self.trust_path):
            self._create_default_trust()

        print("   [+] Phase 14+26: Meta-Agent (Self-Correcting + ASC Tracking) Initialized.")

    # ----------------------------------------------------------------------
    # LEDGER
    # ----------------------------------------------------------------------

    def _create_ledger(self):
        """Create CSV with Phase 14 + Phase 26 columns."""
        headers = [
            # Phase 14 original columns
            "timestamp", "ticker", "lstm_score", "sent_score",
            "regime_label", "risk_score", "fusion_confidence",
            "final_decision", "price_at_decision", "evaluated",
            "actual_price_t5", "price_change_pct",
            "lstm_grade", "sent_grade", "regime_grade",
            "hypothetical_buy_pnl", "hypothetical_sell_pnl",
            "optimal_decision", "regret_score", "llm_retrospective",
            # Phase 26 new columns
            "asc_score", "asc_reliable",
        ]
        with open(self.ledger_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
        print("      [+] Created decision ledger with Phase 26 ASC columns.")

    def log_decision(
        self,
        ticker,
        lstm_score,
        sent_score,
        regime_label,
        risk_score,
        fusion_confidence,
        final_decision,
        price_at_decision,
        # Phase 26 new parameters (optional with defaults for backward compat)
        asc_score: float = 0.5,
        asc_reliable: bool = False,
    ):
        """Record a decision to the ledger, including Phase 26 ASC data."""
        row = [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ticker,
            round(lstm_score, 4),
            round(sent_score, 4),
            regime_label,
            round(risk_score, 4),
            round(fusion_confidence, 4),
            final_decision,
            round(price_at_decision, 2),
            "NO",         # evaluated
            "", "", "", "", "",    # t5 columns
            "", "", "", "", "",    # counterfactual columns
            round(asc_score, 4),  # Phase 26
            str(asc_reliable),    # Phase 26
        ]
        with open(self.ledger_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)
        print(f"   [Meta-Agent] Decision logged: {ticker} @ ${price_at_decision:.2f} | ASC={asc_score:.3f}")

    # ----------------------------------------------------------------------
    # HINDSIGHT EVALUATOR
    # ----------------------------------------------------------------------

    def evaluate_past_decisions(self):
        """
        Evaluate un-graded decisions, update trust scores, and run the
        Phase 26 ASC accuracy correlation analysis for paper validation.
        """
        print("\n" + "=" * 60)
        print("[Meta-Agent] HINDSIGHT EVALUATION SESSION (Phase 14+26)")
        print("=" * 60)

        try:
            df = pd.read_csv(self.ledger_path, encoding="utf-8")
            str_cols = [
                "evaluated", "actual_price_t5", "price_change_pct",
                "lstm_grade", "sent_grade", "regime_grade",
                "hypothetical_buy_pnl", "hypothetical_sell_pnl",
                "optimal_decision", "regret_score", "llm_retrospective",
            ]
            for col in str_cols:
                if col in df.columns:
                    df[col] = df[col].astype(object)

            # Ensure Phase 26 columns exist (backward compat with old ledgers)
            if "asc_score" not in df.columns:
                df["asc_score"] = 0.5
            if "asc_reliable" not in df.columns:
                df["asc_reliable"] = False

        except Exception as e:
            print(f"   [!] Cannot read ledger: {e}")
            return

        if df.empty:
            print("   [i] Ledger is empty. Nothing to evaluate.")
            return

        unevaluated = df[df["evaluated"] == "NO"].copy()
        if unevaluated.empty:
            print("   [i] All decisions already evaluated.")
            # Still run ASC accuracy table on already-evaluated decisions
            self._print_asc_accuracy_table(df)
            return

        unevaluated["timestamp_dt"] = pd.to_datetime(unevaluated["timestamp"])
        grades_log = []

        for idx, row in unevaluated.iterrows():
            ticker         = row["ticker"]
            decision_date  = row["timestamp_dt"]
            decision_price = float(row["price_at_decision"])
            lstm_score     = float(row["lstm_score"])
            sent_score     = float(row["sent_score"])
            regime_label   = row["regime_label"]
            final_decision = row["final_decision"]

            print(f"\n   --- Evaluating {ticker} from {decision_date.strftime('%Y-%m-%d')} ---")

            actual_price = self._get_price_t5(ticker, decision_date)
            if actual_price is None:
                print("      [!] Could not fetch T+5 price. Skipping.")
                continue

            price_change_pct = (actual_price - decision_price) / decision_price
            print(f"      Decision Price : ${decision_price:.2f}")
            print(f"      Actual Price   : ${actual_price:.2f}")
            print(f"      Change         : {price_change_pct * 100:.2f}%")

            lstm_grade   = self._grade_agent(lstm_score, price_change_pct, "technical")
            sent_grade   = self._grade_agent(sent_score, price_change_pct, "sentiment")
            regime_grade = self._grade_regime(regime_label, price_change_pct)

            print(f"      Grades -> LSTM: {lstm_grade}, Sent: {sent_grade}, Regime: {regime_grade}")

            df.at[idx, "evaluated"]        = "YES"
            df.at[idx, "actual_price_t5"]  = round(actual_price, 2)
            df.at[idx, "price_change_pct"] = round(price_change_pct, 4)
            df.at[idx, "lstm_grade"]       = lstm_grade
            df.at[idx, "sent_grade"]       = sent_grade
            df.at[idx, "regime_grade"]     = regime_grade

            grades_log.append({
                "ticker": ticker,
                "lstm": lstm_grade,
                "sentiment": sent_grade,
                "regime": regime_grade,
                "regret_penalty": 0.0,
            })

            # Phase 15 counterfactual
            if CounterfactualEngine:
                if not hasattr(self, "_cf_engine"):
                    self._cf_engine = CounterfactualEngine()
                cf_result = self._cf_engine.analyze(
                    actual_decision=final_decision,
                    decision_price=decision_price,
                    actual_price_t5=actual_price,
                    confidence=float(row.get("fusion_confidence", 0.5)),
                )
                retrospective = self._cf_engine.generate_retrospective(
                    ticker=ticker,
                    decision_date=decision_date.strftime("%Y-%m-%d"),
                    cf_result=cf_result,
                    regime_label=regime_label,
                    confidence=float(row.get("fusion_confidence", 0.5)),
                )
                df.at[idx, "hypothetical_buy_pnl"]  = round(cf_result["hypothetical_buy_pnl"], 6)
                df.at[idx, "hypothetical_sell_pnl"] = round(cf_result["hypothetical_sell_pnl"], 6)
                df.at[idx, "optimal_decision"]      = cf_result["optimal_decision"]
                df.at[idx, "regret_score"]          = round(cf_result["regret_score"], 6)
                df.at[idx, "llm_retrospective"]     = retrospective[:500]
                regret_penalty = self._cf_engine.get_regret_penalty(cf_result["regret_score"])
                grades_log[-1]["regret_penalty"] = regret_penalty

        if "timestamp_dt" in df.columns:
            df = df.drop(columns=["timestamp_dt"])
        df.to_csv(self.ledger_path, index=False, encoding="utf-8")
        print(f"\n   [+] Ledger updated ({len(grades_log)} decisions graded).")

        if grades_log:
            self._update_trust_scores(grades_log)

        # Phase 26: Print ASC accuracy correlation table
        self._print_asc_accuracy_table(df)

    def _print_asc_accuracy_table(self, df: pd.DataFrame):
        """
        Phase 26 paper validation: print accuracy bucketed by ASC range.
        Tests hypothesis: low ASC -> higher decision accuracy.
        """
        evaluated = df[df["evaluated"] == "YES"].copy()
        if evaluated.empty or "asc_score" not in evaluated.columns:
            return

        evaluated["asc_score_num"] = pd.to_numeric(evaluated["asc_score"], errors="coerce").fillna(0.5)
        evaluated["correct"] = evaluated["lstm_grade"].apply(
            lambda g: 1 if g == "RIGHT" else (0 if g == "WRONG" else None)
        )
        evaluated_valid = evaluated.dropna(subset=["correct"])

        if evaluated_valid.empty:
            return

        buckets = {
            f"Low ASC (< {ASC_LOW_THRESHOLD})":
                evaluated_valid[evaluated_valid["asc_score_num"] < ASC_LOW_THRESHOLD],
            f"Medium ASC ({ASC_LOW_THRESHOLD}–{ASC_HIGH_THRESHOLD})":
                evaluated_valid[
                    (evaluated_valid["asc_score_num"] >= ASC_LOW_THRESHOLD) &
                    (evaluated_valid["asc_score_num"] < ASC_HIGH_THRESHOLD)
                ],
            f"High ASC (>= {ASC_HIGH_THRESHOLD})":
                evaluated_valid[evaluated_valid["asc_score_num"] >= ASC_HIGH_THRESHOLD],
        }

        print("\n   ╔==========================================================╗")
        print("   ║   PHASE 26 HOLD ASC ACCURACY CORRELATION TABLE              ║")
        print("   ║   Hypothesis: Low ASC -> Higher Decision Accuracy          ║")
        print("   ╠==========================================================╣")
        print(f"   ║  {'ASC Bucket':<35s} {'N':>4s}  {'Accuracy':>8s}          ║")
        print("   ╠==========================================================╣")

        for label, subset in buckets.items():
            n = len(subset)
            if n == 0:
                print(f"   ║  {label:<35s} {'0':>4s}  {'N/A':>8s}          ║")
            else:
                accuracy = subset["correct"].mean() * 100
                bar_len  = int(accuracy / 5)
                bar      = "█" * bar_len + "░" * (20 - bar_len)
                print(f"   ║  {label:<35s} {n:>4d}  {accuracy:>6.1f}%  {bar}  ║")

        print("   ╠==========================================================╣")

        # Overall Pearson correlation between asc_score and correctness
        if len(evaluated_valid) >= 5:
            try:
                corr = float(np.corrcoef(
                    evaluated_valid["asc_score_num"].values,
                    evaluated_valid["correct"].values,
                )[0, 1])
                direction = "inverse (supports hypothesis)" if corr < 0 else "positive (rejects hypothesis)"
                print(f"   ║  Pearson r(ASC, accuracy) = {corr:+.4f}  {direction}  ║")
            except Exception:
                pass

        print("   ╚==========================================================╝\n")

    # ----------------------------------------------------------------------
    # T+5 PRICE FETCH
    # ----------------------------------------------------------------------

    def _get_price_t5(self, ticker, decision_date):
        try:
            start = decision_date + timedelta(days=5)
            end   = decision_date + timedelta(days=12)
            hist  = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                                end=end.strftime("%Y-%m-%d"), progress=False)
            if not hist.empty:
                close_col = hist["Close"]
                if hasattr(close_col, "columns"):
                    close_col = close_col.iloc[:, 0]
                return float(close_col.iloc[0])
            # Proxy with latest price
            latest = yf.download(ticker, period="5d", progress=False)
            if not latest.empty:
                close_col = latest["Close"]
                if hasattr(close_col, "columns"):
                    close_col = close_col.iloc[:, 0]
                return float(close_col.iloc[-1])
            return None
        except Exception as e:
            logger.warning(f"T+5 price fetch failed for {ticker}: {e}")
            return None

    # ----------------------------------------------------------------------
    # GRADING
    # ----------------------------------------------------------------------

    def _grade_agent(self, agent_score, price_change_pct, agent_type):
        if agent_type == "technical":
            predicted_bullish = agent_score > 0.5
        elif agent_type == "sentiment":
            predicted_bullish = agent_score > 0.0
        else:
            predicted_bullish = agent_score > 0.5
        if abs(price_change_pct) < MOVEMENT_THRESHOLD:
            return "NEUTRAL"
        market_went_up = price_change_pct > 0
        return "RIGHT" if (predicted_bullish == market_went_up) else "WRONG"

    def _grade_regime(self, regime_label, price_change_pct):
        if abs(price_change_pct) < MOVEMENT_THRESHOLD:
            return "NEUTRAL"
        if regime_label == "Bull" and price_change_pct > 0:
            return "RIGHT"
        elif regime_label == "Bear" and price_change_pct < 0:
            return "RIGHT"
        elif regime_label == "Sideways" and abs(price_change_pct) < 0.02:
            return "RIGHT"
        return "WRONG"

    # ----------------------------------------------------------------------
    # TRUST SCORE MANAGER
    # ----------------------------------------------------------------------

    def _create_default_trust(self):
        default_scores = {
            "technical": TRUST_DEFAULT,
            "sentiment": TRUST_DEFAULT,
            "regime": TRUST_DEFAULT,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "evaluation_count": 0,
        }
        with open(self.trust_path, "w", encoding="utf-8") as f:
            json.dump(default_scores, f, indent=2)

    def _update_trust_scores(self, grades_log):
        current = self.load_trust_scores()
        agent_rewards = {"technical": [], "sentiment": [], "regime": []}
        ticker_rewards: dict = {}

        for entry in grades_log:
            tk = entry.get("ticker", "GLOBAL")
            regret_penalty = entry.get("regret_penalty", 0.0)
            if tk not in ticker_rewards:
                ticker_rewards[tk] = {"technical": [], "sentiment": [], "regime": []}

            for agent_key in agent_rewards:
                grade_key = "lstm" if agent_key == "technical" else agent_key
                grade     = entry.get(grade_key, "NEUTRAL")
                reward    = (1.0 if grade == "RIGHT" else -1.0 if grade == "WRONG" else 0.0)
                reward   += regret_penalty
                agent_rewards[agent_key].append(reward)
                ticker_rewards[tk][agent_key].append(reward)

        print("\n   [Meta-Agent] Updating Trust Scores (EMA):")
        for agent_key in ["technical", "sentiment", "regime"]:
            old_trust = current.get(agent_key, TRUST_DEFAULT)
            rewards   = agent_rewards[agent_key]
            if not rewards:
                continue
            avg_reward = np.mean(rewards)
            target     = TRUST_DEFAULT + (avg_reward * 0.5)
            new_trust  = old_trust + EMA_ALPHA * (target - old_trust)
            new_trust  = max(TRUST_MIN, min(TRUST_MAX, new_trust))
            direction  = "+" if new_trust > old_trust else "-" if new_trust < old_trust else "="
            print(f"      {agent_key:12s}: {old_trust:.3f} -> {new_trust:.3f} "
                  f"({direction}) [avg_reward={avg_reward:+.2f}]")
            current[agent_key] = round(new_trust, 4)

        for tk, tk_rewards in ticker_rewards.items():
            ticker_key = f"ticker_{tk.upper()}"
            existing   = current.get(ticker_key, {})
            for agent_key in ["technical", "sentiment", "regime"]:
                rewards = tk_rewards[agent_key]
                if not rewards:
                    continue
                avg_reward = np.mean(rewards)
                old_t      = existing.get(agent_key, TRUST_DEFAULT)
                target     = TRUST_DEFAULT + (avg_reward * 0.5)
                new_t      = old_t + EMA_ALPHA * (target - old_t)
                new_t      = max(TRUST_MIN, min(TRUST_MAX, new_t))
                existing[agent_key] = round(new_t, 4)
            current[ticker_key] = existing

        current["last_updated"]     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        current["evaluation_count"] = current.get("evaluation_count", 0) + len(grades_log)

        with open(self.trust_path, "w", encoding="utf-8") as f:
            json.dump(current, f, indent=2)
        print("      [+] Trust scores saved.")

    def load_trust_scores(self):
        try:
            with open(self.trust_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"technical": TRUST_DEFAULT, "sentiment": TRUST_DEFAULT, "regime": TRUST_DEFAULT}

    def get_trust_scores(self, ticker=None):
        scores = self.load_trust_scores()
        global_scores = {
            "technical": scores.get("technical", TRUST_DEFAULT),
            "sentiment": scores.get("sentiment", TRUST_DEFAULT),
            "regime":    scores.get("regime", TRUST_DEFAULT),
        }
        if ticker:
            ticker_key    = f"ticker_{ticker.upper()}"
            ticker_scores = scores.get(ticker_key, {})
            if ticker_scores:
                for agent in list(global_scores.keys()):
                    if agent in ticker_scores:
                        global_scores[agent] = round(
                            0.70 * global_scores[agent] + 0.30 * ticker_scores[agent], 4
                        )
        return global_scores

    @staticmethod
    def print_trust_report(trust_scores):
        print("\n   [Meta-Agent] Current Agent Trust Multipliers:")
        print("   " + "-" * 50)
        for agent, score in trust_scores.items():
            if agent in ("technical", "sentiment", "regime"):
                bar_len = int(score * 20)
                bar     = "#" * bar_len + "." * (30 - bar_len)
                status  = "BOOSTED" if score > 1.05 else "PENALIZED" if score < 0.95 else "NORMAL"
                print(f"      {agent:12s}: {score:.3f}  [{bar}]  {status}")
        print("   " + "-" * 50)