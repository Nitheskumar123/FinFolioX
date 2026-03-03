"""
PHASE 14: SELF-CORRECTING META-AGENT (Continuous Learning)
-----------------------------------------------------------
The "Performance Reviewer" for all AI agents.

This module does NOT look at stock prices — it looks at the agents
themselves. After every analysis, it logs the decision. Later, it
wakes up, checks what actually happened, grades each agent, and
adjusts their Trust Multipliers so the system self-corrects.

Three Components:
  A. Historical Ledger   – CSV log of every decision.
  B. Hindsight Evaluator – Grades agents T+5 trading days later.
  C. Trust Score Manager  – Rolling EMA-based trust multipliers.

Safety Rails (Pro-Tips):
  1. Trust scores clamped to [0.5, 1.5] — no agent is ever muted or god-mode.
  2. Weekend/holiday-safe T+5: uses closest valid trading day.
  3. 1% movement threshold: stock must move >1% to count as Right/Wrong.
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

# Phase 15 Import
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
TRUST_MIN = 0.50           # Floor: agent never fully muted
TRUST_MAX = 1.50           # Ceiling: agent never overpowered
TRUST_DEFAULT = 1.0        # Starting trust for every agent
EMA_ALPHA = 0.15           # Exponential Moving Average smoothing factor
EVAL_LOOKBACK_DAYS = 5     # Compare price T+5 trading days later
MOVEMENT_THRESHOLD = 0.01  # 1% — minimum price move to count as Right/Wrong
EVAL_WINDOW_DAYS = 30      # Only evaluate decisions from the last 30 days


class MetaAgent:
    """
    The Self-Correcting Meta-Agent.

    Maintains a decision ledger, evaluates past predictions against
    actual market outcomes, and dynamically adjusts trust multipliers
    for each agent using an Exponential Moving Average (EMA).

    Trust Scores affect:
      - Phase 5 (Fusion Engine): Scales agent inputs before attention.
      - Phase 13 (Conflict Resolver): Extra tie-breaker based on
        historical accuracy.

    File Paths:
      - Ledger:       data/meta/decision_ledger.csv
      - Trust Scores: data/meta/trust_scores.json
    """

    def __init__(self):
        # Resolve paths relative to project root
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.meta_dir = os.path.join(BASE_DIR, "data", "meta")
        self.ledger_path = os.path.join(self.meta_dir, "decision_ledger.csv")
        self.trust_path = os.path.join(self.meta_dir, "trust_scores.json")

        # Create meta directory if it doesn't exist
        os.makedirs(self.meta_dir, exist_ok=True)

        # Initialize ledger CSV with headers if it doesn't exist
        if not os.path.exists(self.ledger_path):
            self._create_ledger()

        # Initialize trust scores JSON if it doesn't exist
        if not os.path.exists(self.trust_path):
            self._create_default_trust()

        print("   [+] Phase 14: Meta-Agent (Self-Correcting) Initialized.")
        print(f"      - Ledger : {self.ledger_path}")
        print(f"      - Trust  : {self.trust_path}")

    # ------------------------------------------------------------------
    # A. HISTORICAL LEDGER
    # ------------------------------------------------------------------
    def _create_ledger(self):
        """Creates the CSV ledger file with column headers."""
        headers = [
            "timestamp", "ticker", "lstm_score", "sent_score",
            "regime_label", "risk_score", "fusion_confidence",
            "final_decision", "price_at_decision", "evaluated",
            "actual_price_t5", "price_change_pct",
            "lstm_grade", "sent_grade", "regime_grade",
            # Phase 15: Counterfactual columns
            "hypothetical_buy_pnl", "hypothetical_sell_pnl",
            "optimal_decision", "regret_score", "llm_retrospective"
        ]
        with open(self.ledger_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
        print("      [+] Created new decision ledger (with Phase 15 columns).")

    def log_decision(self, ticker, lstm_score, sent_score, regime_label,
                     risk_score, fusion_confidence, final_decision,
                     price_at_decision):
        """
        Records a single analysis decision to the CSV ledger.
        Called at the end of every analyze_stock() run.
        """
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
            "NO",   # evaluated = not yet
            "",     # actual_price_t5
            "",     # price_change_pct
            "",     # lstm_grade
            "",     # sent_grade
            "",     # regime_grade
            "",     # hypothetical_buy_pnl  (Phase 15)
            "",     # hypothetical_sell_pnl (Phase 15)
            "",     # optimal_decision      (Phase 15)
            "",     # regret_score          (Phase 15)
            "",     # llm_retrospective     (Phase 15)
        ]
        with open(self.ledger_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)
        print(f"   [Meta-Agent] Decision logged for {ticker} "
              f"@ ${price_at_decision:.2f}")

    # ------------------------------------------------------------------
    # B. HINDSIGHT EVALUATOR
    # ------------------------------------------------------------------
    def evaluate_past_decisions(self):
        """
        Reads the ledger, finds un-evaluated decisions older than
        EVAL_LOOKBACK_DAYS trading days, fetches actual prices, grades
        each agent, and updates trust scores.

        Safety Rail #2: Uses flexible look-forward to handle weekends/holidays.
        Safety Rail #3: 1% movement threshold for grading.
        """
        print("\n" + "=" * 60)
        print("[Meta-Agent] HINDSIGHT EVALUATION SESSION")
        print("=" * 60)

        # Read the full ledger
        try:
            df = pd.read_csv(self.ledger_path, encoding="utf-8")
            # Force grade/status columns to string dtype (Pandas infers
            # empty columns as float64, which crashes when writing strings)
            str_cols = ["evaluated", "actual_price_t5", "price_change_pct",
                        "lstm_grade", "sent_grade", "regime_grade",
                        "hypothetical_buy_pnl", "hypothetical_sell_pnl",
                        "optimal_decision", "regret_score", "llm_retrospective"]
            for col in str_cols:
                if col in df.columns:
                    df[col] = df[col].astype(object)
        except Exception as e:
            print(f"   [!] Cannot read ledger: {e}")
            return

        if df.empty:
            print("   [i] Ledger is empty. Nothing to evaluate.")
            return

        # Find rows that haven't been evaluated yet
        unevaluated = df[df["evaluated"] == "NO"].copy()
        if unevaluated.empty:
            print("   [i] All decisions already evaluated.")
            return

        # Filter to decisions old enough (at least 7 calendar days
        # to guarantee 5 trading days have passed)
        cutoff = datetime.now() - timedelta(days=7)
        unevaluated["timestamp_dt"] = pd.to_datetime(unevaluated["timestamp"])
        eligible = unevaluated[unevaluated["timestamp_dt"] <= cutoff]

        if eligible.empty:
            print("   [i] No decisions old enough for T+5 evaluation yet.")
            return

        print(f"   [i] Found {len(eligible)} decisions ready for grading.\n")

        grades_log = []  # Collect grades for trust update

        for idx, row in eligible.iterrows():
            ticker = row["ticker"]
            decision_date = row["timestamp_dt"]
            decision_price = float(row["price_at_decision"])
            lstm_score = float(row["lstm_score"])
            sent_score = float(row["sent_score"])
            regime_label = row["regime_label"]
            final_decision = row["final_decision"]

            print(f"   --- Evaluating {ticker} from "
                  f"{decision_date.strftime('%Y-%m-%d')} ---")

            # Fetch actual price T+5 (with weekend/holiday safety)
            actual_price = self._get_price_t5(ticker, decision_date)
            if actual_price is None:
                print(f"      [!] Could not fetch T+5 price. Skipping.")
                continue

            # Calculate price change
            price_change_pct = (actual_price - decision_price) / decision_price

            print(f"      Decision Price : ${decision_price:.2f}")
            print(f"      Actual Price   : ${actual_price:.2f}")
            print(f"      Change         : {price_change_pct*100:.2f}%")

            # Grade each agent (Safety Rail #3: 1% threshold)
            lstm_grade = self._grade_agent(
                lstm_score, price_change_pct, agent_type="technical"
            )
            sent_grade = self._grade_agent(
                sent_score, price_change_pct, agent_type="sentiment"
            )
            regime_grade = self._grade_regime(
                regime_label, price_change_pct
            )

            print(f"      Grades -> LSTM: {lstm_grade}, "
                  f"Sent: {sent_grade}, Regime: {regime_grade}")

            # Update the ledger row
            df.at[idx, "evaluated"] = "YES"
            df.at[idx, "actual_price_t5"] = round(actual_price, 2)
            df.at[idx, "price_change_pct"] = round(price_change_pct, 4)
            df.at[idx, "lstm_grade"] = lstm_grade
            df.at[idx, "sent_grade"] = sent_grade
            df.at[idx, "regime_grade"] = regime_grade

            grades_log.append({
                "lstm": lstm_grade,
                "sentiment": sent_grade,
                "regime": regime_grade,
                "regret_penalty": 0.0  # default, updated below
            })

            # ==============================================================
            # PHASE 15: COUNTERFACTUAL REGRET ANALYSIS
            # ==============================================================
            if CounterfactualEngine:
                if not hasattr(self, '_cf_engine'):
                    self._cf_engine = CounterfactualEngine()

                cf_result = self._cf_engine.analyze(
                    actual_decision=final_decision,
                    decision_price=decision_price,
                    actual_price_t5=actual_price,
                    confidence=float(row.get("fusion_confidence", 0.5))
                )

                # Generate LLM Retrospective (Trader's Diary)
                retrospective = self._cf_engine.generate_retrospective(
                    ticker=ticker,
                    decision_date=decision_date.strftime('%Y-%m-%d'),
                    cf_result=cf_result,
                    regime_label=regime_label,
                    confidence=float(row.get("fusion_confidence", 0.5))
                )

                # Print the Regret Audit
                self._cf_engine.print_regret_audit(cf_result, retrospective)

                # Update the ledger with counterfactual data
                df.at[idx, "hypothetical_buy_pnl"] = round(cf_result["hypothetical_buy_pnl"], 6)
                df.at[idx, "hypothetical_sell_pnl"] = round(cf_result["hypothetical_sell_pnl"], 6)
                df.at[idx, "optimal_decision"] = cf_result["optimal_decision"]
                df.at[idx, "regret_score"] = round(cf_result["regret_score"], 6)
                df.at[idx, "llm_retrospective"] = retrospective[:500]  # cap length

                # Feed regret into trust penalty
                regret_penalty = self._cf_engine.get_regret_penalty(
                    cf_result["regret_score"]
                )
                grades_log[-1]["regret_penalty"] = regret_penalty

        # Save updated ledger back
        # Drop the helper column before saving
        if "timestamp_dt" in df.columns:
            df = df.drop(columns=["timestamp_dt"])
        df.to_csv(self.ledger_path, index=False, encoding="utf-8")
        print(f"\n   [+] Ledger updated ({len(grades_log)} decisions graded).")

        # Update trust scores based on grades
        if grades_log:
            self._update_trust_scores(grades_log)

    def _get_price_t5(self, ticker, decision_date):
        """
        Fetches the closing price ~5 trading days after the decision.

        Safety Rail #2: Uses a flexible window (T+5 to T+10 calendar days)
        to handle weekends and market holidays. Returns the first available
        closing price in that range.
        """
        try:
            # Look from T+5 to T+10 calendar days (covers weekends + holidays)
            start = decision_date + timedelta(days=5)
            end = decision_date + timedelta(days=12)

            hist = yf.download(
                ticker,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                progress=False
            )

            if hist.empty:
                return None

            # Return the first available closing price
            close_col = hist["Close"]
            if hasattr(close_col, "columns"):
                # MultiIndex case from yfinance
                close_col = close_col.iloc[:, 0]
            return float(close_col.iloc[0])

        except Exception as e:
            logger.warning(f"T+5 price fetch failed for {ticker}: {e}")
            return None

    def _grade_agent(self, agent_score, price_change_pct, agent_type):
        """
        Grades a single agent's prediction against actual outcome.

        Safety Rail #3: Price must move >1% to count as a decisive
        Right or Wrong. Movements within +/-1% are graded as NEUTRAL.

        For Technical Agent: score > 0.5 means bullish prediction
        For Sentiment Agent: score > 0.0 means bullish prediction

        Returns: "RIGHT", "WRONG", or "NEUTRAL"
        """
        # Determine if agent predicted bullish or bearish
        if agent_type == "technical":
            predicted_bullish = agent_score > 0.5
        elif agent_type == "sentiment":
            predicted_bullish = agent_score > 0.0
        else:
            predicted_bullish = agent_score > 0.5

        # Check if price moved enough (Safety Rail #3)
        if abs(price_change_pct) < MOVEMENT_THRESHOLD:
            return "NEUTRAL"

        # Did the market actually go up or down?
        market_went_up = price_change_pct > 0

        # Grade
        if predicted_bullish == market_went_up:
            return "RIGHT"
        else:
            return "WRONG"

    def _grade_regime(self, regime_label, price_change_pct):
        """
        Grades the Regime Agent's state detection.

        Bull regime + market went up = RIGHT
        Bear regime + market went down = RIGHT
        Sideways + small move (<2%) = RIGHT
        Otherwise = WRONG
        """
        if abs(price_change_pct) < MOVEMENT_THRESHOLD:
            return "NEUTRAL"

        if regime_label == "Bull" and price_change_pct > 0:
            return "RIGHT"
        elif regime_label == "Bear" and price_change_pct < 0:
            return "RIGHT"
        elif regime_label == "Sideways" and abs(price_change_pct) < 0.02:
            return "RIGHT"
        else:
            return "WRONG"

    # ------------------------------------------------------------------
    # C. TRUST SCORE MANAGER
    # ------------------------------------------------------------------
    def _create_default_trust(self):
        """Creates the initial trust_scores.json with default values."""
        default_scores = {
            "technical": TRUST_DEFAULT,
            "sentiment": TRUST_DEFAULT,
            "regime": TRUST_DEFAULT,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "evaluation_count": 0
        }
        with open(self.trust_path, "w", encoding="utf-8") as f:
            json.dump(default_scores, f, indent=2)
        print("      [+] Created default trust scores (all 1.0).")

    def _update_trust_scores(self, grades_log):
        """
        Updates trust scores using Exponential Moving Average (EMA).

        For each agent:
          - RIGHT  -> reward = +1
          - WRONG  -> reward = -1
          - NEUTRAL -> reward = 0

        New Trust = Old Trust + alpha * (reward - (Old Trust - 1.0))

        Safety Rail #1: Result clamped to [0.5, 1.5].
        """
        current = self.load_trust_scores()

        # Aggregate grades
        agent_rewards = {"technical": [], "sentiment": [], "regime": []}

        for entry in grades_log:
            for agent_key in agent_rewards:
                grade_key = "lstm" if agent_key == "technical" else agent_key
                grade = entry.get(grade_key, "NEUTRAL")
                regret_penalty = entry.get("regret_penalty", 0.0)
                if grade == "RIGHT":
                    agent_rewards[agent_key].append(1.0 + regret_penalty)
                elif grade == "WRONG":
                    agent_rewards[agent_key].append(-1.0 + regret_penalty)
                else:
                    agent_rewards[agent_key].append(0.0 + regret_penalty)

        print("\n   [Meta-Agent] Updating Trust Scores (EMA):")

        for agent_key in ["technical", "sentiment", "regime"]:
            old_trust = current.get(agent_key, TRUST_DEFAULT)
            rewards = agent_rewards[agent_key]

            if not rewards:
                continue

            # Average reward for this batch
            avg_reward = np.mean(rewards)

            # EMA Update: nudge trust towards 1.0 + avg_reward direction
            # If avg_reward = +1 (all right), trust goes up
            # If avg_reward = -1 (all wrong), trust goes down
            target = TRUST_DEFAULT + (avg_reward * 0.5)  # Target: 0.5 to 1.5
            new_trust = old_trust + EMA_ALPHA * (target - old_trust)

            # Safety Rail #1: Clamp to [0.5, 1.5]
            new_trust = max(TRUST_MIN, min(TRUST_MAX, new_trust))

            direction = "+" if new_trust > old_trust else "-" if new_trust < old_trust else "="
            print(f"      {agent_key:12s}: {old_trust:.3f} -> "
                  f"{new_trust:.3f} ({direction}) "
                  f"[avg_reward={avg_reward:+.2f}]")

            current[agent_key] = round(new_trust, 4)

        current["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        current["evaluation_count"] = current.get("evaluation_count", 0) + len(grades_log)

        # Save
        with open(self.trust_path, "w", encoding="utf-8") as f:
            json.dump(current, f, indent=2)

        print("      [+] Trust scores saved.\n")

    def load_trust_scores(self):
        """Reads trust_scores.json from disk."""
        try:
            with open(self.trust_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {
                "technical": TRUST_DEFAULT,
                "sentiment": TRUST_DEFAULT,
                "regime": TRUST_DEFAULT
            }

    def get_trust_scores(self):
        """
        Public API — returns the current trust multipliers.
        Used by Fusion Agent and Conflict Resolver.
        """
        scores = self.load_trust_scores()
        return {
            "technical": scores.get("technical", TRUST_DEFAULT),
            "sentiment": scores.get("sentiment", TRUST_DEFAULT),
            "regime": scores.get("regime", TRUST_DEFAULT),
        }

    # ------------------------------------------------------------------
    # PRETTY PRINT (for console reports)
    # ------------------------------------------------------------------
    @staticmethod
    def print_trust_report(trust_scores):
        """Prints a formatted trust score report."""
        print("\n   [Meta-Agent] Current Agent Trust Multipliers:")
        print("   " + "-" * 50)
        for agent, score in trust_scores.items():
            if agent in ("technical", "sentiment", "regime"):
                bar_len = int(score * 20)
                bar = "#" * bar_len + "." * (30 - bar_len)
                status = "BOOSTED" if score > 1.05 else "PENALIZED" if score < 0.95 else "NORMAL"
                print(f"      {agent:12s}: {score:.3f}  [{bar}]  {status}")
        print("   " + "-" * 50)
