"""
PHASE 15: COUNTERFACTUAL DECISION ENGINE (The "What-If" Simulator)
------------------------------------------------------------------
Adds "Regret and Imagination" to the AI.

Standard ML only learns from the action taken. Phase 15 simulates
parallel realities — what if the AI had forced BUY? SELL? HOLD? —
and quantifies the Opportunity Cost using Counterfactual Regret
Minimization (CFR).

Three Components:
  A. Regret Matrix       – Simulates BUY/SELL/HOLD universes, finds optimal.
  B. Ledger Expansion    – Logs hypothetical P&L and regret scores.
  C. LLM Retrospective   – Generates "Trader's Diary" entries via Groq.
"""

import os
import sys
import numpy as np

# Ensure project root is on the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ==============================================================================
# CONFIGURATION
# ==============================================================================
HIGH_REGRET_THRESHOLD = 0.05   # 5% — regret above this triggers extra penalty
TRADE_COMMISSION = 0.001       # 0.1% simulated trading cost (round-trip)


class CounterfactualEngine:
    """
    The Multiverse Simulator.

    For every evaluated decision, simulates what would have happened
    if the AI had chosen each of the 3 alternatives (BUY, SELL, HOLD).
    Calculates the Regret Score and identifies the Optimal Decision.

    Inputs:
        actual_decision  – What the AI actually did ("BUY", "SELL", "HOLD")
        decision_price   – Stock price at the time of the decision
        actual_price_t5  – Stock price T+5 trading days later
        confidence       – The fusion confidence at decision time

    Outputs (dict):
        hypothetical_buy_pnl   – P&L if the AI had bought (%)
        hypothetical_sell_pnl  – P&L if the AI had sold short (%)
        hypothetical_hold_pnl  – P&L for holding (always 0%)
        optimal_decision       – The best action in hindsight
        optimal_pnl            – P&L of the optimal decision
        actual_pnl             – P&L of the actual decision
        regret_score           – Optimal P&L minus Actual P&L
        regret_level           – "NONE", "LOW", "MODERATE", "HIGH", "EXTREME"
    """

    def __init__(self):
        self.llm = None
        self._init_llm()
        print("   [+] Phase 15: Counterfactual Engine (What-If Simulator) Initialized.")

    def _init_llm(self):
        """Try to load Groq LLM for retrospective generation."""
        try:
            from finfolio_x.settings import GROQ_API_KEY, LLM_MODEL_NAME, LLM_TEMPERATURE
            if GROQ_API_KEY:
                from langchain_groq import ChatGroq
                self.llm = ChatGroq(
                    groq_api_key=GROQ_API_KEY,
                    model_name=LLM_MODEL_NAME,
                    temperature=LLM_TEMPERATURE
                )
                print("      - LLM Retrospective Generator: ONLINE (Groq)")
            else:
                print("      - LLM Retrospective Generator: OFFLINE (No API key)")
        except Exception as e:
            print(f"      - LLM Retrospective Generator: OFFLINE ({e})")

    # ------------------------------------------------------------------
    # A. REGRET MATRIX (The Math Engine)
    # ------------------------------------------------------------------
    def analyze(self, actual_decision, decision_price, actual_price_t5,
                confidence=0.5, tlt_price_start=None, tlt_price_end=None):
        """
        Simulates BUY/SELL/HOLD universes and calculates regret.

        Returns a result dict (see class docstring for schema).
        """
        # Calculate price change
        price_change = actual_price_t5 - decision_price
        price_change_pct = price_change / decision_price

        # --- Universe A: BUY ---
        # If bought: profit = price increase minus commission
        buy_pnl = price_change_pct - TRADE_COMMISSION

        # --- Universe B: SELL (Short) ---
        # If sold short: profit = price decrease minus commission
        sell_pnl = -price_change_pct - TRADE_COMMISSION

        # --- Universe C: HOLD ---
        # If held cash: L3 FIX - dynamic risk-free rate proxy rather than hard 0.0
        # Assuming ~5% annual risk-free rate ≈ 0.02% per 1 trading day (5 days ~ 0.1%)
        hold_pnl = 0.0
        if tlt_price_start is not None and tlt_price_end is not None:
            # TLT represents the bond alternative
            try:
                tlt_change = (tlt_price_end - tlt_price_start) / tlt_price_start
                # Scale it down to mimic a safe cash yield vs long bond
                hold_pnl = tlt_change * 0.20
            except ZeroDivisionError:
                hold_pnl = 0.001
        else:
            hold_pnl = 0.001 # 0.1% for 5 days is a reasonable cash proxy

        # Build the universe map
        universes = {
            "BUY": round(buy_pnl, 6),
            "SELL": round(sell_pnl, 6),
            "HOLD": round(hold_pnl, 6),
        }

        # Find optimal decision
        optimal_decision = max(universes, key=universes.get)
        optimal_pnl = universes[optimal_decision]

        # Map actual decision to its PnL
        actual_clean = self._clean_decision(actual_decision)
        actual_pnl = universes.get(actual_clean, hold_pnl)

        # Regret = what you missed
        regret_score = max(0.0, optimal_pnl - actual_pnl)
        regret_level = self._classify_regret(regret_score)

        return {
            "hypothetical_buy_pnl": universes["BUY"],
            "hypothetical_sell_pnl": universes["SELL"],
            "hypothetical_hold_pnl": universes["HOLD"],
            "optimal_decision": optimal_decision,
            "optimal_pnl": round(optimal_pnl, 6),
            "actual_decision": actual_clean,
            "actual_pnl": round(actual_pnl, 6),
            "regret_score": round(regret_score, 6),
            "regret_level": regret_level,
        }

    def _clean_decision(self, decision_str):
        """Normalize decision strings to BUY/SELL/HOLD."""
        d = str(decision_str).upper().strip()
        if "BUY" in d:
            return "BUY"
        elif "SELL" in d:
            return "SELL"
        else:
            return "HOLD"

    def _classify_regret(self, regret_score):
        """Classify regret into human-readable levels."""
        if regret_score <= 0.005:
            return "NONE"
        elif regret_score <= 0.02:
            return "LOW"
        elif regret_score <= 0.05:
            return "MODERATE"
        elif regret_score <= 0.15:
            return "HIGH"
        else:
            return "EXTREME"

    # ------------------------------------------------------------------
    # B. REGRET-WEIGHTED TRUST ADJUSTMENT
    # ------------------------------------------------------------------
    def get_regret_penalty(self, regret_score):
        """
        Returns an extra trust penalty multiplier based on regret.
        High regret = harsher penalty on agents that caused the miss.

        Returns a value between 0.0 (no extra penalty) and -0.5
        (severe penalty added to the EMA reward).
        """
        if regret_score <= 0.01:
            return 0.0        # No regret, no penalty
        elif regret_score <= 0.05:
            return -0.15      # Moderate: "You missed a small opportunity"
        elif regret_score <= 0.15:
            return -0.30      # High: "You missed a big move"
        else:
            return -0.50      # Extreme: "You missed a massive rally/crash"

    # ------------------------------------------------------------------
    # C. LLM RETROSPECTIVE (The AI Diary)
    # ------------------------------------------------------------------
    def generate_retrospective(self, ticker, decision_date, cf_result,
                                regime_label="Unknown", confidence=0.5):
        """
        Uses Groq LLM to write a 'Trader's Diary' entry reflecting
        on the decision and its counterfactual outcomes.

        Returns a string (the diary entry), or a fallback if LLM fails.
        """
        context = f"""
        Ticker: {ticker}
        Decision Date: {decision_date}
        Market Regime: {regime_label}
        AI Confidence: {confidence:.4f}
        Actual Decision: {cf_result['actual_decision']}
        Actual P&L: {cf_result['actual_pnl']*100:.2f}%
        
        Counterfactual Analysis:
          - If BUY:  {cf_result['hypothetical_buy_pnl']*100:.2f}%
          - If SELL: {cf_result['hypothetical_sell_pnl']*100:.2f}%
          - If HOLD: {cf_result['hypothetical_hold_pnl']*100:.2f}%
        
        Optimal Decision: {cf_result['optimal_decision']} ({cf_result['optimal_pnl']*100:.2f}%)
        Regret Score: {cf_result['regret_score']*100:.2f}%
        Regret Level: {cf_result['regret_level']}
        """

        if not self.llm:
            return self._fallback_retrospective(cf_result, ticker, decision_date)

        try:
            from langchain_core.messages import SystemMessage, HumanMessage

            sys_msg = SystemMessage(content=(
                "You are the Chief Performance Auditor for FinFolio-X, an AI trading system. "
                "You are writing a 3-sentence 'Trader's Diary' entry reflecting on a past "
                "decision. Analyze the counterfactual data to explain: (1) what the AI did, "
                "(2) what it should have done, and (3) what lesson the system should learn. "
                "Be concise, analytical, and reference the specific percentages. "
                "Write in third person about 'the AI' or 'the system'."
            ))
            hum_msg = HumanMessage(content=f"Write the diary entry:\n{context}")

            response = self.llm.invoke([sys_msg, hum_msg])
            return response.content.strip()

        except Exception as e:
            print(f"      [!] LLM Retrospective failed: {e}")
            return self._fallback_retrospective(cf_result, ticker, decision_date)

    def _fallback_retrospective(self, cf_result, ticker, decision_date):
        """Generates a non-LLM retrospective using templates."""
        actual = cf_result['actual_decision']
        optimal = cf_result['optimal_decision']
        regret = cf_result['regret_score'] * 100
        opt_pnl = cf_result['optimal_pnl'] * 100
        act_pnl = cf_result['actual_pnl'] * 100

        if actual == optimal:
            return (
                f"[{decision_date}] {ticker}: The AI correctly chose {actual} "
                f"(P&L: {act_pnl:+.2f}%). No regret. The decision was optimal."
            )
        else:
            return (
                f"[{decision_date}] {ticker}: The AI chose {actual} "
                f"(P&L: {act_pnl:+.2f}%), but the optimal move was {optimal} "
                f"(P&L: {opt_pnl:+.2f}%). Regret: {regret:.2f}%. "
                f"The system should recalibrate agent weights to capture "
                f"similar opportunities in the future."
            )

    # ------------------------------------------------------------------
    # PRETTY PRINT (Regret Audit)
    # ------------------------------------------------------------------
    @staticmethod
    def print_regret_audit(cf_result, retrospective=""):
        """Prints a human-readable Regret Audit to the console."""
        print("\n      --- [Counterfactual Regret Audit] ---")

        # Universe table
        print("      | Universe |  Action  |    P&L    |")
        print("      |----------|----------|-----------|")
        for action in ["BUY", "SELL", "HOLD"]:
            pnl_key = f"hypothetical_{action.lower()}_pnl"
            pnl = cf_result.get(pnl_key, 0.0)
            marker = " <-- ACTUAL" if action == cf_result['actual_decision'] else ""
            marker = " <-- OPTIMAL" if action == cf_result['optimal_decision'] and not marker else marker
            if action == cf_result['actual_decision'] and action == cf_result['optimal_decision']:
                marker = " <-- ACTUAL + OPTIMAL"
            print(f"      |    {action:4s}  |  {action:6s}  | {pnl*100:+7.2f}%  |{marker}")

        print(f"\n      Regret Score : {cf_result['regret_score']*100:.2f}% "
              f"({cf_result['regret_level']})")

        if retrospective:
            print(f"\n      AI Diary: {retrospective}")

        print("      " + "-" * 42)
