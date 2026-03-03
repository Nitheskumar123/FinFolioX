"""
PHASE 19: DJANGO REST FRAMEWORK API VIEWS
-------------------------------------------
Exposes the entire FinFolio-X AI engine as a set of REST APIs.

4 Endpoints:
  POST /api/analyze/       — Run full LangGraph inference
  GET  /api/history/       — Fetch decision ledger as JSON
  GET  /api/trust-scores/  — Fetch current trust multipliers
  POST /api/evaluate/      — Trigger T+5 hindsight evaluation
"""

import os
import sys
import json
import traceback

import pandas as pd
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

# Ensure project root is on the path for ml_engine imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Lazy-load the heavy AI system (only on first request)
_system_instance = None


def _get_system():
    """
    Singleton loader for FinFolioSystem.
    Loads once and reuses — avoids reloading LSTM/HMM on every request.
    """
    global _system_instance
    if _system_instance is None:
        from ml_engine.master_system import FinFolioSystem
        _system_instance = FinFolioSystem()
    return _system_instance


# ==============================================================================
# 1. POST /api/analyze/ — Run the full AI pipeline
# ==============================================================================
class AnalyzeView(APIView):
    """
    Accepts {"ticker": "AAPL"} and runs the LangGraph orchestrator.
    Returns the full AgentState as JSON.
    """

    def post(self, request):
        ticker = request.data.get("ticker", "").strip().upper()

        if not ticker:
            return Response(
                {"error": "Missing 'ticker' field. Send {'ticker': 'AAPL'}"},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            system = _get_system()

            # Check if LangGraph orchestrator is available
            try:
                from ml_engine.langgraph_orchestrator import FinFolioGraphOrchestrator
                orchestrator = FinFolioGraphOrchestrator(system)
                final_state = orchestrator.run_analysis(ticker)

                if final_state.get("error"):
                    return Response(
                        {"error": final_state["error"]},
                        status=status.HTTP_500_INTERNAL_SERVER_ERROR
                    )

                # Convert AgentState to a clean JSON-safe dictionary
                result = _state_to_json(final_state, ticker)
                return Response(result, status=status.HTTP_200_OK)

            except ImportError:
                # Fallback: use master_system.analyze_stock() directly
                system.analyze_stock(ticker)
                return Response(
                    {"message": f"Analysis complete for {ticker}. Check terminal logs.",
                     "ticker": ticker},
                    status=status.HTTP_200_OK
                )

        except Exception as e:
            traceback.print_exc()
            return Response(
                {"error": str(e), "traceback": traceback.format_exc()},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


# ==============================================================================
# 2. GET /api/history/ — Read decision ledger
# ==============================================================================
class HistoryView(APIView):
    """Returns the decision_ledger.csv as a JSON array."""

    def get(self, request):
        ledger_path = os.path.join(BASE_DIR, "data", "meta", "decision_ledger.csv")

        if not os.path.exists(ledger_path):
            return Response(
                {"error": "Decision ledger not found. Run an analysis first."},
                status=status.HTTP_404_NOT_FOUND
            )

        try:
            df = pd.read_csv(ledger_path, encoding="utf-8")
            # Convert to JSON string first (handles NaN → null properly),
            # then parse back to Python dict for DRF Response
            records = json.loads(df.to_json(orient="records"))
            return Response({
                "count": len(records),
                "decisions": records
            }, status=status.HTTP_200_OK)
        except Exception as e:
            return Response(
                {"error": f"Failed to read ledger: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


# ==============================================================================
# 3. GET /api/trust-scores/ — Read trust multipliers
# ==============================================================================
class TrustScoresView(APIView):
    """Returns the current trust_scores.json."""

    def get(self, request):
        trust_path = os.path.join(BASE_DIR, "data", "meta", "trust_scores.json")

        if not os.path.exists(trust_path):
            return Response(
                {"error": "Trust scores not found. Run an analysis first."},
                status=status.HTTP_404_NOT_FOUND
            )

        try:
            with open(trust_path, "r", encoding="utf-8") as f:
                scores = json.load(f)

            # Add status labels
            for agent in ["technical", "sentiment", "regime"]:
                val = scores.get(agent, 1.0)
                if val > 1.05:
                    scores[f"{agent}_status"] = "BOOSTED"
                elif val < 0.95:
                    scores[f"{agent}_status"] = "PENALIZED"
                else:
                    scores[f"{agent}_status"] = "NORMAL"

            return Response(scores, status=status.HTTP_200_OK)
        except Exception as e:
            return Response(
                {"error": f"Failed to read trust scores: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


# ==============================================================================
# 4. POST /api/evaluate/ — Trigger hindsight evaluation
# ==============================================================================
class EvaluateView(APIView):
    """Runs the Meta-Agent's T+5 hindsight evaluation."""

    def post(self, request):
        try:
            from ml_engine.meta_agent import MetaAgent
            meta = MetaAgent()

            # Get trust scores BEFORE
            before = meta.get_trust_scores()

            # Run evaluation
            meta.evaluate_past_decisions()

            # Get trust scores AFTER
            after = meta.get_trust_scores()

            # Read the updated ledger for the response
            ledger_path = os.path.join(BASE_DIR, "data", "meta", "decision_ledger.csv")
            evaluated = []
            if os.path.exists(ledger_path):
                df = pd.read_csv(ledger_path, encoding="utf-8")
                evaluated_df = df[df["evaluated"] == "YES"]
                evaluated = json.loads(evaluated_df.to_json(orient="records"))

            return Response({
                "message": "Hindsight evaluation complete.",
                "trust_before": before,
                "trust_after": after,
                "evaluated_decisions": evaluated,
                "total_evaluated": len(evaluated)
            }, status=status.HTTP_200_OK)

        except Exception as e:
            traceback.print_exc()
            return Response(
                {"error": f"Evaluation failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


# ==============================================================================
# HELPER: Convert LangGraph AgentState to JSON-safe dict
# ==============================================================================
def _state_to_json(state, ticker):
    """
    Converts the LangGraph AgentState (which may contain numpy arrays,
    pandas DataFrames, yfinance objects) into a clean JSON-serializable dict.
    """
    result = {
        "ticker": ticker,
        "system_version": "16.0 (Disagreement Heatmap)",

        # Core Signals
        "regime": {
            "label": state.get("regime_label", "Unknown"),
            "volatility": _safe_float(state.get("current_vol", 0)),
        },
        "technical": {
            "lstm_signal": _safe_float(state.get("lstm_signal", 0)),
            "mc_mean": _safe_float(state.get("mc_mean", 0)),
            "mc_std": _safe_float(state.get("mc_std", 0)),
            "uncertainty_status": state.get("uncertainty_status", "Unknown"),
            "top_driver": state.get("top_driver", "Unknown"),
        },
        "sentiment": {
            "score": _safe_float(state.get("sent_score", 0)),
        },
        "systemic_risk": {
            "risk_score": _safe_float(state.get("risk_score", 0)),
            "div_status": state.get("div_status", "Unknown"),
        },

        # Fusion & Decision
        "fusion": {
            "confidence": _safe_float(state.get("fusion_confidence", 0)),
            "attention_weights": _safe_dict(state.get("attention_weights", {})),
        },
        "decision": {
            "action": state.get("final_decision", "UNKNOWN"),
            "allocation_pct": _safe_float(state.get("alloc_pct", 0)) * 100,
            "recommended_shares": state.get("recommended_shares", 0),
            "cash_value": _safe_float(state.get("cash_value", 0)),
        },

        # Phase 13: Conflict Resolution
        "conflict": {
            "detected": state.get("conflict_detected", False),
            "ruling": state.get("conflict_ruling", "N/A"),
            "reasoning": state.get("conflict_reasoning", ""),
        },

        # Phase 14: Trust Scores
        "trust_scores": state.get("trust_scores", {}),

        # Phase 16: Disagreement Heatmap
        "disagreement": {
            "gdi": _safe_float(state.get("gdi", 0)) * 100,
            "tension": state.get("gdi_tension", "N/A"),
            "kelly_penalty": _safe_float(state.get("gdi_penalty", 1.0)),
        },

        # Red Team
        "red_team": {
            "passed": state.get("red_team_passed", True),
            "delta": _safe_float(state.get("red_team_delta", 0)),
        },

        # LLM Summary
        "executive_summary": state.get("executive_summary", ""),
    }

    return result


def _safe_float(val):
    """Safely convert a value to a Python float."""
    try:
        import numpy as np
        if isinstance(val, (np.floating, np.integer)):
            return float(val)
    except ImportError:
        pass
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def _safe_dict(d):
    """Convert dict values to Python floats."""
    if not isinstance(d, dict):
        return {}
    return {k: _safe_float(v) for k, v in d.items()}
