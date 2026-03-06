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
import numpy as np

import pandas as pd
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.views import View

from ml_engine.topology_agent import TopologyAgent  # Phase 24

# Ensure project root is on the path for ml_engine imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# ==============================================================================
# PHASE 24: NUMPY-SAFE JSON ENCODER & TOPOLOGY HELPERS
# ==============================================================================
class NumpySafeEncoder(json.JSONEncoder):
    """Handle numpy floats / ints that default JSON encoder rejects."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _topology_to_dict(topology_result: dict) -> dict:
    """
    Serialise the topology_result from TopologyAgent.analyze() into a
    JSON-safe dict. Strips the raw point_cloud numpy array (too large),
    returning a compact version suitable for API responses.
    """
    if not topology_result:
        return {}

    return {
        "betti0": topology_result.get("betti0", 0.5),
        "betti1": topology_result.get("betti1", 0.5),
        "persistence_entropy": topology_result.get("persistence_entropy", 0.5),
        "topology_chaos_score": topology_result.get("topology_chaos_score", 0.5),
        "dominant_structure": topology_result.get("dominant_structure", "UNKNOWN"),
        "market_shape_signal": topology_result.get("market_shape_signal", "UNKNOWN"),
        "topology_modifier": topology_result.get("topology_modifier", 1.0),
        "h0_bars": topology_result.get("h0_bars", []),
        "h1_bars": topology_result.get("h1_bars", []),
        "status": topology_result.get("status", "unknown"),
    }


# Lazy-load the heavy AI system (only on first request)
_system_instance = None
_topology_agent = None


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


def _get_topology_agent():
    """
    Singleton loader for TopologyAgent (Phase 24).
    """
    global _topology_agent
    if _topology_agent is None:
        try:
            _topology_agent = TopologyAgent(time_delay=5, dimension=3, lookback=60)
        except Exception as e:
            print(f"   ⚠️ TopologyAgent initialization failed: {e}")
            _topology_agent = None
    return _topology_agent


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

        # Phase 24: Topological Shape Agent
        "topology": _topology_to_dict(state.get("topology_result", {})),

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


# ==============================================================================
# 5. POST /api/simulate/ — Phase 21: Digital Twin Simulation
# ==============================================================================
class SimulateView(APIView):
    """
    Runs the Digital Twin Simulation Engine.
    Accepts: ticker, start_date, end_date, starting_capital,
             decision_interval, scenarios, data_mode, gbm_params
    """

    def post(self, request):
        ticker = request.data.get("ticker", "").strip().upper()
        start_date = request.data.get("start_date", "2024-01-01")
        end_date = request.data.get("end_date", "2024-12-31")
        starting_capital = float(request.data.get("starting_capital", 10000))
        decision_interval = int(request.data.get("decision_interval", 5))
        scenarios = request.data.get("scenarios", [])
        data_mode = request.data.get("data_mode", "historical")
        gbm_params = request.data.get("gbm_params", None)

        if not ticker:
            return Response(
                {"error": "Missing 'ticker' field."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            system = _get_system()

            from ml_engine.simulation_engine import DigitalTwinSimulator
            twin = DigitalTwinSimulator(system=system)

            results = twin.run_simulation(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                starting_capital=starting_capital,
                decision_interval=decision_interval,
                scenarios=scenarios,
                data_mode=data_mode,
                gbm_params=gbm_params,
            )

            # Sanitize NaN/Inf → None recursively before DRF serializes
            def _sanitize_nan(obj):
                if isinstance(obj, float) and (obj != obj or obj == float('inf') or obj == float('-inf')):
                    return None
                if isinstance(obj, dict):
                    return {k: _sanitize_nan(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [_sanitize_nan(v) for v in obj]
                return obj

            return Response(_sanitize_nan(results), status=status.HTTP_200_OK)

        except Exception as e:
            traceback.print_exc()
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


# ==============================================================================
# 6. GET /api/topology/<ticker>/ — Phase 24: Topological Shape Agent Analysis
# ==============================================================================
@method_decorator(csrf_exempt, name="dispatch")
class TopologyView(View):
    """
    GET /api/topology/<ticker>/

    Returns the full Phase 24 Topological Shape Agent analysis for a ticker.
    Includes Betti numbers, persistence diagrams, chaos score, and market shape signal.

    Response:
    {
      "ticker": "AAPL",
      "betti0": 0.34,
      "betti1": 0.68,
      "persistence_entropy": 0.51,
      "topology_chaos_score": 0.57,
      "dominant_structure": "LOOP",
      "market_shape_signal": "SIDEWAYS",
      "topology_modifier": 0.91,
      "h0_bars": [[0.0, 0.12], [0.0, 0.08], ...],
      "h1_bars": [[0.15, 0.43], [0.22, -1.0], ...],
      "point_cloud_3d": [[x, y, z], ...],
      "status": "ok"
    }
    """

    def get(self, request, ticker: str):
        try:
            import yfinance as yf

            hist_df = yf.download(
                ticker.upper(),
                period="6mo",
                interval="1d",
                progress=False,
            )

            if hist_df.empty:
                return JsonResponse(
                    {"error": f"No data found for ticker '{ticker}'"},
                    status=404,
                )

            topology_agent = _get_topology_agent()
            if topology_agent is None:
                return JsonResponse(
                    {"error": "Topology Agent not available"},
                    status=500,
                )

            result = topology_agent.analyze(hist_df)

            # Include compact point cloud for frontend 3-D scatter
            cloud = result.get("point_cloud")
            cloud_3d = []
            if cloud is not None:
                pts = cloud[:200] if len(cloud) > 200 else cloud
                cloud_3d = pts.tolist() if hasattr(pts, 'tolist') else pts

            payload = _topology_to_dict(result)
            payload["ticker"] = ticker.upper()
            payload["point_cloud_3d"] = cloud_3d

            return JsonResponse(payload, encoder=NumpySafeEncoder)

        except Exception as exc:
            return JsonResponse(
                {"error": str(exc), "status": "error"},
                status=500,
            )
