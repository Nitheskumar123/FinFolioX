import os
import sys
from typing import TypedDict, Dict, Any, Optional, List
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from finfolio_x.settings import GROQ_API_KEY, LLM_MODEL_NAME, LLM_TEMPERATURE
from ml_engine.topology_agent import TopologyAgent

# BUY threshold constant
# FIX v2: Lowered from 0.60 → 0.55 to compensate for the mandatory 0.95
# systemic penalty applied by Phase 13 on every run (risk_score=0.5 → ×0.95).
BUY_CONFIDENCE_THRESHOLD = 0.55
BUY_GDI_MAX = 40.0  # Don't enter if boardroom tension > 40%


# ==============================================================================
# 1. AGENT STATE
# ==============================================================================
class AgentState(TypedDict):
    ticker: str
    hist_data: Any
    stock_obj: Any

    regime_label: str
    current_vol: float
    risk_score: float
    div_status: str

    lstm_signal: float
    mc_mean: float
    mc_std: float
    uncertainty_status: str
    top_driver: str

    sent_score: float

    fusion_confidence: float
    attention_weights: Dict[str, float]
    alloc_pct: float
    recommended_shares: int
    cash_value: float
    final_decision: str

    conflict_detected: bool
    conflict_ruling: str
    conflict_reasoning: str

    trust_scores: Dict[str, float]

    gdi: float
    gdi_tension: str
    gdi_penalty: float

    topology_result: Optional[Dict[str, Any]]
    topology_chaos: float
    topology_modifier: float

    red_team_passed: bool
    red_team_delta: float

    executive_summary: str
    error: Optional[str]


# ==============================================================================
# 2. LANGGRAPH ORCHESTRATOR
# ==============================================================================
class FinFolioGraphOrchestrator:
    """
    Manages the LangGraph execution pipeline for FinFolio-X.
    """

    def __init__(self, master_system):
        self.master = master_system

        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is missing from .env file.")

        self.llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name=LLM_MODEL_NAME,
            temperature=LLM_TEMPERATURE,
        )

        # Phase 24: Initialize TopologyAgent once
        try:
            self.topology_agent = TopologyAgent(time_delay=5, dimension=3, lookback=60)
        except Exception as e:
            print(f"   ⚠️ TopologyAgent initialization failed: {e}")
            self.topology_agent = None

        self.graph = self._build_graph()

    # --------------------------------------------------------------------------
    # NODE 1: Data Ingestion
    # --------------------------------------------------------------------------
    def node_fetch_data(self, state: AgentState) -> AgentState:
        print(f" [Node: Data Ingestion] Fetching data for {state['ticker']}...")
        stock_obj, hist = self.master._fetch_stock_data(state["ticker"])

        # Phase 14: Load per-ticker trust scores
        trust_scores = {"technical": 1.0, "sentiment": 1.0, "regime": 1.0}
        if hasattr(self.master, "meta_agent") and self.master.meta_agent:
            trust_scores = self.master.meta_agent.get_trust_scores(ticker=state["ticker"])
            self.master.meta_agent.print_trust_report(trust_scores)

        if stock_obj is None:
            return {"error": hist, "trust_scores": trust_scores}

        return {
            "stock_obj": stock_obj,
            "hist_data": hist,
            "error": None,
            "trust_scores": trust_scores,
        }

    # --------------------------------------------------------------------------
    # NODE 2: Market Context
    # --------------------------------------------------------------------------
    def node_market_context(self, state: AgentState) -> AgentState:
        print(" [Node: Market Context] Analyzing Volatility and Systemic Risk...")
        regime_label, current_vol = self.master._analyze_regime_module(state["hist_data"])
        risk_score, div_status = self.master._analyze_correlation_module(state["ticker"])
        return {
            "regime_label": regime_label,
            "current_vol": current_vol,
            "risk_score": risk_score,
            "div_status": div_status,
        }

    # --------------------------------------------------------------------------
    # NODE 3: Technical Analysis
    # --------------------------------------------------------------------------
    def node_technical_analysis(self, state: AgentState) -> AgentState:
        print(" [Node: Technical Analysis] Running LSTM, SHAP, and Monte Carlo Dropout...")
        lstm_signal, mc_mean, mc_std, uncertainty_status, top_driver = (
            self.master._analyze_technicals_and_uncertainty(state["hist_data"])
        )
        return {
            "lstm_signal": lstm_signal,
            "mc_mean": mc_mean,
            "mc_std": mc_std,
            "uncertainty_status": uncertainty_status,
            "top_driver": top_driver,
        }

    # --------------------------------------------------------------------------
    # NODE 4: Sentiment Analysis
    # --------------------------------------------------------------------------
    def node_sentiment_analysis(self, state: AgentState) -> AgentState:
        print(" [Node: Sentiment Analysis] Scraping and evaluating News via FinBERT...")
        sent_score = self.master._analyze_sentiment_module(
            state["ticker"], state["stock_obj"], state["lstm_signal"]
        )
        return {"sent_score": sent_score}

    # --------------------------------------------------------------------------
    # NODE 4.5: Topology Analysis (Phase 24)
    # --------------------------------------------------------------------------
    def node_topology_analysis(self, state: AgentState) -> AgentState:
        print(" [Node: Topology Analysis] Phase 24 — Analyzing Market Geometry via TDA...")
        
        topology_result = {}
        topology_chaos = 0.0
        topology_modifier = 1.0
        
        if self.topology_agent and state.get("hist_data") is not None:
            try:
                topology_result = self.topology_agent.analyze(state["hist_data"])
                topology_chaos = topology_result.get("topology_chaos_score", 0.0)
                topology_modifier = topology_result.get("topology_modifier", 1.0)
                print(f"      - Topology Chaos: {topology_chaos:.4f}")
                print(f"      - Dominant Structure: {topology_result.get('dominant_structure', 'Unknown')}")
                print(f"      - Topology Modifier: {topology_modifier:.4f}")
            except Exception as e:
                print(f"      ⚠️ Topology analysis failed: {e}")
                topology_result = {}
                topology_modifier = 1.0
        else:
            print("      ⚠️ Topology Agent not available or no hist_data. Using defaults.")
            topology_modifier = 1.0
        
        return {
            "topology_result": topology_result,
            "topology_chaos": topology_chaos,
            "topology_modifier": topology_modifier,
        }

    # --------------------------------------------------------------------------
    # NODE 5: Fusion Engine
    # --------------------------------------------------------------------------
    def node_fusion_engine(self, state: AgentState) -> AgentState:
        print(" [Node: Fusion Engine] Fusing signals via Multi-Head Attention (Phase 24 Topology-Enhanced)...")

        vol_input = (
            0.9 if state["regime_label"] == "Bear"
            else 0.2 if state["regime_label"] == "Bull"
            else 0.5
        )
        trust_scores = state.get("trust_scores", None)
        
        # Phase 24: Apply topology modifier to input signals
        topology_modifier = state.get("topology_modifier", 1.0)
        topology_chaos = state.get("topology_chaos", 0.0)
        
        final_conf, weights = self.master.fusion_agent.predict(
            lstm_p=state["mc_mean"] * topology_modifier,
            sent_s=state["sent_score"] * topology_modifier,
            vol_v=vol_input * topology_modifier,
            trust_scores=trust_scores,
        )
        
        # Log topology contribution
        print(f"      - Topology modifier applied: {topology_modifier:.3f}x (chaos: {topology_chaos:.3f})")

        return {
            "fusion_confidence": final_conf,
            "attention_weights": weights,
            "alloc_pct": 0.0,
            "recommended_shares": 0,
            "cash_value": 0.0,
            "final_decision": "PENDING",
            "conflict_detected": False,
            "conflict_ruling": "PENDING",
            "conflict_reasoning": "",
            "red_team_passed": True,
            "red_team_delta": 0.0,
        }

    # --------------------------------------------------------------------------
    # NODE 5.5: Conflict Resolution (Phase 13)
    # --------------------------------------------------------------------------
    def node_conflict_resolution(self, state: AgentState) -> AgentState:
        print(" [Node: Conflict Resolution] Phase 13 — Arbitrating agent signals...")

        fusion_conf = state["fusion_confidence"]

        if self.master.conflict_resolver:
            trust_scores = state.get("trust_scores", None)
            result = self.master.conflict_resolver.arbitrate(
                tech_score=state["lstm_signal"],
                sent_score=state["sent_score"],
                mc_std=state["mc_std"],
                regime_label=state["regime_label"],
                risk_score=state["risk_score"],
                fusion_confidence=fusion_conf,
                trust_scores=trust_scores,
            )
            self.master.conflict_resolver.print_report(result)
            adj_conf = result["adjusted_confidence"]
            conflict_detected = result["arbitrated"]
            conflict_ruling = result["ruling"]
            conflict_reasoning = "; ".join(result["reasoning"])
        else:
            adj_conf = fusion_conf
            if state["risk_score"] > 0.70:
                adj_conf *= 0.5
            if state["mc_std"] > 0.10:
                adj_conf *= 0.8
            conflict_detected = False
            conflict_ruling = "NO_MODULE"
            conflict_reasoning = "Phase 13 not loaded; legacy overrides applied."

        # Phase 16: Disagreement Heatmap
        gdi = 0.0
        gdi_tension = "HARMONY"
        gdi_penalty = 1.0
        if hasattr(self.master, "heatmap_agent") and self.master.heatmap_agent:
            heatmap_result = self.master.heatmap_agent.analyze(
                lstm_score=state["lstm_signal"],
                sent_score=state["sent_score"],
                regime_label=state["regime_label"],
                regime_vol=state.get("current_vol", 0.5),
            )
            self.master.heatmap_agent.print_heatmap(heatmap_result)
            gdi = heatmap_result["gdi"]
            gdi_tension = heatmap_result["tension"]
            gdi_penalty = heatmap_result["penalty"]

        # Kelly sizing with regime-aware b
        last_price = state["hist_data"]["Close"].iloc[-1]
        alloc_pct, _ = self.master.risk_engine.calculate_position_size(
            adj_conf,
            state["current_vol"],
            disagreement_penalty=gdi_penalty,
            regime=state["regime_label"],
        )
        num_shares, cash_value = self.master.risk_engine.get_shares_amount(last_price, alloc_pct)

        # FIX v2: Corrected BUY decision logic with lower threshold + GDI gate
        gdi_pct = gdi * 100  # convert to percentage for threshold comparison
        is_buy = (
            alloc_pct > 0.0
            and adj_conf > BUY_CONFIDENCE_THRESHOLD
            and state["regime_label"] != "Bear"
            and gdi_pct < BUY_GDI_MAX
        )
        decision = "BUY" if is_buy else "SELL / HOLD"

        return {
            "fusion_confidence": adj_conf,
            "alloc_pct": alloc_pct,
            "recommended_shares": num_shares,
            "cash_value": cash_value,
            "final_decision": decision,
            "conflict_detected": conflict_detected,
            "conflict_ruling": conflict_ruling,
            "conflict_reasoning": conflict_reasoning,
            "gdi": gdi,
            "gdi_tension": gdi_tension,
            "gdi_penalty": gdi_penalty,
        }

    # --------------------------------------------------------------------------
    # NODE 6: Red Team
    # --------------------------------------------------------------------------
    def node_red_team(self, state: AgentState) -> AgentState:
        print(" [Node: Red Team] High confidence detected. Running Flash Crash Simulation...")
        if self.master.red_team:
            try:
                crashed_df = self.master.red_team.generate_flash_crash(
                    state["hist_data"], drop_pct=0.10  # 10% crash (realistic stress test)
                )
                input_crashed = self.master.red_team._prepare_data_for_ai(crashed_df)
                crashed_score = (
                    self.master.tech_agent.predict_signal(input_crashed)
                    if hasattr(self.master.tech_agent, "predict_signal")
                    else self.master.tech_agent.predict(input_crashed)
                )
                delta = state["lstm_signal"] - crashed_score
                # Veto only if model COMPLETELY ignores the crash (delta < 0.005)
                # Deltas of 0.005+ prove the model detected the crash scenario
                passed = delta >= 0.005

                if not passed:
                    print("    ❌ Red Team Veto! Model failed stress test. Revoking BUY order.")
                    return {
                        "red_team_passed": False,
                        "red_team_delta": delta,
                        "final_decision": "VETOED BY RED TEAM",
                        "alloc_pct": 0.0,
                        "recommended_shares": 0,
                        "cash_value": 0.0,
                    }
                print(f"    ✅ Red Team PASSED. Crash delta: {delta:.4f}")
                return {"red_team_passed": True, "red_team_delta": delta}
            except Exception as e:
                print(f"    ⚠️ Red Team encountered an error: {e}")
        return {}

    # --------------------------------------------------------------------------
    # NODE 7: LLM Supervisor
    # --------------------------------------------------------------------------
    def node_llm_supervisor(self, state: AgentState) -> AgentState:
        print(" [Node: Supervisor] Groq LLM is synthesizing the final executive report...")

        context = f"""
        Ticker: {state['ticker']}
        Regime: {state['regime_label']} (Volatility: {state['current_vol']:.4f})
        Systemic Risk: {state['div_status']}
        Tech Signal: {state['lstm_signal']:.4f} (SHAP Top Driver: {state['top_driver']})
        Uncertainty: {state['uncertainty_status']} (StdDev: {state['mc_std']:.4f})
        Sentiment: {state['sent_score']:.4f}
        Topology Chaos (Phase 24): {state.get('topology_chaos', 0.0):.4f} (Modifier: {state.get('topology_modifier', 1.0):.3f}x)
        Fusion Confidence: {state['fusion_confidence']:.4f}
        Conflict Detected: {state['conflict_detected']}
        Conflict Ruling: {state['conflict_ruling']}
        Trust Scores: Technical={state.get('trust_scores', {}).get('technical', 1.0):.2f}, Sentiment={state.get('trust_scores', {}).get('sentiment', 1.0):.2f}, Regime={state.get('trust_scores', {}).get('regime', 1.0):.2f}
        Disagreement Index (GDI): {state.get('gdi', 0.0) * 100:.1f}% ({state.get('gdi_tension', 'N/A')}), Kelly Penalty: {state.get('gdi_penalty', 1.0):.2f}x
        Final Decision: {state['final_decision']}
        Capital Allocation: {state['alloc_pct'] * 100:.2f}%
        Red Team Passed: {state['red_team_passed']}
        """

        sys_msg = SystemMessage(
            content=(
                "You are the Chief Risk Officer AI for FinFolio-X, an institutional "
                "algorithmic trading framework. Your job is to read the raw outputs from "
                "our mathematical models (LSTM, FinBERT, HMM, Kelly Criterion) and write "
                "a single, highly professional, 4-sentence executive summary explaining "
                "the rationale behind the final decision. Be direct, authoritative, and "
                "mention the mathematical metrics. Do not give financial advice, just "
                "explain the model's reasoning."
            )
        )
        hum_msg = HumanMessage(content=f"Synthesize this state into a summary:\n{context}")

        try:
            response = self.llm.invoke([sys_msg, hum_msg])
            summary = response.content
        except Exception as e:
            summary = f"⚠️ LLM Synthesis failed due to API Error: {e}"

        # Phase 14: Log decision
        if hasattr(self.master, "meta_agent") and self.master.meta_agent:
            try:
                last_price = state["hist_data"]["Close"].iloc[-1]
                self.master.meta_agent.log_decision(
                    ticker=state["ticker"],
                    lstm_score=state["lstm_signal"],
                    sent_score=state["sent_score"],
                    regime_label=state["regime_label"],
                    risk_score=state["risk_score"],
                    fusion_confidence=state["fusion_confidence"],
                    final_decision=state["final_decision"],
                    price_at_decision=float(last_price),
                )
            except Exception as e:
                print(f"    [!] Meta-Agent logging failed in LangGraph: {e}")

        return {"executive_summary": summary}

    # --------------------------------------------------------------------------
    # ROUTING LOGIC
    # --------------------------------------------------------------------------
    def route_after_data(self, state: AgentState) -> str:
        return "end" if state.get("error") else "continue"

    def route_after_arbitration(self, state: AgentState) -> str:
        return "run_red_team" if state["final_decision"] == "BUY" else "skip_to_llm"

    # --------------------------------------------------------------------------
    # GRAPH COMPILATION
    # --------------------------------------------------------------------------
    def _build_graph(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("fetch_data", self.node_fetch_data)
        workflow.add_node("market_context", self.node_market_context)
        workflow.add_node("technical_analysis", self.node_technical_analysis)
        workflow.add_node("sentiment_analysis", self.node_sentiment_analysis)
        workflow.add_node("topology_analysis", self.node_topology_analysis)  # Phase 24
        workflow.add_node("fusion_engine", self.node_fusion_engine)
        workflow.add_node("conflict_resolution", self.node_conflict_resolution)
        workflow.add_node("red_team", self.node_red_team)
        workflow.add_node("llm_supervisor", self.node_llm_supervisor)

        workflow.set_entry_point("fetch_data")

        workflow.add_conditional_edges(
            "fetch_data",
            self.route_after_data,
            {"end": END, "continue": "market_context"},
        )

        workflow.add_edge("market_context", "technical_analysis")
        workflow.add_edge("technical_analysis", "sentiment_analysis")
        workflow.add_edge("sentiment_analysis", "topology_analysis")  # Phase 24: topology after sentiment
        workflow.add_edge("topology_analysis", "fusion_engine")  # Phase 24: topology feeds fusion
        workflow.add_edge("fusion_engine", "conflict_resolution")

        workflow.add_conditional_edges(
            "conflict_resolution",
            self.route_after_arbitration,
            {"run_red_team": "red_team", "skip_to_llm": "llm_supervisor"},
        )

        workflow.add_edge("red_team", "llm_supervisor")
        workflow.add_edge("llm_supervisor", END)

        return workflow.compile()

    # --------------------------------------------------------------------------
    # PUBLIC API
    # --------------------------------------------------------------------------
    def run_analysis(self, ticker: str):
        """Executes the full graph for a given ticker."""
        initial_state = {
            "ticker": ticker,
            "error": None,
            "topology_result": None,
            "topology_chaos": 0.0,
            "topology_modifier": 1.0,
        }
        print(f"\n🚀 [LangGraph Orchestrator] Starting Multi-Agent Graph for {ticker}...")

        final_state = self.graph.invoke(initial_state)

        if final_state.get("error"):
            print(f"❌ Analysis Aborted: {final_state['error']}")
            return final_state

        print("\n" + "█" * 72)
        print("🗣️  CHIEF RISK OFFICER (LLM) SUMMARY")
        print("█" * 72)
        print(f"\n{final_state['executive_summary']}\n")
        print("█" * 72)

        return final_state