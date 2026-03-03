"""
PHASE 13: CONFLICT RESOLUTION ENGINE (The "Arbitrator")
-------------------------------------------------------
Neuro-Symbolic Arbitration Module.

When agents produce wildly conflicting signals (e.g., Technical says BUY
while Sentiment says SELL), simple averaging produces a dangerous
"mediocre compromise."  This engine detects high disagreement and
resolves it via hard logical rules instead of neural averaging.

Tie-Breakers (evaluated in priority order):
  C. Systemic Veto       – Blocks trade if macro environment is toxic.
  A. Bayesian Certainty  – Favours the agent with lower uncertainty.
  B. Regime Context      – Aligns with the prevailing market regime.
"""

import numpy as np
import logging

# Configure logger for this module
logger = logging.getLogger("ConflictResolver")

# ==============================================================================
# CONFIGURATION THRESHOLDS
# ==============================================================================
CONFLICT_THRESHOLD = 0.60          # |tech - sent| > this → conflict detected
UNCERTAINTY_HIGH   = 0.10          # MC StdDev above this → Technical is guessing
SYSTEMIC_VETO_THRESHOLD = 0.70     # Correlation divergence above this → veto
HOLD_CONFIDENCE    = 0.50          # Neutral confidence forced on HOLD ruling


class ConflictResolver:
    """
    The Neuro-Symbolic Arbitrator.

    Sits between the Fusion Engine and the Risk Engine. When the
    "Disagreement Spread" between the Technical and Sentiment agents
    exceeds a critical threshold, it suspends the attention-fused
    confidence score and applies deterministic tie-breaking rules.

    Inputs (all floats / strings from upstream agents):
        tech_score       – LSTM / Bayesian Mean confidence  (0.0 → 1.0)
        sent_score       – FinBERT aggregated sentiment     (-1.0 → +1.0 mapped to 0→1)
        mc_std           – Monte Carlo Dropout uncertainty   (≥ 0)
        regime_label     – HMM state: "Bull", "Bear", or "Sideways"
        risk_score       – Correlation Divergence score      (0.0 → 1.0)
        fusion_confidence– Raw output of the Attention Fusion Engine (0→1)

    Output:
        dict with keys:
            arbitrated          (bool)  – Whether the engine intervened
            original_confidence (float) – The fusion score before arbitration
            adjusted_confidence (float) – The (possibly overridden) confidence
            ruling              (str)   – "HOLD", "ALIGN_BULL", "ALIGN_BEAR",
                                          "SYSTEMIC_VETO", or "NO_CONFLICT"
            reasoning           (list[str]) – Human-readable audit trail
    """

    def __init__(self,
                 conflict_threshold=CONFLICT_THRESHOLD,
                 uncertainty_high=UNCERTAINTY_HIGH,
                 systemic_veto_threshold=SYSTEMIC_VETO_THRESHOLD):
        self.conflict_threshold = conflict_threshold
        self.uncertainty_high = uncertainty_high
        self.systemic_veto_threshold = systemic_veto_threshold
        print("   ✅ Phase 13: Conflict Resolution Engine (Arbitrator) Initialized.")

    # ------------------------------------------------------------------
    # MAIN ARBITRATION ENTRY POINT
    # ------------------------------------------------------------------
    def arbitrate(self, tech_score, sent_score, mc_std,
                  regime_label, risk_score, fusion_confidence,
                  trust_scores=None):
        """
        Evaluate agent disagreement and, if necessary, override the
        fusion confidence with a rule-based decision.

        Args:
            trust_scores: Optional dict from Phase 14 Meta-Agent.
                          Keys: 'technical', 'sentiment'
                          If provided, used as extra tie-breaker.

        Returns a result dictionary (see class docstring for schema).
        """
        reasoning = []
        adjusted_confidence = fusion_confidence
        ruling = "NO_CONFLICT"
        arbitrated = False

        # Store trust scores for use in tie-breakers
        self._trust = trust_scores or {}

        # ----- Normalise Sentiment to 0-1 scale (FinBERT outputs -1 → +1) -----
        sent_normalised = (sent_score + 1.0) / 2.0  # map [-1,+1] → [0,1]
        sent_normalised = max(0.0, min(1.0, sent_normalised))

        # ----- Step 0: Calculate Disagreement Spread -----
        spread = abs(tech_score - sent_normalised)
        reasoning.append(
            f"Disagreement Spread: {spread:.4f}  "
            f"(Tech={tech_score:.4f}, Sent_norm={sent_normalised:.4f}, "
            f"Threshold={self.conflict_threshold})"
        )

        # =====================================================================
        # TIE-BREAKER C (Highest Priority): SYSTEMIC VETO
        # Checked FIRST because macro-level toxicity overrides everything.
        # =====================================================================
        if risk_score > self.systemic_veto_threshold:
            reasoning.append(
                f"⛔  SYSTEMIC VETO: Correlation Divergence ({risk_score:.4f}) "
                f"> threshold ({self.systemic_veto_threshold}).  "
                f"Macro environment is toxic — trade blocked."
            )
            adjusted_confidence = min(fusion_confidence * 0.30, 0.35)
            ruling = "SYSTEMIC_VETO"
            arbitrated = True

            return self._build_result(
                arbitrated, fusion_confidence, adjusted_confidence,
                ruling, reasoning
            )

        # If no systemic veto, check for agent-level conflict
        if spread < self.conflict_threshold:
            reasoning.append(
                f"✅  Agents are in agreement (Spread {spread:.4f} "
                f"< {self.conflict_threshold}).  No arbitration needed."
            )
            # Even without conflict, apply mild systemic / uncertainty penalties
            adjusted_confidence = self._apply_mild_adjustments(
                fusion_confidence, risk_score, mc_std, reasoning
            )
            return self._build_result(
                arbitrated, fusion_confidence, adjusted_confidence,
                ruling, reasoning
            )

        # =====================================================================
        # CONFLICT DETECTED — Agents strongly disagree
        # =====================================================================
        reasoning.append(
            f"🚨  CONFLICT DETECTED: Spread {spread:.4f} ≥ "
            f"{self.conflict_threshold}.  Activating Arbitration."
        )
        arbitrated = True

        # =====================================================================
        # TIE-BREAKER A: BAYESIAN CERTAINTY CHECK ("Who is more certain?")
        # =====================================================================
        if mc_std > self.uncertainty_high:
            # Technical Agent is guessing → Sentiment wins
            reasoning.append(
                f"🎲  Bayesian Check: MC StdDev ({mc_std:.4f}) > "
                f"{self.uncertainty_high} → Technical Agent is UNCERTAIN.  "
                f"Favouring Sentiment signal."
            )
            # If Sentiment is bearish → confidence drops; if bullish → rises
            if sent_normalised < 0.40:
                adjusted_confidence = min(fusion_confidence, 0.35)
                ruling = "ALIGN_BEAR"
                reasoning.append(
                    "   → Sentiment is bearish + Technical uncertain → "
                    "Confidence capped at 0.35 (HOLD / SELL territory)."
                )
            else:
                adjusted_confidence = max(fusion_confidence, sent_normalised)
                ruling = "ALIGN_BULL"
                reasoning.append(
                    f"   → Sentiment is bullish ({sent_normalised:.4f}) "
                    f"+ Technical uncertain → Confidence raised to "
                    f"{adjusted_confidence:.4f}."
                )
        else:
            # Both agents are certain but disagree
            reasoning.append(
                f"🎲  Bayesian Check: MC StdDev ({mc_std:.4f}) ≤ "
                f"{self.uncertainty_high} → Both agents are confident."
            )

            # =================================================================
            # TIE-BREAKER A.5: TRUST SCORE CHECK (Phase 14 Meta-Agent)
            # If one agent has a significantly better historical track record,
            # side with the more trusted agent before deferring to regime.
            # =================================================================
            tech_trust = self._trust.get("technical", 1.0)
            sent_trust = self._trust.get("sentiment", 1.0)
            trust_gap = abs(tech_trust - sent_trust)

            if trust_gap >= 0.10 and self._trust:
                if tech_trust > sent_trust:
                    reasoning.append(
                        f"📊  Trust Check: Technical ({tech_trust:.2f}) > "
                        f"Sentiment ({sent_trust:.2f}).  "
                        f"Historical accuracy favours Technicals."
                    )
                    adjusted_confidence = max(fusion_confidence, tech_score * 0.9)
                    ruling = "TRUST_TECHNICAL"
                else:
                    reasoning.append(
                        f"📊  Trust Check: Sentiment ({sent_trust:.2f}) > "
                        f"Technical ({tech_trust:.2f}).  "
                        f"Historical accuracy favours Sentiment."
                    )
                    if sent_normalised < 0.40:
                        adjusted_confidence = min(fusion_confidence, 0.40)
                        ruling = "TRUST_SENTIMENT_BEAR"
                    else:
                        adjusted_confidence = max(fusion_confidence, sent_normalised * 0.9)
                        ruling = "TRUST_SENTIMENT_BULL"
            else:
                # No meaningful trust gap → fall back to Regime Context
                if self._trust:
                    reasoning.append(
                        f"📊  Trust Check: Gap ({trust_gap:.2f}) too small "
                        f"→ Deferring to Regime Context."
                    )
                else:
                    reasoning.append(
                        "📊  Trust Check: No trust data available "
                        "→ Deferring to Regime Context."
                    )
                # =============================================================
                # TIE-BREAKER B: REGIME CONTEXT OVERRIDE
                # =============================================================
                adjusted_confidence, ruling = self._regime_tiebreak(
                    tech_score, sent_normalised, regime_label,
                    fusion_confidence, reasoning
                )
        return self._build_result(
            arbitrated, fusion_confidence, adjusted_confidence,
            ruling, reasoning
        )

    # ------------------------------------------------------------------
    # REGIME TIE-BREAKER (internal)
    # ------------------------------------------------------------------
    def _regime_tiebreak(self, tech_score, sent_norm, regime_label,
                         fusion_confidence, reasoning):
        """
        When both agents are confident but disagree, the market regime
        acts as the deciding context.
        """
        if regime_label == "Bear":
            reasoning.append(
                f"⛈️   Regime = BEAR → 'Pessimism wins.'  "
                f"In volatile bear markets, bullish chart patterns are often "
                f"fake breakouts.  Forcing HOLD."
            )
            adjusted = HOLD_CONFIDENCE * 0.8  # 0.40 → very cautious
            return adjusted, "HOLD"

        elif regime_label == "Bull":
            # In a bull market, trust the more bullish agent
            bullish_val = max(tech_score, sent_norm)
            reasoning.append(
                f"☀️   Regime = BULL → 'Optimism wins.'  "
                f"Aligning confidence with the more bullish agent "
                f"({bullish_val:.4f})."
            )
            adjusted = max(fusion_confidence, bullish_val * 0.9)
            return adjusted, "ALIGN_BULL"

        else:  # Sideways
            reasoning.append(
                f"🌥️   Regime = SIDEWAYS → Ambiguous context.  "
                f"Forcing conservative HOLD to avoid false signals."
            )
            adjusted = HOLD_CONFIDENCE
            return adjusted, "HOLD"

    # ------------------------------------------------------------------
    # MILD ADJUSTMENTS (when no major conflict)
    # ------------------------------------------------------------------
    def _apply_mild_adjustments(self, confidence, risk_score, mc_std,
                                reasoning):
        """
        Even when agents agree, apply gentle penalties for elevated
        systemic risk or moderate uncertainty.  This replaces the old
        hard-coded overrides that used to live in the Fusion section.
        """
        adj = confidence

        # Moderate systemic risk penalty (below veto threshold)
        if risk_score > 0.40:
            penalty = 1.0 - (risk_score - 0.40) * 0.5  # linear scale-down
            penalty = max(penalty, 0.60)
            adj *= penalty
            reasoning.append(
                f"⚠️   Mild Systemic Penalty: risk_score={risk_score:.4f} "
                f"→ Confidence scaled by {penalty:.2f} → {adj:.4f}"
            )

        # Moderate uncertainty penalty
        if mc_std > 0.05:
            penalty = 1.0 - (mc_std - 0.05) * 2.0  # linear
            penalty = max(penalty, 0.70)
            adj *= penalty
            reasoning.append(
                f"⚠️   Mild Uncertainty Penalty: mc_std={mc_std:.4f} "
                f"→ Confidence scaled by {penalty:.2f} → {adj:.4f}"
            )

        return adj

    # ------------------------------------------------------------------
    # RESULT BUILDER (internal)
    # ------------------------------------------------------------------
    @staticmethod
    def _build_result(arbitrated, original, adjusted, ruling, reasoning):
        return {
            "arbitrated": arbitrated,
            "original_confidence": round(original, 4),
            "adjusted_confidence": round(adjusted, 4),
            "ruling": ruling,
            "reasoning": reasoning,
        }

    # ------------------------------------------------------------------
    # PRETTY PRINT (for console reports)
    # ------------------------------------------------------------------
    @staticmethod
    def print_report(result):
        """Prints a human-readable arbitration report to the console."""
        print("\n   ⚖️  [Conflict Arbitrator] Phase 13 – Decision Audit")
        print("   " + "-" * 55)

        if not result["arbitrated"]:
            print(f"      ✅  Ruling: {result['ruling']} "
                  f"(Agents agree — no intervention)")
        else:
            print(f"      🚨  Ruling: {result['ruling']}")

        print(f"      • Original Confidence : {result['original_confidence']:.4f}")
        print(f"      • Adjusted Confidence : {result['adjusted_confidence']:.4f}")
        print(f"      • Reasoning Chain:")
        for i, r in enumerate(result["reasoning"], 1):
            print(f"        {i}. {r}")
        print("   " + "-" * 55)
