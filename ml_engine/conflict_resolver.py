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

v2.2 PATCH: Regime-Aware Risk Discount
  Bull   + risk_score > 0.60 → discount by 50%
  Sideways + risk_score > 0.75 → discount by 25%
  Bear   → no discount (risk taken at face value)
  Applied at the very start of arbitrate() before any risk-based logic,
  preventing high systemic-risk scores from triggering SELL in Bull markets.
"""

import numpy as np
import logging

logger = logging.getLogger("ConflictResolver")

# ==============================================================================
# CONFIGURATION THRESHOLDS
# ==============================================================================
CONFLICT_THRESHOLD       = 0.60
UNCERTAINTY_HIGH         = 0.10
SYSTEMIC_VETO_THRESHOLD  = 0.70
HOLD_CONFIDENCE          = 0.50

# v2.2 PATCH — Regime-aware risk discount table
REGIME_RISK_DISCOUNT = {
    "Bull":     (0.60, 0.50),   # risk > 0.60 in Bull  → halved
    "Sideways": (0.75, 0.75),   # risk > 0.75 in Sideways → 25% cut
    "Bear":     (1.01, 1.00),   # Bear → no discount
}


def _apply_regime_risk_discount(risk_score: float, regime_label: str) -> tuple:
    """
    Returns (discounted_risk_score, discount_applied: bool, discount_note: str).
    Called at the top of arbitrate() before any risk-based logic.
    """
    threshold, factor = REGIME_RISK_DISCOUNT.get(regime_label, (1.01, 1.00))
    if risk_score > threshold:
        discounted = risk_score * factor
        note = (
            f"Regime={regime_label}: raw risk {risk_score:.3f} discounted "
            f"to {discounted:.3f} (factor {factor:.2f} — regime-aware gate)"
        )
        return discounted, True, note
    return risk_score, False, ""


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
        self.conflict_threshold       = conflict_threshold
        self.uncertainty_high         = uncertainty_high
        self.systemic_veto_threshold  = systemic_veto_threshold
        print("   ✅ Phase 13: Conflict Resolution Engine v2.2 (Arbitrator) Initialized.")

    # ------------------------------------------------------------------
    # MAIN ARBITRATION ENTRY POINT
    # ------------------------------------------------------------------
    def arbitrate(self, tech_score, sent_score, mc_std,
                  regime_label, risk_score, fusion_confidence,
                  trust_scores=None):
        """
        Evaluate agent disagreement and, if necessary, override the
        fusion confidence with a rule-based decision.
        """
        reasoning  = []
        adjusted_confidence = fusion_confidence
        ruling     = "NO_CONFLICT"
        arbitrated = False

        self._trust = trust_scores or {}

        # ── v2.2 PATCH: Regime-Aware Risk Discount ────────────────────────
        # Apply BEFORE any risk-based confidence adjustments below.
        # Prevents high systemic-risk scores from triggering SELL in Bull market.
        risk_score, discount_applied, discount_note = _apply_regime_risk_discount(
            risk_score, regime_label
        )
        if discount_applied:
            reasoning.append(discount_note)
            arbitrated = True
        # ── END PATCH ─────────────────────────────────────────────────────

        # Normalise Sentiment to 0-1 scale (FinBERT outputs -1 → +1)
        sent_normalised = (sent_score + 1.0) / 2.0
        sent_normalised = max(0.0, min(1.0, sent_normalised))

        # Step 0: Calculate Disagreement Spread
        spread = abs(tech_score - sent_normalised)
        reasoning.append(
            f"Disagreement Spread: {spread:.4f}  "
            f"(Tech={tech_score:.4f}, Sent_norm={sent_normalised:.4f}, "
            f"Threshold={self.conflict_threshold})"
        )

        # =====================================================================
        # TIE-BREAKER C (Highest Priority): SYSTEMIC VETO
        # =====================================================================
        # H3 FIX: Only veto when divergence opposes the technical signal
        # i.e., risk is high AND the LSTM confirms bearish state
        tech_is_bearish = tech_score < 0.45
        if risk_score > self.systemic_veto_threshold and tech_is_bearish:
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

        if spread < self.conflict_threshold:
            # C4 audit trail fix: if sentiment is missing/frozen, it sits at exactly 0.50
            if abs(sent_normalised - 0.50) < 0.001:
                reasoning.append(
                    f"✅  Agents are 'in agreement' (Spread {spread:.4f} "
                    f"< {self.conflict_threshold}) mainly because Sentiment "
                    f"data is neutral/missing."
                )
            else:
                reasoning.append(
                    f"✅  Agents are in agreement (Spread {spread:.4f} "
                    f"< {self.conflict_threshold}).  No arbitration needed."
                )
            adjusted_confidence = self._apply_mild_adjustments(
                fusion_confidence, risk_score, mc_std, reasoning
            )
            return self._build_result(
                arbitrated, fusion_confidence, adjusted_confidence,
                ruling, reasoning
            )

        # =====================================================================
        # CONFLICT DETECTED
        # =====================================================================
        reasoning.append(
            f"🚨  CONFLICT DETECTED: Spread {spread:.4f} ≥ "
            f"{self.conflict_threshold}.  Activating Arbitration."
        )
        arbitrated = True

        # =====================================================================
        # TIE-BREAKER A: BAYESIAN CERTAINTY CHECK
        # =====================================================================
        if mc_std > self.uncertainty_high:
            reasoning.append(
                f"🎲  Bayesian Check: MC StdDev ({mc_std:.4f}) > "
                f"{self.uncertainty_high} → Technical Agent is UNCERTAIN.  "
                f"Favouring Sentiment signal."
            )
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
            reasoning.append(
                f"🎲  Bayesian Check: MC StdDev ({mc_std:.4f}) ≤ "
                f"{self.uncertainty_high} → Both agents are confident."
            )

            # =================================================================
            # TIE-BREAKER A.5: TRUST SCORE CHECK (Phase 14 Meta-Agent)
            # =================================================================
            tech_trust = self._trust.get("technical", 1.0)
            sent_trust = self._trust.get("sentiment", 1.0)
            trust_gap  = abs(tech_trust - sent_trust)

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
                adjusted_confidence, ruling = self._regime_tiebreak(
                    tech_score, sent_normalised, regime_label,
                    fusion_confidence, reasoning
                )

        return self._build_result(
            arbitrated, fusion_confidence, adjusted_confidence,
            ruling, reasoning
        )

    # ------------------------------------------------------------------
    # REGIME TIE-BREAKER
    # ------------------------------------------------------------------
    def _regime_tiebreak(self, tech_score, sent_norm, regime_label,
                         fusion_confidence, reasoning):
        if regime_label == "Bear":
            reasoning.append(
                f"⛈️   Regime = BEAR → 'Pessimism wins.'  "
                f"In volatile bear markets, bullish chart patterns are often "
                f"fake breakouts.  Forcing HOLD."
            )
            return HOLD_CONFIDENCE * 0.8, "HOLD"

        elif regime_label == "Bull":
            bullish_val = max(tech_score, sent_norm)
            reasoning.append(
                f"☀️   Regime = BULL → 'Optimism wins.'  "
                f"Aligning confidence with the more bullish agent "
                f"({bullish_val:.4f})."
            )
            return max(fusion_confidence, bullish_val * 0.9), "ALIGN_BULL"

        else:
            reasoning.append(
                f"🌥️   Regime = SIDEWAYS → Ambiguous context.  "
                f"Forcing conservative HOLD to avoid false signals."
            )
            return HOLD_CONFIDENCE, "HOLD"

    # ------------------------------------------------------------------
    # MILD ADJUSTMENTS (when no major conflict)
    # ------------------------------------------------------------------
    def _apply_mild_adjustments(self, confidence, risk_score, mc_std, reasoning):
        adj = confidence

        if risk_score > 0.40:
            penalty = max(1.0 - (risk_score - 0.40) * 0.5, 0.60)
            adj *= penalty
            reasoning.append(
                f"⚠️   Mild Systemic Penalty: risk_score={risk_score:.4f} "
                f"→ Confidence scaled by {penalty:.2f} → {adj:.4f}"
            )

        if mc_std > 0.05:
            penalty = max(1.0 - (mc_std - 0.05) * 2.0, 0.70)
            adj *= penalty
            reasoning.append(
                f"⚠️   Mild Uncertainty Penalty: mc_std={mc_std:.4f} "
                f"→ Confidence scaled by {penalty:.2f} → {adj:.4f}"
            )

        return adj

    # ------------------------------------------------------------------
    # RESULT BUILDER
    # ------------------------------------------------------------------
    @staticmethod
    def _build_result(arbitrated, original, adjusted, ruling, reasoning):
        return {
            "arbitrated":           arbitrated,
            "original_confidence":  round(original, 4),
            "adjusted_confidence":  round(adjusted, 4),
            "ruling":               ruling,
            "reasoning":            reasoning,
        }

    # ------------------------------------------------------------------
    # PRETTY PRINT
    # ------------------------------------------------------------------
    @staticmethod
    def print_report(result):
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