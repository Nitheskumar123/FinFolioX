"""
PHASE 16: AGENT DISAGREEMENT HEATMAP (The "Boardroom Tension" Metric)
----------------------------------------------------------------------
Exposes the hidden ensemble variance that standard Fusion hides.

A confidence of 0.50 from three agreeing agents is SAFE.
A confidence of 0.50 from warring agents is DANGEROUS.

Phase 16 calculates the Global Disagreement Index (GDI) — a single
number from 0% (total harmony) to 100% (total chaos) — and uses it
to automatically shrink position sizes when the boardroom is fighting.

Three Components:
  A. Disagreement Matrix  – Pairwise distance between every agent.
  B. Position Sizing Penalty – GDI > 40% → Kelly allocation halved.
  C. ASCII Visual Heatmap  – Terminal grid showing the boardroom tension.
"""

import numpy as np


# ==============================================================================
# CONFIGURATION
# ==============================================================================
GDI_LOW_THRESHOLD = 0.20       # Below 20% = Harmony (green)
GDI_MED_THRESHOLD = 0.40       # 20-40% = Moderate tension (yellow)
GDI_HIGH_THRESHOLD = 0.60      # 40-60% = High tension (orange)
# Above 60% = Extreme tension (red)

PENALTY_NONE = 1.0             # GDI < 20%: no penalty
PENALTY_MODERATE = 0.75        # GDI 20-40%: reduce allocation by 25%
PENALTY_HIGH = 0.50            # GDI 40-60%: reduce allocation by 50%
PENALTY_EXTREME = 0.25         # GDI > 60%: reduce allocation by 75%


class HeatmapAgent:
    """
    The Boardroom Tension Monitor.

    Takes raw agent scores, normalizes them to a common [0, 1] scale,
    calculates pairwise disagreements, and produces:
      1. A 3x3 Disagreement Matrix
      2. A Global Disagreement Index (GDI)
      3. A position sizing penalty multiplier
      4. A visual ASCII heatmap for the console
    """

    def __init__(self):
        print("   [+] Phase 16: Heatmap Agent (Boardroom Tension) Initialized.")

    def analyze(self, lstm_score, sent_score, regime_label, regime_vol=0.5):
        """
        Main entry point. Takes raw agent scores and produces the
        full disagreement analysis.

        Args:
            lstm_score:   Technical agent output (0.0 to 1.0)
            sent_score:   Sentiment agent output (-1.0 to +1.0)
            regime_label: Market regime string ("Bull", "Bear", "Sideways")
            regime_vol:   Current volatility (used for regime normalization)

        Returns dict:
            matrix:       3x3 numpy array of pairwise spreads
            agents:       dict of normalized agent scores
            gdi:          Global Disagreement Index (0.0 to 1.0)
            gdi_pct:      GDI as a percentage string
            tension:      "HARMONY" / "MODERATE" / "HIGH" / "EXTREME"
            penalty:      Position sizing multiplier (0.25 to 1.0)
            pairs:        dict of pairwise distances
        """
        # Step 1: Normalize all agents to the same [0, 1] scale
        norm_lstm = max(0.0, min(1.0, lstm_score))
        norm_sent = max(0.0, min(1.0, (sent_score + 1.0) / 2.0))
        norm_regime = self._regime_to_score(regime_label)

        agents = {
            "LSTM": round(norm_lstm, 4),
            "FinBERT": round(norm_sent, 4),
            "Regime": round(norm_regime, 4),
        }

        # Step 2: Calculate pairwise disagreements
        spread_lf = abs(norm_lstm - norm_sent)      # LSTM vs FinBERT
        spread_lr = abs(norm_lstm - norm_regime)     # LSTM vs Regime
        spread_fr = abs(norm_sent - norm_regime)     # FinBERT vs Regime

        pairs = {
            "LSTM_vs_FinBERT": round(spread_lf, 4),
            "LSTM_vs_Regime": round(spread_lr, 4),
            "FinBERT_vs_Regime": round(spread_fr, 4),
        }

        # Step 3: Build the 3x3 matrix
        matrix = np.array([
            [0.0,       spread_lf, spread_lr],
            [spread_lf, 0.0,       spread_fr],
            [spread_lr, spread_fr, 0.0      ],
        ])

        # Step 4: Global Disagreement Index
        # H5 FIX: If sentiment is frozen/missing, use only LSTM vs Regime distance
        sentiment_frozen = abs(sent_score) < 0.001
        if sentiment_frozen:
            gdi = spread_lr * 1.5
        else:
            gdi = np.mean([spread_lf, spread_lr, spread_fr]) * 1.5
            
        if gdi > 0.40:
            print(f"      ⚠️ [Heatmap] High Boardroom Tension detected: {gdi*100:.1f}%")
            
        gdi = max(0.0, min(1.0, gdi))  # Cap at 100%, not 20%

        # Step 5: Classify tension level and penalty
        tension, penalty = self._classify_tension(gdi)

        return {
            "matrix": matrix,
            "agents": agents,
            "gdi": round(gdi, 4),
            "gdi_pct": f"{gdi * 100:.1f}%",
            "tension": tension,
            "penalty": penalty,
            "pairs": pairs,
        }

    def _regime_to_score(self, regime_label):
        """Convert regime label to a bullish/bearish score on [0, 1]."""
        label = str(regime_label).strip().lower()
        if label == "bull":
            return 0.65
        elif label == "bear":
            return 0.35
        else:
            return 0.50  # Sideways

    def _classify_tension(self, gdi):
        """Returns (tension_label, penalty_multiplier) based on GDI."""
        if gdi < GDI_LOW_THRESHOLD:
            return "HARMONY", PENALTY_NONE
        elif gdi < GDI_MED_THRESHOLD:
            return "MODERATE", PENALTY_MODERATE
        elif gdi < GDI_HIGH_THRESHOLD:
            return "HIGH", PENALTY_HIGH
        else:
            return "EXTREME", PENALTY_EXTREME

    def get_sizing_penalty(self, gdi):
        """
        Public API — returns the Kelly penalty multiplier for a given GDI.
        Used by the Risk Engine to shrink position sizes.
        """
        _, penalty = self._classify_tension(gdi)
        return penalty

    # ------------------------------------------------------------------
    # ASCII VISUAL HEATMAP
    # ------------------------------------------------------------------
    @staticmethod
    def print_heatmap(result):
        """
        Renders the Disagreement Matrix as a styled ASCII grid.

        Color coding (via emoji):
          0.00 - 0.20: Green  (Agreement)
          0.20 - 0.40: Yellow (Mild tension)
          0.40 - 0.60: Orange (High tension)
          0.60 - 1.00: Red    (War zone)
        """
        agents = result["agents"]
        matrix = result["matrix"]
        gdi = result["gdi"]
        tension = result["tension"]
        penalty = result["penalty"]

        # Emoji for tension level
        tension_emoji = {
            "HARMONY": "##",
            "MODERATE": "!!",
            "HIGH": "!!",
            "EXTREME": "XX",
        }

        labels = ["LSTM", "FinBERT", "Regime"]

        print(f"\n   {'=' * 60}")
        print(f"   {tension_emoji.get(tension, '??')} [Disagreement Heatmap] "
              f"Boardroom Tension: {tension} ({gdi*100:.1f}% GDI)")
        print(f"   {'=' * 60}")

        # Agent scores row
        print(f"   Agent Positions (Normalized 0-1 Bullish Scale):")
        for name, score in agents.items():
            bar_len = int(score * 20)
            bar = "#" * bar_len + "." * (20 - bar_len)
            print(f"      {name:8s}: {score:.4f}  [{bar}]")

        # Matrix header
        print(f"\n   Pairwise Distance Matrix:")
        print(f"   {'':13s}| {'LSTM':^10s}| {'FinBERT':^10s}| {'Regime':^10s}|")
        print(f"   {'-'*13}|{'-'*11}|{'-'*11}|{'-'*11}|")

        for i, row_label in enumerate(labels):
            row_str = f"   {row_label:13s}|"
            for j in range(3):
                if i == j:
                    cell = f"{'--':^10s}"
                else:
                    val = matrix[i][j]
                    icon = _cell_icon(val)
                    cell = f" {val:.2f} {icon}  "
                row_str += f"{cell}|"
            print(row_str)

        print(f"   {'-'*13}|{'-'*11}|{'-'*11}|{'-'*11}|")

        # Impact line
        if penalty < 1.0:
            cut_pct = int((1.0 - penalty) * 100)
            print(f"\n   !! IMPACT: {tension} tension detected. "
                  f"Shrinking Kelly allocation by {cut_pct}%.")
        else:
            print(f"\n   ** IMPACT: Boardroom in harmony. "
                  f"No position sizing penalty.")

        print(f"   {'=' * 60}")


def _cell_icon(val):
    """Returns a text indicator for the heatmap cell."""
    if val < 0.20:
        return "[OK]"
    elif val < 0.40:
        return "[~~]"
    elif val < 0.60:
        return "[!!]"
    else:
        return "[XX]"
