"""
ml_engine/asc_memory.py  —  Agent Sycophancy Coefficient (ASC) Engine
======================================================================
Phase 26 — Fixed v2.2

FIXES IN THIS VERSION:

  FIX-1 · KSG Saturation Guard (Root cause of 50% trigger rate)
    The buffer fills at stock #15 with 14 near-identical warm-up sessions.
    KSG measures high mutual information across that homogeneous batch and
    reports ASC = 0.93-1.00 on every subsequent stock. That is NOT sycophancy
    — it is the estimator saturating on a low-variance corpus.
    Detection: if std(lstm_arr) < SATURATION_STD_THRESHOLD (0.04), suppress
    the penalty entirely and log asc_saturated=True.

  FIX-2 · Raised minimum reliable window: 15 -> 20
    At 15 sessions the KSG estimator does not have enough entropy to separate
    genuine sycophancy from normal inter-agent correlation.

  FIX-3 · Raised penalty fire threshold: 0.70 -> 0.85
    Scores 0.70-0.84 represent normal correlated-but-healthy ensemble behaviour
    in a batch where all stocks run through the same pipeline on the same day.
    Only scores >= 0.85 warrant a penalty.

  FIX-4 · Softer graduated penalty table
    Old:  STRONG >= 0.70 -> -35%,  EXTREME >= 0.70 + high DS -> -50%
    New:  MILD   >= 0.50 -> -5%
          MODERATE >= 0.70 -> -15%
          HIGH    >= 0.85 -> -25%
          EXTREME >= 0.95 -> -35%
"""

import os
import pickle
import logging
import numpy as np
from collections import deque
from typing import Optional, Tuple, Dict

logger = logging.getLogger("ASCMemory")

try:
    from sklearn.feature_selection import mutual_info_regression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not found — ASC will use fallback correlation estimator.")


# ==============================================================================
# CONSTANTS
# ==============================================================================

WINDOW_SIZE               = 30
MIN_RELIABLE_SAMPLES      = 20       # FIX-2: was 15
N_HISTOGRAM_BINS          = 10

# FIX-1: std of LSTM scores below this = KSG is saturating on a homogeneous batch
SATURATION_STD_THRESHOLD  = 0.02

# ASC zone boundaries
ASC_LOW_THRESHOLD         = 0.50    # Below -> no penalty
ASC_MED_THRESHOLD         = 0.70    # Mild zone
ASC_HIGH_THRESHOLD        = 0.85    # FIX-3: penalty zone starts here (was 0.70)
ASC_EXTREME_THRESHOLD     = 0.95    # Extreme zone

# Dissent Sensitivity thresholds (unchanged)
DS_LOW_THRESHOLD          = 0.10
DS_HIGH_THRESHOLD         = 0.25

# FIX-4: rebalanced penalty multipliers
PENALTY_NONE              = 1.00   # ASC < 0.50
PENALTY_MILD              = 0.95   # ASC 0.50-0.70   (was 0.85)
PENALTY_MODERATE          = 0.85   # ASC 0.70-0.85   (was 0.75)
PENALTY_HIGH              = 0.75   # ASC 0.85-0.95   (was 0.65)
PENALTY_EXTREME           = 0.65   # ASC >= 0.95     (was 0.50)


# ==============================================================================
# AGENT DECISION MEMORY
# ==============================================================================

class AgentDecisionMemory:
    """
    Rolling buffer that stores raw agent outputs, computes ASC via KSG
    mutual information, and maps (ASC, dissent_sensitivity) to a
    confidence penalty multiplier applied before the Conflict Resolver.

    v2.2 changes: saturation guard (FIX-1), raised window (FIX-2),
    raised threshold (FIX-3), softer penalties (FIX-4).
    """

    def __init__(self, window_size: int = WINDOW_SIZE, cache_path: Optional[str] = None):
        self.window_size = window_size

        if cache_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            cache_path = os.path.join(base_dir, "data", "meta", "asc_buffer.pkl")
        self.cache_path = cache_path

        self.buffer: deque = self._load_buffer()

        print(f"   [+] Phase 26: ASC Memory Engine v2.2 Initialized.")
        print(f"      - Window      : {window_size} sessions")
        print(f"      - Buffer      : {len(self.buffer)}/{window_size} sessions loaded")
        print(f"      - Min reliable: {MIN_RELIABLE_SAMPLES} (raised from 15)")
        print(f"      - Penalty gate: ASC >= {ASC_HIGH_THRESHOLD} (raised from 0.70)")
        status = "RELIABLE" if len(self.buffer) >= MIN_RELIABLE_SAMPLES else \
                 f"WARMING ({len(self.buffer)}/{MIN_RELIABLE_SAMPLES})"
        print(f"      - Status      : {status}")

    # ── Persistence ───────────────────────────────────────────────────────

    def _load_buffer(self) -> deque:
        try:
            if os.path.exists(self.cache_path):
                with open(self.cache_path, "rb") as f:
                    data = pickle.load(f)
                
                # M5 FIX: Upgrade old 3-tuple entries to 4-tuples with a dummy timestamp
                upgraded_data = []
                for item in data:
                    if len(item) == 3:
                        # Append timestamp 0.0 for old entries (treated as extremely stale)
                        upgraded_data.append(tuple(list(item) + [0.0]))
                    else:
                        upgraded_data.append(item)
                        
                return deque(upgraded_data, maxlen=self.window_size)
        except Exception as e:
            logger.warning(f"Could not load ASC buffer: {e}")
        return deque(maxlen=self.window_size)

    def _save_buffer(self):
        try:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, "wb") as f:
                pickle.dump(list(self.buffer), f)
        except Exception as e:
            logger.warning(f"Could not save ASC buffer: {e}")

    # ── Step 1: Record session ────────────────────────────────────────────

    def record_session(self, lstm_score: float, sent_score: float, regime_prob: float):
        """
        Append one session's raw agent outputs to the rolling buffer.
        Anti-spam: skip if LSTM score is identical to the last entry (same session).
        """
        import time as _time
        entry = (
            float(np.clip(lstm_score,  0.0, 1.0)),
            float(np.clip(sent_score, -1.0, 1.0)),
            float(np.clip(regime_prob, 0.0, 1.0)),
            _time.time(), # timestamp for staleness detection
        )
        if len(self.buffer) > 0 and abs(self.buffer[-1][0] - entry[0]) < 0.001:
            return
        self.buffer.append(entry)
        self._save_buffer()

    @staticmethod
    def regime_label_to_prob(regime_label: str) -> float:
        """Convert HMM regime label to a continuous bullish probability."""
        label = str(regime_label).strip().lower()
        if label == "bull":   return 0.80
        elif label == "bear": return 0.20
        else:                 return 0.50

    # ── Step 2: Compute ASC ───────────────────────────────────────────────

    def compute_asc(self) -> Dict:
        """
        Compute the Agent Sycophancy Coefficient over the current buffer.

        Returns:
            asc             (float)  0-1
            asc_reliable    (bool)   False if buffer < MIN_RELIABLE_SAMPLES
            asc_saturated   (bool)   FIX-1: True if LSTM variance too low to trust
            lstm_std        (float)  Standard deviation of LSTM scores in window
            ... MI and entropy fields ...
        """
        n = len(self.buffer)

        if n < MIN_RELIABLE_SAMPLES:
            return {
                "asc": 0.50, "asc_reliable": False, "asc_saturated": False,
                "lstm_std": 0.0, "mi_lstm_sent": 0.0, "mi_lstm_hmm": 0.0,
                "mi_sent_hmm": 0.0, "h_lstm": 0.0, "h_sent": 0.0,
                "h_hmm": 0.0, "n_samples": n,
            }

        # Handle old 3-tuple entries vs new 4-tuple entries (with timestamp)
        arr      = np.array(list(self.buffer))
        # Ensure we only take the first 3 columns for computation, ignoring timestamp
        lstm_arr = arr[:, 0]
        sent_arr = (arr[:, 1] + 1.0) / 2.0   # normalise to [0,1]
        hmm_arr  = arr[:, 2]

        # FIX-1: Saturation guard — check LSTM variance before trusting KSG
        lstm_std  = float(np.std(lstm_arr))
        # Additional guard: require at least 25 samples for reliable variance
        saturated = lstm_std < SATURATION_STD_THRESHOLD or n < 25
        if saturated:
            logger.info(
                f"ASC saturation: LSTM std={lstm_std:.4f} < {SATURATION_STD_THRESHOLD}. "
                "Buffer too homogeneous — KSG output unreliable, penalty suppressed."
            )
            print(f"      ⚠️  [ASC] Saturation detected (LSTM std={lstm_std:.4f}). "
                  "Penalty suppressed.")

        mi_lstm_sent = self._compute_mi(lstm_arr, sent_arr)
        mi_lstm_hmm  = self._compute_mi(lstm_arr, hmm_arr)
        mi_sent_hmm  = self._compute_mi(sent_arr, hmm_arr)

        h_lstm = self._compute_entropy(lstm_arr)
        h_sent = self._compute_entropy(sent_arr)
        h_hmm  = self._compute_entropy(hmm_arr)

        sum_mi = mi_lstm_sent + mi_lstm_hmm + mi_sent_hmm
        sum_h  = h_lstm + h_sent + h_hmm

        asc = 1.0 if sum_h < 1e-8 else float(
            np.clip(1.0 - (sum_mi / (sum_h + 1e-8)), 0.0, 1.0)
        )

        return {
            "asc":           round(asc, 4),
            "asc_reliable":  True,
            "asc_saturated": saturated,
            "lstm_std":      round(lstm_std, 4),
            "mi_lstm_sent":  round(mi_lstm_sent, 4),
            "mi_lstm_hmm":   round(mi_lstm_hmm, 4),
            "mi_sent_hmm":   round(mi_sent_hmm, 4),
            "h_lstm":        round(h_lstm, 4),
            "h_sent":        round(h_sent, 4),
            "h_hmm":         round(h_hmm, 4),
            "n_samples":     n,
        }

    def _compute_mi(self, x: np.ndarray, y: np.ndarray) -> float:
        try:
            if SKLEARN_AVAILABLE:
                mi = mutual_info_regression(
                    x.reshape(-1, 1), y, n_neighbors=3, random_state=42
                )[0]
                return float(max(mi, 0.0))
            else:
                r = float(np.corrcoef(x, y)[0, 1])
                r = np.clip(r, -0.9999, 0.9999)
                return float(-0.5 * np.log(1.0 - r ** 2))
        except Exception as e:
            logger.debug(f"MI estimation failed: {e}")
            return 0.0

    def _compute_entropy(self, x: np.ndarray) -> float:
        try:
            counts, _ = np.histogram(x, bins=N_HISTOGRAM_BINS, range=(0.0, 1.0))
            total = counts.sum()
            if total == 0:
                return 0.0
            probs = counts[counts > 0] / total
            return float(-np.sum(probs * np.log(probs + 1e-12)))
        except Exception as e:
            logger.debug(f"Entropy estimation failed: {e}")
            return 0.0

    # ── Step 3: Forced Dissent Protocol (FDP) ────────────────────────────

    def run_forced_dissent(
        self,
        lstm_signal: float,
        sent_score: float,
        regime_label: str,
        fusion_agent,
        trust_scores: Optional[Dict] = None,
    ) -> Dict:
        """
        Invert LSTM signal, re-run Fusion, measure Dissent Sensitivity.
        Read-only synthetic test — does NOT update any system state.
        """
        vol_input = (
            0.9 if regime_label.strip().lower() == "bear"
            else 0.2 if regime_label.strip().lower() == "bull"
            else 0.5
        )

        try:
            conf_original, _ = fusion_agent.predict(
                lstm_p=lstm_signal, sent_s=sent_score,
                vol_v=vol_input, trust_scores=trust_scores,
            )
            conf_original = float(conf_original)
        except Exception as e:
            logger.warning(f"FDP original fusion failed: {e}")
            return self._fdp_fallback()

        lstm_inverted = float(1.0 - lstm_signal)

        try:
            conf_inverted, _ = fusion_agent.predict(
                lstm_p=lstm_inverted, sent_s=sent_score,
                vol_v=vol_input, trust_scores=trust_scores,
            )
            conf_inverted = float(conf_inverted)
        except Exception as e:
            logger.warning(f"FDP inverted fusion failed: {e}")
            return self._fdp_fallback()

        ds = float(abs(conf_original - conf_inverted))

        if ds < DS_LOW_THRESHOLD:
            interp = (
                f"LSTM barely influences fusion (DS={ds:.3f}). "
                "Decision driven by FinBERT + HMM. Effective ensemble size ~ 2 agents."
            )
        elif ds < DS_HIGH_THRESHOLD:
            interp = (
                f"LSTM has moderate fusion influence (DS={ds:.3f}). "
                "All three agents contribute; FinBERT/HMM dominate."
            )
        else:
            interp = (
                f"LSTM is the dominant fusion driver (DS={ds:.3f}). "
                "In a sycophantic ensemble this single agent controls outcome. "
                "High structural fragility detected."
            )

        return {
            "confidence_original": round(conf_original, 4),
            "confidence_inverted": round(conf_inverted, 4),
            "dissent_sensitivity": round(ds, 4),
            "lstm_inverted":       round(lstm_inverted, 4),
            "interpretation":      interp,
            "fdp_ran":             True,
        }

    def _fdp_fallback(self) -> Dict:
        return {
            "confidence_original": 0.5, "confidence_inverted": 0.5,
            "dissent_sensitivity": 0.0, "lstm_inverted": 0.5,
            "interpretation": "FDP could not run — fusion agent error. Neutral result.",
            "fdp_ran": False,
        }

    # ── Step 4: Penalty multiplier ────────────────────────────────────────

    def get_penalty_multiplier(
        self,
        asc: float,
        dissent_sensitivity: float,
        asc_saturated: bool = False,
    ) -> Tuple[float, str]:
        """
        Map (ASC, DS) to a confidence penalty multiplier and label.

        FIX-1: asc_saturated=True -> always return PENALTY_NONE.
        FIX-3: penalty zone starts at ASC_HIGH_THRESHOLD (0.85), not 0.70.
        FIX-4: softer penalty values throughout.
        """
        # FIX-1: Never penalise when KSG output is unreliable
        if asc_saturated:
            return PENALTY_NONE, "KSG SATURATED — homogeneous batch, no penalty"

        if asc < ASC_LOW_THRESHOLD:
            return PENALTY_NONE, "INDEPENDENT — healthy ensemble, no penalty"

        if asc < ASC_MED_THRESHOLD:
            # 0.50 - 0.70: mild zone, minimal impact
            return PENALTY_MILD, "MILD SYCOPHANCY — correlated but acceptable (−5%)"

        if asc < ASC_HIGH_THRESHOLD:
            # 0.70 - 0.85: moderate zone
            if dissent_sensitivity < DS_LOW_THRESHOLD:
                return PENALTY_MODERATE, "MODERATE SYCOPHANCY — low dominance (−15%)"
            else:
                return PENALTY_MODERATE, "MODERATE SYCOPHANCY — high fragility (−15%)"

        if asc < ASC_EXTREME_THRESHOLD:
            # 0.85 - 0.95: strong zone  (FIX-4: was -35%, now -25%)
            if dissent_sensitivity < DS_HIGH_THRESHOLD:
                return PENALTY_HIGH, "STRONG SYCOPHANCY — low dominance (−25%)"
            else:
                return PENALTY_HIGH, "STRONG SYCOPHANCY — LSTM dominant (−25%)"

        # >= 0.95: extreme  (FIX-4: was -50%, now -35%)
        return PENALTY_EXTREME, "EXTREME SYCOPHANCY — ensemble collapsed (−35%)"

    # ── Summary ───────────────────────────────────────────────────────────

    def get_asc_summary(
        self,
        asc_result: Dict,
        fdp_result: Optional[Dict] = None,
        penalty: float = 1.0,
        quadrant: str = "",
    ) -> Dict:
        return {
            "asc_score":              asc_result.get("asc", 0.5),
            "asc_reliable":           asc_result.get("asc_reliable", False),
            "asc_saturated":          asc_result.get("asc_saturated", False),
            "lstm_std":               asc_result.get("lstm_std", 0.0),
            "n_samples":              asc_result.get("n_samples", 0),
            "mi_lstm_sent":           asc_result.get("mi_lstm_sent", 0.0),
            "mi_lstm_hmm":            asc_result.get("mi_lstm_hmm", 0.0),
            "mi_sent_hmm":            asc_result.get("mi_sent_hmm", 0.0),
            "h_lstm":                 asc_result.get("h_lstm", 0.0),
            "h_sent":                 asc_result.get("h_sent", 0.0),
            "h_hmm":                  asc_result.get("h_hmm", 0.0),
            "asc_penalty_multiplier": round(penalty, 4),
            "asc_quadrant":           quadrant,
            "fdp_ran":                fdp_result.get("fdp_ran", False) if fdp_result else False,
            "dissent_sensitivity":    fdp_result.get("dissent_sensitivity", 0.0) if fdp_result else 0.0,
            "fdp_interpretation":     fdp_result.get("interpretation", "") if fdp_result else "",
        }

    # ── Console report ────────────────────────────────────────────────────

    @staticmethod
    def print_asc_report(summary: Dict):
        asc      = summary.get("asc_score", 0.5)
        n        = summary.get("n_samples", 0)
        pen      = summary.get("asc_penalty_multiplier", 1.0)
        quad     = summary.get("asc_quadrant", "")
        fdp      = summary.get("fdp_ran", False)
        ds       = summary.get("dissent_sensitivity", 0.0)
        sat      = summary.get("asc_saturated", False)
        lstm_std = summary.get("lstm_std", 0.0)
        bar      = "█" * int(asc * 28) + "░" * (28 - int(asc * 28))

        print("\n   ╔══════════════════════════════════════════════════╗")
        print("   ║   PHASE 26 — ASC ENGINE v2.2                     ║")
        print("   ╠══════════════════════════════════════════════════╣")
        print(f"   ║  ASC Score  : {asc:.4f}  [{bar}]  ║")
        print(f"   ║  Samples    : {n}/{WINDOW_SIZE}  |  LSTM std: {lstm_std:.4f}  Saturated: {str(sat):<5}  ║")
        print(f"   ║  Quadrant   : {quad:<42s}  ║")
        print(f"   ║  FDP Ran    : {'YES' if fdp else 'NO '}  |  Dissent Sensitivity : {ds:.4f}           ║")
        print(f"   ║  Penalty    : {pen:.2f}x applied to fusion confidence           ║")
        print("   ╚══════════════════════════════════════════════════╝")