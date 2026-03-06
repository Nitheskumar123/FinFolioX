"""
PHASE 24: TOPOLOGICAL SHAPE AGENT (Persistent Homology)
--------------------------------------------------------
Implements Research Idea 1: TDA + Persistent Homology for Market Structure.

CORE NOVELTY:
  While HMM detects regimes via STATISTICAL state transitions, this agent
  detects regime structure from the GEOMETRY of the market's attractor
  manifold via Takens Delay Embedding and Persistent Homology — providing
  a fundamentally ORTHOGONAL signal that HMM is mathematically blind to.

Pipeline:
  1. INPUT   : 60-day Close price time series
  2. Takens  : Delay Embedding → Reconstruct the dynamical attractor as R³ point cloud
  3. Ripser  : Vietoris-Rips filtration → Simplicial complex → Persistence diagrams
  4. Extract :
       Betti-0 (H0) → Connected components → Market Fragmentation Score
       Betti-1 (H1) → Loops / 1-cycles   → Oscillation / Mean-Reversion Score
       Persistence Entropy               → Global Chaos Score (pre-crash indicator)
  5. OUTPUT  : topology_chaos_score (0-1), dominant_structure, market_shape_signal

Academic Foundations:
  - Takens (1981)       : "Detecting Strange Attractors in Turbulence"
  - Edelsbrunner & Harer (2010) : "Computational Topology"
  - ripser (Tralie et al. 2018) : https://github.com/scikit-tda/ripser.py  (500+ citations)

Install:
    pip install ripser persim
"""

import numpy as np
import logging

logger = logging.getLogger("TopologyAgent")

# ── Optional TDA imports (graceful degradation) ────────────────────────────
try:
    from ripser import ripser as _ripser
    RIPSER_AVAILABLE = True
except ImportError:
    RIPSER_AVAILABLE = False
    logger.warning(
        "ripser not installed — TopologyAgent running in FALLBACK mode. "
        "Install with: pip install ripser persim"
    )

try:
    from persim import plot_diagrams as _plot_diagrams   # noqa: F401 (visualisation only)
    PERSIM_AVAILABLE = True
except ImportError:
    PERSIM_AVAILABLE = False


# ==============================================================================
# TOPOLOGICAL SHAPE AGENT
# ==============================================================================

class TopologyAgent:
    """
    The Topological Shape Agent — Phase 24.

    Detects the *geometric structure* of the market's attractor manifold
    using Takens Delay Embedding + Persistent Homology.

    Key Topological Signals:
    ─────────────────────────────────────────────────────────────────────
    betti0 (H0):  Number of connected components.
                  Many components → fragmented / volatile market.

    betti1 (H1):  Number of independent 1-cycles (loops).
                  Many loops → oscillating / mean-reverting market.
                  ZERO loops → directional / trending market.

    entropy:      Shannon entropy of persistence lifetimes.
                  High entropy → complex, unpredictable shape → pre-crash signal.
    ─────────────────────────────────────────────────────────────────────

    Final Output:
        topology_chaos_score  (float, 0-1)  Weighted blend
        dominant_structure    (str)  LOOP | TREND | FRAGMENTED | SMOOTH | UNKNOWN
        market_shape_signal   (str)  SIDEWAYS | TRENDING | CHAOTIC | NEUTRAL | UNKNOWN
    """

    # Composite score weights
    W_BETTI0   = 0.30   # fragmentation
    W_BETTI1   = 0.45   # loops / oscillation (primary novelty)
    W_ENTROPY  = 0.25   # global chaos

    def __init__(
        self,
        time_delay: int   = 5,
        dimension: int    = 3,
        lookback: int     = 60,
        betti1_threshold: float = 0.35,
        entropy_threshold: float = 0.60,
    ):
        """
        Args:
            time_delay        : τ for Takens embedding (default = 5 trading days).
            dimension         : Embedding dimension d (default = 3).
            lookback          : Number of historical bars to use (default = 60).
            betti1_threshold  : Normalised H1 score above which → LOOP regime.
            entropy_threshold : Normalised entropy above which → CHAOTIC regime.
        """
        self.time_delay        = time_delay
        self.dimension         = dimension
        self.lookback          = lookback
        self.betti1_threshold  = betti1_threshold
        self.entropy_threshold = entropy_threshold
        self._ready            = RIPSER_AVAILABLE

        status = "✅" if self._ready else "⚠️  (ripser missing — using fallback)"
        print(f"   [+] Phase 24: Topological Shape Agent (TDA) Initialized. {status}")

    # ──────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────────────────────

    def analyze(self, hist_df):
        """
        Run the full TDA pipeline on historical market data.

        Args:
            hist_df : pandas DataFrame with at least a 'Close' column.

        Returns dict:
            betti0               (float)   Normalised H0 score          [0, 1]
            betti1               (float)   Normalised H1 score          [0, 1]
            persistence_entropy  (float)   Normalised entropy score     [0, 1]
            topology_chaos_score (float)   Composite weighted score     [0, 1]
            dominant_structure   (str)     LOOP | TREND | FRAGMENTED | SMOOTH | UNKNOWN
            market_shape_signal  (str)     SIDEWAYS | TRENDING | CHAOTIC | NEUTRAL | UNKNOWN
            topology_modifier    (float)   Confidence modifier for Fusion (0.6 – 1.2)
            h0_bars              (list)    [[birth, death], ...] for serialisation
            h1_bars              (list)    [[birth, death], ...] (-1 = infinite)
            point_cloud          (ndarray) Embedded point cloud (for frontend scatter)
            status               (str)     "ok" | "fallback" | "error"
        """
        if not self._ready:
            return self._fallback_result("ripser_missing")

        try:
            # ── 1. Prepare time series ────────────────────────────────────
            series = self._prepare_series(hist_df)

            # ── 2. Takens Delay Embedding ─────────────────────────────────
            point_cloud = self._takens_embedding(series)

            # ── 3. Vietoris-Rips filtration via ripser ────────────────────
            diagrams = self._compute_persistence(point_cloud)
            if diagrams is None:
                return self._fallback_result("ripser_error")

            # ── 4. Extract topological features ──────────────────────────
            betti0      = self._score_h0(diagrams[0])
            betti1      = self._score_h1(diagrams[1])
            pers_entropy = self._persistence_entropy_score(diagrams[0], diagrams[1])

            # ── 5. Composite chaos score ──────────────────────────────────
            chaos_score = float(np.clip(
                self.W_BETTI0 * betti0 + self.W_BETTI1 * betti1 + self.W_ENTROPY * pers_entropy,
                0.0, 1.0
            ))

            # ── 6. Interpret ──────────────────────────────────────────────
            dominant_structure   = self._classify_structure(betti1, pers_entropy)
            market_shape_signal  = self._market_signal(dominant_structure, chaos_score)
            topology_modifier    = self._confidence_modifier(chaos_score, dominant_structure)

            # ── 7. Serialise diagrams ─────────────────────────────────────
            h0_bars, h1_bars = self._serialise_diagrams(diagrams)

            result = {
                "betti0":               round(betti0, 4),
                "betti1":               round(betti1, 4),
                "persistence_entropy":  round(pers_entropy, 4),
                "topology_chaos_score": round(chaos_score, 4),
                "dominant_structure":   dominant_structure,
                "market_shape_signal":  market_shape_signal,
                "topology_modifier":    round(topology_modifier, 4),
                "h0_bars":              h0_bars,
                "h1_bars":              h1_bars,
                "point_cloud":          point_cloud,
                "status":               "ok",
            }

            self._print_report(result)
            return result

        except Exception as exc:
            logger.error(f"TopologyAgent.analyze failed: {exc}", exc_info=True)
            return self._fallback_result(f"error:{exc}")

    # ──────────────────────────────────────────────────────────────────────
    # STEP 1 — DATA PREPARATION
    # ──────────────────────────────────────────────────────────────────────

    def _prepare_series(self, hist_df):
        """Extract last N Close prices, min-max normalised to [0, 1]."""
        series = hist_df["Close"].values[-self.lookback:].astype(float).flatten()
        mn, mx = series.min(), series.max()
        if (mx - mn) < 1e-8:
            return np.zeros_like(series)
        return (series - mn) / (mx - mn)

    # ──────────────────────────────────────────────────────────────────────
    # STEP 2 — TAKENS DELAY EMBEDDING  (Takens 1981)
    # ──────────────────────────────────────────────────────────────────────

    def _takens_embedding(self, series):
        """
        Reconstructs the dynamical attractor of the market system.

        Theorem (Takens 1981):  Given a scalar measurement x(t) of a smooth
        dynamical system on a d-dimensional attractor, the delay-embedding
          φ(t) = [ x(t), x(t+τ), x(t+2τ), ..., x(t+(d-1)τ) ]
        is generically an embedding of the attractor into R^d.

        For markets this means we reconstruct the hidden 'market phase space'
        from Close prices alone — capturing momentum, oscillation, and regime
        structure that is invisible to linear statistics.

        Args:
            series : 1-D float array, normalised to [0, 1]
        Returns:
            ndarray of shape (n_points, dimension)
        """
        τ = self.time_delay
        d = self.dimension
        n = len(series)
        n_points = n - (d - 1) * τ

        if n_points < 8:
            raise ValueError(
                f"Takens embedding produced only {n_points} points "
                f"(need ≥ 8). Increase lookback or decrease time_delay/dimension."
            )

        cloud = np.zeros((n_points, d), dtype=float)
        for i in range(n_points):
            for k in range(d):
                cloud[i, k] = series[i + k * τ]

        return cloud

    # ──────────────────────────────────────────────────────────────────────
    # STEP 3 — VIETORIS-RIPS + PERSISTENCE
    # ──────────────────────────────────────────────────────────────────────

    def _compute_persistence(self, point_cloud):
        """
        Build Vietoris-Rips filtration and return persistence diagrams.

        diagrams[0] = H0 (connected components)
        diagrams[1] = H1 (independent loops / 1-cycles)

        ripser is chosen over gudhi for its C++ speed core and
        500+ academic citations; gudhi is reserved for visualisation.
        """
        try:
            result = _ripser(point_cloud, maxdim=1)
            return result["dgms"]
        except Exception as exc:
            logger.warning(f"ripser computation failed: {exc}")
            return None

    # ──────────────────────────────────────────────────────────────────────
    # STEP 4 — FEATURE EXTRACTION
    # ──────────────────────────────────────────────────────────────────────

    def _score_h0(self, h0_diagram):
        """
        H0 (Betti-0): Connected components → Market Fragmentation Score.

        One infinite component always survives (the whole dataset).
        We score the SHORT-LIVED components: many transient clusters
        means fragmented, disconnected price action (risk signal).
        """
        if len(h0_diagram) == 0:
            return 0.5

        finite = h0_diagram[np.isfinite(h0_diagram[:, 1])]
        if len(finite) == 0:
            return 0.0  # Single component — coherent, normal

        lifetimes = finite[:, 1] - finite[:, 0]
        # Number of fragments + their average lifetime
        frag_score  = np.tanh(len(finite) / 5.0)         # 0 → 1
        life_score  = np.tanh(float(lifetimes.mean()) * 3.0)  # 0 → 1
        return float(np.clip(0.6 * frag_score + 0.4 * life_score, 0.0, 1.0))

    def _score_h1(self, h1_diagram):
        """
        H1 (Betti-1): Independent loops → Oscillation Score.

        Each loop in the point cloud corresponds to a cyclic / mean-reverting
        structure in the time series. More long-lived loops = stronger
        oscillation signal. ZERO loops = clean directional trend.

        Academically: this is the *geometric* regime detector that HMM cannot
        replicate because HMM operates in the statistical space of returns, not
        the topological space of the attractor manifold.
        """
        if len(h1_diagram) == 0:
            return 0.0

        finite = h1_diagram[np.isfinite(h1_diagram[:, 1])]
        if len(finite) == 0:
            return 0.0

        lifetimes = finite[:, 1] - finite[:, 0]

        # Significant loops only (lifetime > noise threshold = 0.04)
        significant = lifetimes[lifetimes > 0.04]
        count_score = np.tanh(len(significant) / 3.0)         # 0 → 1
        max_score   = np.tanh(float(lifetimes.max()) * 2.5)   # 0 → 1
        return float(np.clip(0.55 * count_score + 0.45 * max_score, 0.0, 1.0))

    def _persistence_entropy_score(self, h0_diagram, h1_diagram):
        """
        Persistence Entropy: Shannon entropy of all bar lifetimes.

        High entropy → complex, heterogeneous topological structure →
        unpredictable market geometry → potential pre-crash signal.

        Reference: Atienza et al. (2020) "On the stability of persistent
        entropy and new summary functions for topological data analysis."
        """
        all_lifetimes = []
        for diag in [h0_diagram, h1_diagram]:
            finite = diag[np.isfinite(diag[:, 1])]
            if len(finite) > 0:
                all_lifetimes.extend((finite[:, 1] - finite[:, 0]).tolist())

        if not all_lifetimes:
            return 0.0

        L = np.array(all_lifetimes, dtype=float)
        L = L[L > 1e-10]
        if len(L) == 0:
            return 0.0

        total  = L.sum()
        probs  = L / total
        entropy = -np.sum(probs * np.log(probs + 1e-12))

        # Normalise by maximum possible entropy log(N)
        max_ent = np.log(max(len(L), 2))
        return float(np.clip(entropy / (max_ent + 1e-10), 0.0, 1.0))

    # ──────────────────────────────────────────────────────────────────────
    # STEP 5 — INTERPRETATION
    # ──────────────────────────────────────────────────────────────────────

    def _classify_structure(self, betti1_score: float, entropy_score: float) -> str:
        """Map topological features to a dominant market structure label."""
        if betti1_score >= self.betti1_threshold:
            return "LOOP"        # Oscillating / mean-reverting geometry
        elif entropy_score >= self.entropy_threshold:
            return "CHAOTIC"     # Complex, disordered shape
        elif betti1_score < 0.12 and entropy_score < 0.30:
            return "TREND"       # Clean directional attractor
        else:
            return "SMOOTH"      # Stable, transitional shape

    def _market_signal(self, dominant_structure: str, chaos_score: float) -> str:
        """Convert dominant structure to a trading-relevant signal."""
        mapping = {
            "LOOP":    "SIDEWAYS",   # Mean-reverting → fade-the-move strategy
            "TREND":   "TRENDING",   # Directional → momentum strategy
            "CHAOTIC": "CHAOTIC",    # Dangerous → reduce exposure
            "SMOOTH":  "NEUTRAL",    # Transitional → wait and watch
        }
        return mapping.get(dominant_structure, "UNKNOWN")

    def _confidence_modifier(self, chaos_score: float, dominant_structure: str) -> float:
        """
        Maps the topology chaos score to a Fusion confidence modifier.

        Low chaos (structured topology) → slight boost to confidence.
        High chaos (disordered topology) → penalty on confidence.

        LOOP structure is particularly penalised for BUY signals because
        the attractor manifold predicts mean-reversion, not breakout.
        """
        base_mod = 1.0 - (chaos_score - 0.5) * 0.50  # 0.5 chaos → 1.0x  |  1.0 chaos → 0.75x
        if dominant_structure == "LOOP":
            base_mod *= 0.90  # extra 10% cut for loop (likely sideways)
        elif dominant_structure == "TREND":
            base_mod = min(base_mod * 1.05, 1.20)  # slight boost for clean trend
        return float(np.clip(base_mod, 0.55, 1.20))

    # ──────────────────────────────────────────────────────────────────────
    # SERIALISATION
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _serialise_diagrams(diagrams):
        """
        Convert numpy persistence diagrams to JSON-serialisable lists.
        Infinite death values are encoded as -1.0.
        """
        def _convert(diag):
            bars = []
            for birth, death in diag:
                b = float(birth)
                d = float(death) if np.isfinite(death) else -1.0
                bars.append([round(b, 5), round(d, 5)])
            return bars

        h0 = _convert(diagrams[0]) if len(diagrams) > 0 else []
        h1 = _convert(diagrams[1]) if len(diagrams) > 1 else []
        return h0, h1

    # ──────────────────────────────────────────────────────────────────────
    # FALLBACK
    # ──────────────────────────────────────────────────────────────────────

    def _fallback_result(self, reason: str = ""):
        """Returns a neutral/informative result when TDA pipeline is unavailable."""
        logger.info(f"TopologyAgent returning fallback result. Reason: {reason}")
        return {
            "betti0":               0.5,
            "betti1":               0.5,
            "persistence_entropy":  0.5,
            "topology_chaos_score": 0.5,
            "dominant_structure":   "UNKNOWN",
            "market_shape_signal":  "UNKNOWN",
            "topology_modifier":    1.0,
            "h0_bars":              [],
            "h1_bars":              [],
            "point_cloud":          None,
            "status":               f"fallback:{reason}" if reason else "fallback",
        }

    # ──────────────────────────────────────────────────────────────────────
    # CONSOLE REPORT
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _print_report(result):
        score = result["topology_chaos_score"]
        bar_len = int(score * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)

        print("\n   ╔══════════════════════════════════════════════════╗")
        print("   ║   PHASE 24 — TOPOLOGICAL SHAPE AGENT (TDA)      ║")
        print("   ╠══════════════════════════════════════════════════╣")
        print(f"   ║  H0 Fragmentation Score : {result['betti0']:.4f}                ║")
        print(f"   ║  H1 Oscillation Score   : {result['betti1']:.4f}                ║")
        print(f"   ║  Persistence Entropy    : {result['persistence_entropy']:.4f}                ║")
        print(f"   ╠══════════════════════════════════════════════════╣")
        print(f"   ║  Topology Chaos Score   : {score:.4f}  [{bar}]  ║")
        print(f"   ║  Dominant Structure     : {result['dominant_structure']:<16s}          ║")
        print(f"   ║  Market Shape Signal    : {result['market_shape_signal']:<16s}          ║")
        print(f"   ║  Fusion Modifier        : {result['topology_modifier']:.4f}x               ║")
        print("   ╚══════════════════════════════════════════════════╝")