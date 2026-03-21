/**
 * frontend/src/pages/LiveInference.jsx  —  Live Inference Terminal (Phase 26 Updated)
 * =====================================================================================
 */

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
    Search, TrendingUp, TrendingDown, Minus, Brain, Shield,
    BarChart3, AlertTriangle, Layers, Target, Activity, Cpu, ArrowRight
} from 'lucide-react';
import { analyzeStock } from '../services/api';
import EnsembleHealthPanel from '../components/EnsembleHealthPanel';

export default function LiveInference() {
    const [ticker, setTicker] = useState('');
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState('');

    const handleAnalyze = async () => {
        if (!ticker.trim()) return;
        setLoading(true);
        setError('');
        setResult(null);
        try {
            const data = await analyzeStock(ticker.trim());
            setResult(data);
        } catch (err) {
            setError(err.response?.data?.error || err.message || 'Analysis failed');
        } finally {
            setLoading(false);
        }
    };

    const getVerdictClass = (action) => {
        if (!action) return 'verdict-hold';
        const a = action.toUpperCase();
        if (a.includes('BUY')) return 'verdict-buy';
        if (a.includes('SELL')) return 'verdict-sell';
        return 'verdict-hold';
    };

    const getDecisionBadge = (action) => {
        if (!action) return 'badge-hold';
        const a = action.toUpperCase();
        if (a.includes('BUY')) return 'badge-buy';
        if (a.includes('SELL')) return 'badge-sell';
        return 'badge-hold';
    };

    const getTensionBadge = (tension) => {
        const t = (tension || '').toUpperCase();
        if (t === 'HARMONY') return 'badge-harmony';
        if (t === 'MODERATE') return 'badge-moderate';
        if (t === 'HIGH') return 'badge-high';
        if (t === 'EXTREME') return 'badge-extreme';
        return 'badge-normal';
    };

    // --- NEW: Calculate the "Raw" Pre-ASC Decision ---
    const getOriginalDecision = (res) => {
        if (!res) return "UNKNOWN";
        const adjConf = res.fusion?.confidence || 0;
        const penalty = res.ensemble_health?.asc_penalty_multiplier || 1.0;

        // Reverse-engineer the raw confidence before the penalty was applied
        const rawConf = penalty > 0 ? (adjConf / penalty) : adjConf;
        const regime = res.regime?.label;
        const gdi = res.disagreement?.gdi || 0;

        // Apply standard strategy rules
        if (rawConf >= 0.50 && regime !== 'Bear' && gdi < 55.0) {
            return "BUY 🟢";
        } else if (rawConf < 0.40) {
            return "SELL 🔴";
        } else {
            return "HOLD 🟡";
        }
    };

    const originalAction = result ? getOriginalDecision(result) : "N/A";
    const finalAction = result?.decision?.action || "N/A";

    return (
        <div className="page-container">
            <h1 className="page-title">Live Inference Terminal</h1>
            <p className="page-subtitle">
                Type a stock ticker and watch 11 AI agents collaborate in real-time
            </p>

            {/* Search Bar */}
            <div className="search-container">
                <input
                    className="search-input"
                    type="text"
                    placeholder="Enter ticker (e.g., AAPL, NVDA, TSLA)"
                    value={ticker}
                    onChange={(e) => setTicker(e.target.value.toUpperCase())}
                    onKeyDown={(e) => e.key === 'Enter' && handleAnalyze()}
                    disabled={loading}
                />
                <button
                    className="btn btn-primary"
                    onClick={handleAnalyze}
                    disabled={loading || !ticker.trim()}
                >
                    <Search size={18} />
                    {loading ? 'Analyzing...' : 'Analyze'}
                </button>
            </div>

            {/* Loading */}
            <AnimatePresence>
                {loading && (
                    <motion.div
                        className="loading-overlay"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                    >
                        <div className="spinner" />
                        <p className="loading-text">Orchestrating 11 AI Agents...</p>
                        <p style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>
                            LSTM → FinBERT → HMM → Topology → Causal →
                            Fusion → ASC-Check → Arbitrator → Red Team → Groq LLM
                        </p>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Error */}
            {error && (
                <div className="card" style={{ borderColor: 'var(--accent-red)', marginBottom: '1.5rem' }}>
                    <div className="card-header"><AlertTriangle /> Error</div>
                    <p style={{ color: 'var(--accent-red)' }}>{error}</p>
                </div>
            )}

            {/* Results */}
            {result && (
                <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>

                    {/* Row 1: Verdict + Summary */}
                    <div className="grid-2" style={{ marginBottom: '1.5rem' }}>
                        <div className="card" style={{
                            display: 'flex', flexDirection: 'column',
                            alignItems: 'center', justifyContent: 'center', gap: '1rem',
                        }}>
                            <div className="card-header" style={{ alignSelf: 'flex-start' }}>
                                <Target /> The Verdict
                            </div>

                            {/* --- NEW: Side-by-Side Decision Comparison --- */}
                            <div style={{ display: 'flex', gap: '1rem', alignItems: 'center', justifyContent: 'center', width: '100%', marginTop: '0.5rem' }}>

                                {/* Raw Agent Signal */}
                                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '6px' }}>
                                    <span style={{ fontSize: '0.7rem', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                                        Raw AI Signal
                                    </span>
                                    <div className={`verdict-badge ${getVerdictClass(originalAction)}`} style={{ padding: '8px 16px', fontSize: '1rem', opacity: 0.85 }}>
                                        {originalAction?.includes('BUY') ? <TrendingUp size={20} /> :
                                            originalAction?.includes('SELL') ? <TrendingDown size={20} /> :
                                                <Minus size={20} />}
                                        {' '}{originalAction}
                                    </div>
                                </div>

                                {/* Arrow Indicator */}
                                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', color: 'var(--text-muted)' }}>
                                    <span style={{ fontSize: '0.55rem', fontWeight: 700, letterSpacing: '0.5px', marginBottom: '2px' }}>ASC CHECK</span>
                                    <ArrowRight size={24} color={result.ensemble_health?.asc_penalty_multiplier < 1.0 ? 'var(--accent-amber)' : 'var(--text-muted)'} />
                                </div>

                                {/* Final Adjusted Decision */}
                                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '6px' }}>
                                    <span style={{ fontSize: '0.7rem', fontWeight: 700, color: 'var(--text-primary)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                                        Final Decision
                                    </span>
                                    <div className={`verdict-badge ${getVerdictClass(finalAction)}`} style={{ padding: '10px 20px', fontSize: '1.15rem', boxShadow: '0 4px 15px rgba(0,0,0,0.15)' }}>
                                        {finalAction?.includes('BUY') ? <TrendingUp size={24} /> :
                                            finalAction?.includes('SELL') ? <TrendingDown size={24} /> :
                                                <Minus size={24} />}
                                        {' '}{finalAction}
                                    </div>
                                </div>
                            </div>

                            {/* Show sycophancy override warning if penalty was large */}
                            {result.ensemble_health?.asc_penalty_multiplier < 0.75 && (
                                <div style={{
                                    fontSize: '0.72rem', color: 'var(--accent-red)',
                                    background: 'var(--accent-red-glow)',
                                    padding: '4px 10px', borderRadius: '6px',
                                    fontWeight: 600, letterSpacing: '0.3px',
                                    marginTop: '0.5rem'
                                }}>
                                    ⚠ Sycophancy Override: ASC forced trade downgrade
                                </div>
                            )}

                            <div className="grid-3" style={{ width: '100%', textAlign: 'center', gap: '0.5rem', marginTop: '0.5rem' }}>
                                <div className="metric">
                                    <span className="metric-label">Allocation</span>
                                    <span className="metric-value small">
                                        {result.decision?.allocation_pct?.toFixed(1)}%
                                    </span>
                                </div>
                                <div className="metric">
                                    <span className="metric-label">Shares</span>
                                    <span className="metric-value small">
                                        {result.decision?.recommended_shares || 0}
                                    </span>
                                </div>
                                <div className="metric">
                                    <span className="metric-label">Capital</span>
                                    <span className="metric-value small">
                                        ${result.decision?.cash_value?.toFixed(0)}
                                    </span>
                                </div>
                            </div>
                        </div>

                        <div className="card">
                            <div className="card-header"><Brain /> Executive Summary (Groq LLM)</div>
                            <p className="summary-text">{result.executive_summary || 'No summary generated.'}</p>
                        </div>
                    </div>

                    {/* Row 2: Core Signals */}
                    <div className="grid-4" style={{ marginBottom: '1.5rem' }}>
                        <div className="card">
                            <div className="card-header"><BarChart3 /> Technical (LSTM)</div>
                            <div className="metric">
                                <span className="metric-label">Signal Strength</span>
                                <span className="metric-value">{result.technical?.lstm_signal?.toFixed(4)}</span>
                            </div>
                            <div style={{ marginTop: '0.5rem' }}>
                                <span className="metric-label">Top SHAP Driver: </span>
                                <span className={`badge ${getDecisionBadge(result.decision?.action)}`}>
                                    {result.technical?.top_driver}
                                </span>
                            </div>
                            <div style={{ marginTop: '0.5rem' }}>
                                <span className="metric-label">Uncertainty: </span>
                                <span style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>
                                    {result.technical?.mc_std?.toFixed(4)}
                                </span>
                            </div>
                        </div>

                        <div className="card">
                            <div className="card-header"><Activity /> Sentiment (FinBERT)</div>
                            <div className="metric">
                                <span className="metric-label">News Score</span>
                                <span className="metric-value" style={{
                                    color: (result.sentiment?.score || 0) > 0 ? 'var(--accent-green)'
                                        : (result.sentiment?.score || 0) < 0 ? 'var(--accent-red)'
                                            : 'var(--text-primary)',
                                }}>
                                    {result.sentiment?.score?.toFixed(4)}
                                </span>
                            </div>
                        </div>

                        <div className="card">
                            <div className="card-header"><Layers /> Regime (HMM)</div>
                            <div className="metric">
                                <span className="metric-label">Market State</span>
                                <span className="metric-value small">
                                    <span className={`badge ${result.regime?.label === 'Bull' ? 'badge-buy' :
                                        result.regime?.label === 'Bear' ? 'badge-sell' : 'badge-hold'
                                        }`}>
                                        {result.regime?.label}
                                    </span>
                                </span>
                            </div>
                            <div style={{ marginTop: '0.5rem' }}>
                                <span className="metric-label">Volatility: </span>
                                <span style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>
                                    {result.regime?.volatility?.toFixed(4)}
                                </span>
                            </div>
                        </div>

                        <div className="card">
                            <div className="card-header"><Cpu /> Fusion Engine</div>
                            <div className="metric">
                                <span className="metric-label">Confidence</span>
                                <span className="metric-value">
                                    {result.fusion?.confidence?.toFixed(4)}
                                </span>
                            </div>
                            <div className="progress-bar-container" style={{ marginTop: '0.5rem' }}>
                                <div className="progress-bar-track">
                                    <div
                                        className="progress-bar-fill"
                                        style={{
                                            width: `${(result.fusion?.confidence || 0) * 100}%`,
                                            background: 'var(--gradient-hero)',
                                        }}
                                    />
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Row 3: Conflict + Heatmap + Red Team + Trust */}
                    <div className="grid-4" style={{ marginBottom: '1.5rem' }}>
                        <div className="card">
                            <div className="card-header"><Shield /> Arbitrator</div>
                            <span className={`badge ${result.conflict?.detected ? 'badge-sell' : 'badge-harmony'}`}>
                                {result.conflict?.detected ? 'CONFLICT' : 'NO CONFLICT'}
                            </span>
                            <p style={{
                                marginTop: '0.5rem', fontSize: '0.75rem',
                                color: 'var(--text-muted)', lineHeight: 1.5,
                            }}>
                                {result.conflict?.ruling}
                            </p>
                        </div>

                        <div className="card">
                            <div className="card-header"><AlertTriangle /> Boardroom Tension</div>
                            <div className="metric">
                                <span className="metric-label">GDI Score</span>
                                <span className="metric-value">
                                    {result.disagreement?.gdi?.toFixed(1)}%
                                </span>
                            </div>
                            <div style={{ marginTop: '0.5rem', display: 'flex', gap: '8px', alignItems: 'center' }}>
                                <span className={`badge ${getTensionBadge(result.disagreement?.tension)}`}>
                                    {result.disagreement?.tension}
                                </span>
                                {result.disagreement?.kelly_penalty < 1 && (
                                    <span style={{ fontSize: '0.7rem', color: 'var(--accent-red)' }}>
                                        Kelly Cut: {((1 - result.disagreement.kelly_penalty) * 100).toFixed(0)}%
                                    </span>
                                )}
                            </div>
                        </div>

                        <div className="card">
                            <div className="card-header"><Shield /> Red Team</div>
                            <span className={`badge ${result.red_team?.passed ? 'badge-buy' : 'badge-sell'}`}>
                                {result.red_team?.passed ? 'PASSED' : 'VETOED'}
                            </span>
                            <p style={{
                                marginTop: '0.5rem', fontSize: '0.75rem',
                                color: 'var(--text-muted)',
                            }}>
                                Crash delta: {result.red_team?.delta?.toFixed(4)}
                            </p>
                        </div>

                        <div className="card">
                            <div className="card-header"><Brain /> Trust Scores</div>
                            {result.trust_scores && Object.entries(result.trust_scores).map(([k, v]) => (
                                <div key={k} className="progress-bar-container">
                                    <div className="progress-bar-label">
                                        <span>{k}</span>
                                        <span>{Number(v).toFixed(3)}</span>
                                    </div>
                                    <div className="progress-bar-track">
                                        <div
                                            className="progress-bar-fill"
                                            style={{
                                                width: `${Math.min((v / 1.5) * 100, 100)}%`,
                                                background: v > 1.05 ? 'var(--accent-green)'
                                                    : v < 0.95 ? 'var(--accent-red)'
                                                        : 'var(--accent-blue)',
                                            }}
                                        />
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* Row 4: Phase 26 Ensemble Health Panel */}
                    {result.ensemble_health && (
                        <motion.div
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 0.2 }}
                            style={{ marginBottom: '1.5rem' }}
                        >
                            <EnsembleHealthPanel data={result.ensemble_health} />
                        </motion.div>
                    )}

                </motion.div>
            )}
        </div>
    );
}