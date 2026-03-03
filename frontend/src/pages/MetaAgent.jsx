import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Brain, RefreshCw, Gauge, TrendingUp, TrendingDown, Minus } from 'lucide-react';
import { getTrustScores, runEvaluation } from '../services/api';

export default function MetaAgent() {
    const [scores, setScores] = useState(null);
    const [loading, setLoading] = useState(true);
    const [evaluating, setEvaluating] = useState(false);
    const [evalResult, setEvalResult] = useState(null);
    const [error, setError] = useState('');

    const fetchScores = async () => {
        try {
            const data = await getTrustScores();
            setScores(data);
            setError('');
        } catch (err) {
            setError('Failed to load trust scores');
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => { fetchScores(); }, []);

    const handleEvaluate = async () => {
        setEvaluating(true);
        setEvalResult(null);
        try {
            const data = await runEvaluation();
            setEvalResult(data);
            // Refresh trust scores after evaluation
            fetchScores();
        } catch (err) {
            setError(err.response?.data?.error || 'Evaluation failed');
        } finally {
            setEvaluating(false);
        }
    };

    const agents = ['technical', 'sentiment', 'regime'];

    const getStatusBadge = (status) => {
        const s = (status || '').toUpperCase();
        if (s === 'BOOSTED') return 'badge-boosted';
        if (s === 'PENALIZED') return 'badge-penalized';
        return 'badge-normal';
    };

    const getBarColor = (val) => {
        if (val > 1.05) return 'var(--accent-green)';
        if (val < 0.95) return 'var(--accent-red)';
        return 'var(--accent-blue)';
    };

    const getIcon = (status) => {
        const s = (status || '').toUpperCase();
        if (s === 'BOOSTED') return <TrendingUp size={14} />;
        if (s === 'PENALIZED') return <TrendingDown size={14} />;
        return <Minus size={14} />;
    };

    return (
        <div className="page-container">
            <h1 className="page-title">Meta-Agent Control Center</h1>
            <p className="page-subtitle">
                Phase 14 — Self-correcting trust multipliers that evolve based on agent accuracy
            </p>

            {loading ? (
                <div className="loading-overlay">
                    <div className="spinner" />
                    <p className="loading-text">Loading trust scores...</p>
                </div>
            ) : error && !scores ? (
                <div className="card" style={{ borderColor: 'var(--accent-red)' }}>
                    <p style={{ color: 'var(--accent-red)' }}>{error}</p>
                </div>
            ) : scores && (
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                    {/* Trust Score Cards */}
                    <div className="grid-3" style={{ marginBottom: '2rem' }}>
                        {agents.map((agent) => {
                            const val = scores[agent] || 1.0;
                            const status = scores[`${agent}_status`] || 'NORMAL';
                            const pct = ((val - 0.5) / 1.0) * 100; // Map 0.5-1.5 → 0-100%

                            return (
                                <div className="card" key={agent}>
                                    <div className="card-header">
                                        <Gauge size={16} />
                                        {agent.charAt(0).toUpperCase() + agent.slice(1)} Agent
                                    </div>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '1rem' }}>
                                        <span className="metric-value">{val.toFixed(3)}</span>
                                        <span className={`badge ${getStatusBadge(status)}`}>
                                            {getIcon(status)} {status}
                                        </span>
                                    </div>
                                    <div className="progress-bar-track" style={{ height: '12px' }}>
                                        <div
                                            className="progress-bar-fill"
                                            style={{
                                                width: `${Math.max(0, Math.min(100, pct))}%`,
                                                background: getBarColor(val),
                                            }}
                                        />
                                    </div>
                                    <div style={{
                                        display: 'flex', justifyContent: 'space-between',
                                        marginTop: '4px', fontSize: '0.65rem', color: 'var(--text-muted)'
                                    }}>
                                        <span>0.50 (Floor)</span>
                                        <span>1.00</span>
                                        <span>1.50 (Ceiling)</span>
                                    </div>
                                </div>
                            );
                        })}
                    </div>

                    {/* Meta Info */}
                    <div className="grid-2" style={{ marginBottom: '2rem' }}>
                        <div className="card">
                            <div className="card-header"><Brain /> Agent Status</div>
                            <div className="metric" style={{ marginBottom: '0.5rem' }}>
                                <span className="metric-label">Last Updated</span>
                                <span style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
                                    {scores.last_updated || 'Never'}
                                </span>
                            </div>
                            <div className="metric">
                                <span className="metric-label">Total Evaluations</span>
                                <span style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
                                    {scores.evaluation_count || 0} decisions graded
                                </span>
                            </div>
                        </div>

                        {/* Evaluate Button */}
                        <div className="card" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '1rem' }}>
                            <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', textAlign: 'center' }}>
                                Run the T+5 Hindsight Evaluator to grade past decisions and update trust scores
                            </p>
                            <button
                                className="btn btn-primary"
                                onClick={handleEvaluate}
                                disabled={evaluating}
                                style={{ minWidth: '240px' }}
                            >
                                <RefreshCw size={18} className={evaluating ? 'spin' : ''} />
                                {evaluating ? 'Evaluating...' : 'Run Hindsight Evaluation'}
                            </button>
                        </div>
                    </div>

                    {/* Evaluation Result */}
                    {evalResult && (
                        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
                            <div className="card">
                                <div className="card-header"><Brain /> Evaluation Result</div>
                                <p style={{ color: 'var(--accent-green)', marginBottom: '1rem', fontWeight: 600 }}>
                                    {evalResult.message}
                                </p>
                                <div className="grid-2">
                                    <div>
                                        <span className="metric-label">Trust Before</span>
                                        {Object.entries(evalResult.trust_before || {}).map(([k, v]) => (
                                            <p key={k} style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                                                {k}: {Number(v).toFixed(3)}
                                            </p>
                                        ))}
                                    </div>
                                    <div>
                                        <span className="metric-label">Trust After</span>
                                        {Object.entries(evalResult.trust_after || {}).map(([k, v]) => (
                                            <p key={k} style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                                                {k}: {Number(v).toFixed(3)}
                                            </p>
                                        ))}
                                    </div>
                                </div>
                                <p style={{ marginTop: '1rem', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                                    Total evaluated: {evalResult.total_evaluated} decisions
                                </p>
                            </div>
                        </motion.div>
                    )}
                </motion.div>
            )}
        </div>
    );
}
