class RiskEngine:
    def __init__(self, default_account_size=10000, max_risk_per_trade=0.20):
        """
        Calculates optimal position size using Fractional Kelly Criterion.

        Args:
            default_account_size : Total portfolio capital (e.g. $10,000)
            max_risk_per_trade   : Hard cap — never risk more than this fraction
                                   of the account (default 20%)
        """
        self.account_size = default_account_size
        self.max_risk = max_risk_per_trade

    def calculate_position_size(
        self,
        confidence_score,
        volatility,
        disagreement_penalty=1.0,
        regime="Sideways",
    ):
        """
        Calculates the optimal % of portfolio to invest.

        FIX v2 — Regime-aware odds ratio (b):
          • Bull  → b = 2.5  (trend-following — reward outpaces risk)
          • Bear  → b = 1.5  (counter-trend — tighter risk/reward)
          • Other → b = 2.0  (neutral baseline)

        This change makes the Kelly formula more sensitive to market context,
        allowing genuine BUY signals to pass the positive-Kelly threshold in
        Bull regimes while staying conservative in Bear markets.

        Args:
            confidence_score     : AI fusion confidence (0.0 → 1.0)
            volatility           : Market volatility (daily std dev)
            disagreement_penalty : Phase 16 GDI multiplier (0.25 → 1.0).
                                   Applied after Half-Kelly and vol scaling.
                                   Default 1.0 = no penalty.
            regime               : Market regime string passed from master_system.
                                   "Bull", "Bear", or "Sideways".

        Returns:
            allocation_pct  (float) : % of portfolio to invest (0.0 → max_risk)
            kelly_fraction  (float) : Raw Kelly number (for debugging / logging)
        """
        p = confidence_score
        q = 1.0 - p

        # Regime-aware odds ratio
        regime_lower = str(regime).strip().lower()
        if regime_lower == "bull":
            b = 2.5   # Trend-following environment: bigger upside
        elif regime_lower == "bear":
            b = 1.5   # Counter-trend: tighter reward relative to risk
        else:
            b = 2.0   # Sideways / unknown: neutral baseline

        # Kelly formula: f* = p - (q / b)
        kelly_fraction = p - (q / b)

        # Negative Kelly → Expected Value is negative → DO NOT TRADE
        if kelly_fraction <= 0:
            return 0.0, kelly_fraction

        # Half-Kelly: professional standard — halves volatility drag
        safe_kelly = kelly_fraction * 0.5

        # Volatility scaling: high-vol markets get a further 50% cut
        if volatility > 0.02:
            safe_kelly *= 0.5

        # Phase 16: Boardroom-tension penalty
        safe_kelly *= disagreement_penalty

        # Hard caps
        final_allocation = max(0.0, min(safe_kelly, self.max_risk))

        return final_allocation, kelly_fraction

    def get_shares_amount(self, stock_price, allocation_pct):
        """
        Converts % allocation to number of shares and cash value.

        Args:
            stock_price    : Current price per share
            allocation_pct : Fraction of portfolio to invest (0.0 → 1.0)

        Returns:
            num_shares  (int)   : Whole shares to purchase
            cash_value  (float) : Dollar amount to invest
        """
        if allocation_pct <= 0 or stock_price <= 0:
            return 0, 0.0

        capital_to_invest = self.account_size * allocation_pct
        num_shares = int(capital_to_invest // stock_price)

        return num_shares, capital_to_invest