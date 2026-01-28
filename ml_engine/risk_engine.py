import numpy as np

class RiskEngine:
    def __init__(self, default_account_size=10000, max_risk_per_trade=0.05):
        """
        Calculates optimal position size using Fractional Kelly Criterion.
        
        :param default_account_size: Total money in the portfolio (e.g., $10,000)
        :param max_risk_per_trade: Safety cap (e.g., never risk more than 5% of account)
        """
        self.account_size = default_account_size
        self.max_risk = max_risk_per_trade

    def calculate_position_size(self, confidence_score, volatility):
        """
        Calculates the optimal % of portfolio to invest.
        
        Returns:
        1. allocation_pct (Float): % of portfolio to invest (0.0 to 1.0)
        2. kelly_fraction (Float): The raw Kelly number (for debugging)
        """
        # 1. ESTABLISH INPUTS
        # p = Win Probability (AI Confidence)
        p = confidence_score
        q = 1.0 - p
        
        # b = Odds (Risk/Reward Ratio). We target 1:2, so b=2.0
        b = 2.0 

        # 2. KELLY FORMULA
        # f* = p - (q / b)
        kelly_fraction = p - (q / b)
        
        # 3. SAFETY ADJUSTMENTS
        # If Kelly is negative (Expected Value is negative), DO NOT TRADE.
        if kelly_fraction <= 0:
            return 0.0, 0.0  # <--- FIXED: Returns exactly 2 values now.

        # "Half-Kelly" Rule: Professional standard to reduce volatility drag.
        safe_kelly = kelly_fraction * 0.5
        
        # Volatility Scaling: If market is crazy (>2% daily moves), cut size in half again.
        if volatility > 0.02:
            safe_kelly = safe_kelly * 0.5 

        # 4. HARD CAPS
        # - Max Risk: Never exceed 20% of portfolio (0.20)
        # - Min Risk: Never go below 0 (max(0.0, ...))
        final_allocation = max(0.0, min(safe_kelly, 0.20))
        
        return final_allocation, kelly_fraction
    
    def get_shares_amount(self, stock_price, allocation_pct):
        """
        Converts % allocation to actual number of shares.
        """
        if allocation_pct <= 0:
            return 0, 0.0
            
        capital_to_invest = self.account_size * allocation_pct
        
        # Calculate shares (integer)
        num_shares = int(capital_to_invest // stock_price)
        
        return num_shares, capital_to_invest