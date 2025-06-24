"""
Intelligent Position Sizing Algorithms
Advanced position sizing based on signal confidence, risk metrics, and portfolio optimization
"""

import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import math

logger = logging.getLogger(__name__)

class SizingMethod(Enum):
    """Position sizing method enumeration"""
    FIXED_DOLLAR = "fixed_dollar"
    FIXED_PERCENT = "fixed_percent"
    KELLY_CRITERION = "kelly_criterion"
    RISK_PARITY = "risk_parity"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    OPTIMAL_F = "optimal_f"

class RiskLevel(Enum):
    """Risk level enumeration"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    SPECULATIVE = "speculative"

@dataclass
class SignalInput:
    """Signal input for position sizing"""
    signal_id: str
    symbol: str
    strategy_id: str
    confidence: float  # 0.0 to 1.0
    expected_return: float  # Expected return percentage
    expected_volatility: float  # Expected volatility
    time_horizon: int  # Days
    signal_strength: float  # 0.0 to 1.0
    
    # Risk metrics
    max_loss_percent: float = 0.02  # Maximum loss as % of portfolio
    stop_loss_percent: float = 0.05  # Stop loss as % of position
    
    # Market context
    market_regime: str = "normal"  # normal, volatile, trending, ranging
    sector: str = "technology"
    market_cap: str = "large"  # large, mid, small

@dataclass
class PortfolioContext:
    """Portfolio context for position sizing"""
    total_value: float
    available_cash: float
    buying_power: float
    current_positions: int
    max_positions: int
    
    # Risk metrics
    current_var: float  # Current portfolio VaR
    max_var: float  # Maximum allowed VaR
    current_beta: float
    target_beta: float
    
    # Diversification
    sector_exposure: Dict[str, float]  # Sector -> exposure %
    symbol_exposure: Dict[str, float]  # Symbol -> exposure %
    strategy_exposure: Dict[str, float]  # Strategy -> exposure %
    
    # Risk limits
    max_single_position: float = 0.10  # 10% max per position
    max_sector_exposure: float = 0.30  # 30% max per sector
    max_strategy_exposure: float = 0.25  # 25% max per strategy

@dataclass
class SizingResult:
    """Position sizing result"""
    signal_id: str
    symbol: str
    recommended_size: float  # Dollar amount
    recommended_shares: int  # Number of shares
    position_percent: float  # % of portfolio
    
    # Risk metrics
    position_var: float
    expected_return: float
    risk_reward_ratio: float
    
    # Sizing details
    sizing_method: SizingMethod
    confidence_adjustment: float
    risk_adjustment: float
    diversification_adjustment: float
    
    # Validation
    is_valid: bool = True
    rejection_reason: str = ""
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []

class PositionSizer:
    """
    Intelligent position sizing engine
    Calculates optimal position sizes based on multiple factors
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        
        # Risk parameters
        self.max_portfolio_risk = self.config.get('max_portfolio_risk', 0.02)
        self.max_position_risk = self.config.get('max_position_risk', 0.005)
        self.min_confidence = self.config.get('min_confidence', 0.6)
        self.max_leverage = self.config.get('max_leverage', 1.0)
        
        # Sizing parameters
        self.base_position_size = self.config.get('base_position_size', 0.02)  # 2% of portfolio
        self.confidence_multiplier = self.config.get('confidence_multiplier', 2.0)
        self.volatility_lookback = self.config.get('volatility_lookback', 20)
        
        logger.info("Position Sizer initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for position sizing"""
        return {
            'max_portfolio_risk': 0.02,  # 2% max portfolio risk
            'max_position_risk': 0.005,  # 0.5% max position risk
            'min_confidence': 0.6,  # Minimum signal confidence
            'max_leverage': 1.0,  # No leverage by default
            'base_position_size': 0.02,  # 2% base position size
            'confidence_multiplier': 2.0,  # Confidence scaling factor
            'volatility_lookback': 20,  # Days for volatility calculation
            'kelly_fraction': 0.25,  # Kelly criterion fraction
            'risk_free_rate': 0.02,  # Risk-free rate for Sharpe calculation
        }
    
    def calculate_position_size(self, signal: SignalInput, portfolio: PortfolioContext, 
                              method: SizingMethod = SizingMethod.CONFIDENCE_WEIGHTED) -> SizingResult:
        """
        Calculate optimal position size for a signal
        
        Args:
            signal: Signal input data
            portfolio: Portfolio context
            method: Sizing method to use
            
        Returns:
            Position sizing result
        """
        try:
            # Validate inputs
            if not self._validate_inputs(signal, portfolio):
                return SizingResult(
                    signal_id=signal.signal_id,
                    symbol=signal.symbol,
                    recommended_size=0.0,
                    recommended_shares=0,
                    position_percent=0.0,
                    position_var=0.0,
                    expected_return=0.0,
                    risk_reward_ratio=0.0,
                    sizing_method=method,
                    confidence_adjustment=0.0,
                    risk_adjustment=0.0,
                    diversification_adjustment=0.0,
                    is_valid=False,
                    rejection_reason="Input validation failed"
                )
            
            # Calculate base size using selected method
            base_size = self._calculate_base_size(signal, portfolio, method)
            
            # Apply adjustments
            confidence_adj = self._calculate_confidence_adjustment(signal)
            risk_adj = self._calculate_risk_adjustment(signal, portfolio)
            diversification_adj = self._calculate_diversification_adjustment(signal, portfolio)
            
            # Calculate final size
            adjusted_size = base_size * confidence_adj * risk_adj * diversification_adj
            
            # Apply portfolio constraints
            final_size = self._apply_constraints(adjusted_size, signal, portfolio)
            
            # Calculate shares and metrics
            shares = int(final_size / self._get_current_price(signal.symbol))
            position_percent = final_size / portfolio.total_value
            
            # Calculate risk metrics
            position_var = self._calculate_position_var(signal, final_size)
            risk_reward_ratio = abs(signal.expected_return / signal.max_loss_percent) if signal.max_loss_percent > 0 else 0
            
            # Validate result
            is_valid, rejection_reason, warnings = self._validate_result(
                final_size, signal, portfolio
            )
            
            return SizingResult(
                signal_id=signal.signal_id,
                symbol=signal.symbol,
                recommended_size=final_size,
                recommended_shares=shares,
                position_percent=position_percent,
                position_var=position_var,
                expected_return=signal.expected_return,
                risk_reward_ratio=risk_reward_ratio,
                sizing_method=method,
                confidence_adjustment=confidence_adj,
                risk_adjustment=risk_adj,
                diversification_adjustment=diversification_adj,
                is_valid=is_valid,
                rejection_reason=rejection_reason,
                warnings=warnings
            )
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return SizingResult(
                signal_id=signal.signal_id,
                symbol=signal.symbol,
                recommended_size=0.0,
                recommended_shares=0,
                position_percent=0.0,
                position_var=0.0,
                expected_return=0.0,
                risk_reward_ratio=0.0,
                sizing_method=method,
                confidence_adjustment=0.0,
                risk_adjustment=0.0,
                diversification_adjustment=0.0,
                is_valid=False,
                rejection_reason=f"Calculation error: {str(e)}"
            )
    
    def _calculate_base_size(self, signal: SignalInput, portfolio: PortfolioContext, 
                           method: SizingMethod) -> float:
        """Calculate base position size using specified method"""
        try:
            if method == SizingMethod.FIXED_DOLLAR:
                return self._fixed_dollar_sizing(signal, portfolio)
            elif method == SizingMethod.FIXED_PERCENT:
                return self._fixed_percent_sizing(signal, portfolio)
            elif method == SizingMethod.KELLY_CRITERION:
                return self._kelly_criterion_sizing(signal, portfolio)
            elif method == SizingMethod.RISK_PARITY:
                return self._risk_parity_sizing(signal, portfolio)
            elif method == SizingMethod.VOLATILITY_ADJUSTED:
                return self._volatility_adjusted_sizing(signal, portfolio)
            elif method == SizingMethod.CONFIDENCE_WEIGHTED:
                return self._confidence_weighted_sizing(signal, portfolio)
            elif method == SizingMethod.OPTIMAL_F:
                return self._optimal_f_sizing(signal, portfolio)
            else:
                logger.warning(f"Unknown sizing method {method}, using fixed percent")
                return self._fixed_percent_sizing(signal, portfolio)
                
        except Exception as e:
            logger.error(f"Error in base size calculation: {e}")
            return portfolio.total_value * self.base_position_size
    
    def _fixed_dollar_sizing(self, signal: SignalInput, portfolio: PortfolioContext) -> float:
        """Fixed dollar amount sizing"""
        base_amount = self.config.get('fixed_dollar_amount', 10000)
        return min(base_amount, portfolio.available_cash * 0.5)
    
    def _fixed_percent_sizing(self, signal: SignalInput, portfolio: PortfolioContext) -> float:
        """Fixed percentage of portfolio sizing"""
        return portfolio.total_value * self.base_position_size
    
    def _kelly_criterion_sizing(self, signal: SignalInput, portfolio: PortfolioContext) -> float:
        """Kelly Criterion optimal sizing"""
        try:
            # Kelly formula: f = (bp - q) / b
            # where b = odds, p = win probability, q = loss probability
            
            win_prob = signal.confidence
            loss_prob = 1 - win_prob
            
            # Estimate odds from expected return and stop loss
            if signal.stop_loss_percent > 0:
                odds = abs(signal.expected_return / signal.stop_loss_percent)
            else:
                odds = 2.0  # Default 2:1 odds
            
            # Kelly fraction
            kelly_f = (odds * win_prob - loss_prob) / odds
            
            # Apply Kelly fraction limit (typically 25% of Kelly)
            kelly_fraction = self.config.get('kelly_fraction', 0.25)
            kelly_f = max(0, min(kelly_f * kelly_fraction, 0.1))  # Cap at 10%
            
            return portfolio.total_value * kelly_f
            
        except Exception as e:
            logger.error(f"Error in Kelly criterion calculation: {e}")
            return self._fixed_percent_sizing(signal, portfolio)
    
    def _risk_parity_sizing(self, signal: SignalInput, portfolio: PortfolioContext) -> float:
        """Risk parity sizing based on volatility"""
        try:
            # Target risk contribution
            target_risk = self.max_position_risk * portfolio.total_value
            
            # Position size = target_risk / (volatility * price)
            price = self._get_current_price(signal.symbol)
            volatility = signal.expected_volatility
            
            if volatility > 0 and price > 0:
                position_size = target_risk / (volatility * price)
                return min(position_size * price, portfolio.total_value * 0.1)
            else:
                return self._fixed_percent_sizing(signal, portfolio)
                
        except Exception as e:
            logger.error(f"Error in risk parity calculation: {e}")
            return self._fixed_percent_sizing(signal, portfolio)
    
    def _volatility_adjusted_sizing(self, signal: SignalInput, portfolio: PortfolioContext) -> float:
        """Volatility-adjusted sizing"""
        try:
            # Base size adjusted by inverse volatility
            base_size = portfolio.total_value * self.base_position_size
            
            # Normalize volatility (assume 20% as baseline)
            baseline_vol = 0.20
            vol_adjustment = baseline_vol / max(signal.expected_volatility, 0.05)
            
            # Cap adjustment between 0.5x and 2x
            vol_adjustment = max(0.5, min(vol_adjustment, 2.0))
            
            return base_size * vol_adjustment
            
        except Exception as e:
            logger.error(f"Error in volatility adjustment: {e}")
            return self._fixed_percent_sizing(signal, portfolio)
    
    def _confidence_weighted_sizing(self, signal: SignalInput, portfolio: PortfolioContext) -> float:
        """Confidence-weighted sizing (default method)"""
        try:
            # Base size from portfolio
            base_size = portfolio.total_value * self.base_position_size
            
            # Confidence scaling (0.6 confidence = 0.5x, 0.9 confidence = 2x)
            confidence_scale = (signal.confidence - 0.5) * self.confidence_multiplier
            confidence_scale = max(0.5, min(confidence_scale, 2.0))
            
            # Signal strength scaling
            strength_scale = 0.5 + (signal.signal_strength * 0.5)
            
            # Expected return scaling
            return_scale = 1.0 + (signal.expected_return * 2.0)  # 5% return = 1.1x
            return_scale = max(0.5, min(return_scale, 1.5))
            
            return base_size * confidence_scale * strength_scale * return_scale
            
        except Exception as e:
            logger.error(f"Error in confidence weighting: {e}")
            return self._fixed_percent_sizing(signal, portfolio)
    
    def _optimal_f_sizing(self, signal: SignalInput, portfolio: PortfolioContext) -> float:
        """Optimal F sizing method"""
        try:
            # Simplified Optimal F calculation
            # Requires historical trade data, using approximation
            
            win_rate = signal.confidence
            avg_win = signal.expected_return
            avg_loss = signal.max_loss_percent
            
            if avg_loss > 0:
                # Optimal F approximation
                optimal_f = (win_rate * avg_win - (1 - win_rate) * avg_loss) / (avg_win * avg_loss)
                optimal_f = max(0, min(optimal_f, 0.2))  # Cap at 20%
                
                return portfolio.total_value * optimal_f
            else:
                return self._fixed_percent_sizing(signal, portfolio)
                
        except Exception as e:
            logger.error(f"Error in Optimal F calculation: {e}")
            return self._fixed_percent_sizing(signal, portfolio)
    
    def _calculate_confidence_adjustment(self, signal: SignalInput) -> float:
        """Calculate confidence-based adjustment factor"""
        try:
            # Linear scaling from confidence
            # 0.6 confidence = 0.7x, 0.8 confidence = 1.0x, 0.9 confidence = 1.3x
            min_conf = self.min_confidence
            max_conf = 0.95
            
            if signal.confidence < min_conf:
                return 0.5  # Reduce size for low confidence
            
            # Normalize confidence to 0-1 range
            norm_conf = (signal.confidence - min_conf) / (max_conf - min_conf)
            norm_conf = max(0, min(norm_conf, 1))
            
            # Scale from 0.7x to 1.5x
            return 0.7 + (norm_conf * 0.8)
            
        except Exception as e:
            logger.error(f"Error calculating confidence adjustment: {e}")
            return 1.0
    
    def _calculate_risk_adjustment(self, signal: SignalInput, portfolio: PortfolioContext) -> float:
        """Calculate risk-based adjustment factor"""
        try:
            # Portfolio risk adjustment
            risk_utilization = portfolio.current_var / portfolio.max_var if portfolio.max_var > 0 else 0
            
            if risk_utilization > 0.8:
                risk_adj = 0.5  # Reduce size when portfolio risk is high
            elif risk_utilization > 0.6:
                risk_adj = 0.75
            else:
                risk_adj = 1.0
            
            # Volatility adjustment
            if signal.expected_volatility > 0.3:  # High volatility
                vol_adj = 0.7
            elif signal.expected_volatility > 0.2:  # Medium volatility
                vol_adj = 0.85
            else:
                vol_adj = 1.0
            
            # Market regime adjustment
            regime_adj = 1.0
            if signal.market_regime == "volatile":
                regime_adj = 0.8
            elif signal.market_regime == "trending":
                regime_adj = 1.1
            elif signal.market_regime == "ranging":
                regime_adj = 0.9
            
            return risk_adj * vol_adj * regime_adj
            
        except Exception as e:
            logger.error(f"Error calculating risk adjustment: {e}")
            return 1.0
    
    def _calculate_diversification_adjustment(self, signal: SignalInput, portfolio: PortfolioContext) -> float:
        """Calculate diversification-based adjustment factor"""
        try:
            # Symbol concentration adjustment
            current_symbol_exposure = portfolio.symbol_exposure.get(signal.symbol, 0)
            if current_symbol_exposure > 0.05:  # Already 5%+ exposure
                symbol_adj = 0.5
            elif current_symbol_exposure > 0.03:  # 3%+ exposure
                symbol_adj = 0.75
            else:
                symbol_adj = 1.0
            
            # Sector concentration adjustment
            current_sector_exposure = portfolio.sector_exposure.get(signal.sector, 0)
            if current_sector_exposure > 0.2:  # Already 20%+ sector exposure
                sector_adj = 0.6
            elif current_sector_exposure > 0.15:  # 15%+ sector exposure
                sector_adj = 0.8
            else:
                sector_adj = 1.0
            
            # Strategy concentration adjustment
            current_strategy_exposure = portfolio.strategy_exposure.get(signal.strategy_id, 0)
            if current_strategy_exposure > 0.15:  # Already 15%+ strategy exposure
                strategy_adj = 0.7
            elif current_strategy_exposure > 0.1:  # 10%+ strategy exposure
                strategy_adj = 0.85
            else:
                strategy_adj = 1.0
            
            # Position count adjustment
            if portfolio.current_positions >= portfolio.max_positions * 0.9:
                position_adj = 0.5  # Near position limit
            elif portfolio.current_positions >= portfolio.max_positions * 0.7:
                position_adj = 0.75
            else:
                position_adj = 1.0
            
            return symbol_adj * sector_adj * strategy_adj * position_adj
            
        except Exception as e:
            logger.error(f"Error calculating diversification adjustment: {e}")
            return 1.0
    
    def _apply_constraints(self, size: float, signal: SignalInput, portfolio: PortfolioContext) -> float:
        """Apply portfolio and risk constraints"""
        try:
            # Maximum position size constraint
            max_position_value = portfolio.total_value * portfolio.max_single_position
            size = min(size, max_position_value)
            
            # Available cash constraint
            size = min(size, portfolio.available_cash)
            
            # Maximum risk constraint
            max_risk_value = portfolio.total_value * self.max_position_risk
            if signal.max_loss_percent > 0:
                max_size_by_risk = max_risk_value / signal.max_loss_percent
                size = min(size, max_size_by_risk)
            
            # Minimum position size (avoid tiny positions)
            min_position = portfolio.total_value * 0.001  # 0.1% minimum
            if size < min_position:
                size = 0
            
            return max(0, size)
            
        except Exception as e:
            logger.error(f"Error applying constraints: {e}")
            return 0
    
    def _calculate_position_var(self, signal: SignalInput, position_size: float) -> float:
        """Calculate position-level Value at Risk"""
        try:
            # 95% VaR approximation: 1.65 * volatility * position_size
            return 1.65 * signal.expected_volatility * position_size
            
        except Exception as e:
            logger.error(f"Error calculating position VaR: {e}")
            return 0
    
    def _validate_inputs(self, signal: SignalInput, portfolio: PortfolioContext) -> bool:
        """Validate input parameters"""
        try:
            # Signal validation
            if signal.confidence < 0 or signal.confidence > 1:
                logger.error(f"Invalid confidence: {signal.confidence}")
                return False
            
            if signal.confidence < self.min_confidence:
                logger.warning(f"Signal confidence {signal.confidence} below minimum {self.min_confidence}")
                return False
            
            if signal.expected_volatility < 0:
                logger.error(f"Invalid volatility: {signal.expected_volatility}")
                return False
            
            # Portfolio validation
            if portfolio.total_value <= 0:
                logger.error(f"Invalid portfolio value: {portfolio.total_value}")
                return False
            
            if portfolio.available_cash < 0:
                logger.error(f"Invalid available cash: {portfolio.available_cash}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating inputs: {e}")
            return False
    
    def _validate_result(self, size: float, signal: SignalInput, portfolio: PortfolioContext) -> Tuple[bool, str, List[str]]:
        """Validate sizing result"""
        warnings = []
        
        try:
            # Check if size is valid
            if size <= 0:
                return False, "Position size is zero or negative", warnings
            
            # Check portfolio constraints
            position_percent = size / portfolio.total_value
            if position_percent > portfolio.max_single_position:
                return False, f"Position size {position_percent:.1%} exceeds maximum {portfolio.max_single_position:.1%}", warnings
            
            # Check available cash
            if size > portfolio.available_cash:
                return False, f"Position size ${size:,.0f} exceeds available cash ${portfolio.available_cash:,.0f}", warnings
            
            # Add warnings
            if position_percent > 0.05:
                warnings.append(f"Large position size: {position_percent:.1%} of portfolio")
            
            if signal.expected_volatility > 0.3:
                warnings.append(f"High volatility signal: {signal.expected_volatility:.1%}")
            
            if signal.confidence < 0.7:
                warnings.append(f"Low confidence signal: {signal.confidence:.1%}")
            
            return True, "", warnings
            
        except Exception as e:
            logger.error(f"Error validating result: {e}")
            return False, f"Validation error: {str(e)}", warnings
    
    def _get_current_price(self, symbol: str) -> float:
        """Get current price for symbol (mock implementation)"""
        # In production, this would fetch from market data service
        mock_prices = {
            "AAPL": 150.0,
            "MSFT": 300.0,
            "GOOGL": 2500.0,
            "AMZN": 3000.0,
            "TSLA": 200.0,
            "NVDA": 400.0,
            "META": 250.0,
            "SPY": 400.0,
            "QQQ": 350.0
        }
        return mock_prices.get(symbol, 100.0)
    
    def calculate_multiple_positions(self, signals: List[SignalInput], portfolio: PortfolioContext,
                                   method: SizingMethod = SizingMethod.CONFIDENCE_WEIGHTED) -> List[SizingResult]:
        """
        Calculate position sizes for multiple signals simultaneously
        Considers portfolio-level constraints and optimization
        """
        try:
            results = []
            remaining_cash = portfolio.available_cash
            temp_portfolio = portfolio
            
            # Sort signals by confidence * expected_return (priority)
            sorted_signals = sorted(signals, 
                                  key=lambda s: s.confidence * s.expected_return, 
                                  reverse=True)
            
            for signal in sorted_signals:
                # Update portfolio context with previous allocations
                temp_portfolio.available_cash = remaining_cash
                
                # Calculate position size
                result = self.calculate_position_size(signal, temp_portfolio, method)
                
                if result.is_valid and result.recommended_size > 0:
                    # Update remaining cash
                    remaining_cash -= result.recommended_size
                    
                    # Update portfolio exposures for next calculation
                    temp_portfolio.symbol_exposure[signal.symbol] = (
                        temp_portfolio.symbol_exposure.get(signal.symbol, 0) + 
                        result.position_percent
                    )
                    temp_portfolio.sector_exposure[signal.sector] = (
                        temp_portfolio.sector_exposure.get(signal.sector, 0) + 
                        result.position_percent
                    )
                    temp_portfolio.strategy_exposure[signal.strategy_id] = (
                        temp_portfolio.strategy_exposure.get(signal.strategy_id, 0) + 
                        result.position_percent
                    )
                    temp_portfolio.current_positions += 1
                
                results.append(result)
                
                # Stop if no more cash available
                if remaining_cash < portfolio.total_value * 0.01:  # Less than 1% cash remaining
                    break
            
            return results
            
        except Exception as e:
            logger.error(f"Error calculating multiple positions: {e}")
            return []

# Utility functions
def create_signal_input(signal_id: str, symbol: str, strategy_id: str, 
                       confidence: float, expected_return: float, **kwargs) -> SignalInput:
    """Create SignalInput with defaults"""
    return SignalInput(
        signal_id=signal_id,
        symbol=symbol,
        strategy_id=strategy_id,
        confidence=confidence,
        expected_return=expected_return,
        expected_volatility=kwargs.get('expected_volatility', 0.2),
        time_horizon=kwargs.get('time_horizon', 5),
        signal_strength=kwargs.get('signal_strength', confidence),
        max_loss_percent=kwargs.get('max_loss_percent', 0.02),
        stop_loss_percent=kwargs.get('stop_loss_percent', 0.05),
        market_regime=kwargs.get('market_regime', 'normal'),
        sector=kwargs.get('sector', 'technology'),
        market_cap=kwargs.get('market_cap', 'large')
    )

def create_portfolio_context(total_value: float, available_cash: float, **kwargs) -> PortfolioContext:
    """Create PortfolioContext with defaults"""
    return PortfolioContext(
        total_value=total_value,
        available_cash=available_cash,
        buying_power=kwargs.get('buying_power', available_cash * 2),
        current_positions=kwargs.get('current_positions', 0),
        max_positions=kwargs.get('max_positions', 20),
        current_var=kwargs.get('current_var', 0.01),
        max_var=kwargs.get('max_var', 0.02),
        current_beta=kwargs.get('current_beta', 1.0),
        target_beta=kwargs.get('target_beta', 1.0),
        sector_exposure=kwargs.get('sector_exposure', {}),
        symbol_exposure=kwargs.get('symbol_exposure', {}),
        strategy_exposure=kwargs.get('strategy_exposure', {}),
        max_single_position=kwargs.get('max_single_position', 0.10),
        max_sector_exposure=kwargs.get('max_sector_exposure', 0.30),
        max_strategy_exposure=kwargs.get('max_strategy_exposure', 0.25)
    )

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create position sizer
    sizer = PositionSizer()
    
    # Create test signal
    signal = create_signal_input(
        signal_id="test_001",
        symbol="AAPL",
        strategy_id="momentum_breakout",
        confidence=0.8,
        expected_return=0.06,
        expected_volatility=0.25,
        sector="technology"
    )
    
    # Create test portfolio
    portfolio = create_portfolio_context(
        total_value=100000,
        available_cash=50000,
        current_positions=5,
        max_positions=20
    )
    
    # Calculate position size
    result = sizer.calculate_position_size(signal, portfolio, SizingMethod.CONFIDENCE_WEIGHTED)
    
    print(f"Position Sizing Result:")
    print(f"  Symbol: {result.symbol}")
    print(f"  Recommended Size: ${result.recommended_size:,.0f}")
    print(f"  Recommended Shares: {result.recommended_shares}")
    print(f"  Position %: {result.position_percent:.2%}")
    print(f"  Risk/Reward: {result.risk_reward_ratio:.2f}")
    print(f"  Valid: {result.is_valid}")
    if result.warnings:
        print(f"  Warnings: {', '.join(result.warnings)}")
    
    # Test multiple sizing methods
    methods = [SizingMethod.FIXED_PERCENT, SizingMethod.KELLY_CRITERION, 
              SizingMethod.VOLATILITY_ADJUSTED, SizingMethod.CONFIDENCE_WEIGHTED]
    
    print(f"\nComparison of sizing methods:")
    for method in methods:
        result = sizer.calculate_position_size(signal, portfolio, method)
        print(f"  {method.value}: ${result.recommended_size:,.0f} ({result.position_percent:.2%})")
    
    print("Position sizing example completed")

