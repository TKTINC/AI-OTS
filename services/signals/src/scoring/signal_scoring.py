"""
Advanced Signal Scoring and Ranking System for AI Options Trading System
Sophisticated algorithms for scoring, ranking, and validating trading signals
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import math
from datetime import datetime, timezone, timedelta
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import joblib
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class SignalPriority(Enum):
    """Signal priority levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class SignalQuality(Enum):
    """Signal quality grades"""
    POOR = "D"
    FAIR = "C"
    GOOD = "B"
    EXCELLENT = "A"

@dataclass
class SignalScore:
    """Comprehensive signal scoring"""
    overall_score: float  # 0-100
    confidence_score: float  # 0-100
    quality_grade: SignalQuality
    priority: SignalPriority
    risk_score: float  # 0-100 (lower is better)
    reward_score: float  # 0-100
    timing_score: float  # 0-100
    technical_score: float  # 0-100
    fundamental_score: float  # 0-100
    market_score: float  # 0-100
    volatility_score: float  # 0-100
    liquidity_score: float  # 0-100
    
    # Detailed breakdowns
    score_components: Dict[str, float] = field(default_factory=dict)
    risk_factors: Dict[str, float] = field(default_factory=dict)
    confidence_factors: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    scoring_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    model_version: str = "1.0"
    
@dataclass
class RankedSignal:
    """Signal with ranking information"""
    signal_id: str
    symbol: str
    signal_type: str
    strategy_name: str
    score: SignalScore
    rank: int
    percentile: float
    expected_return: float
    max_risk: float
    time_horizon: int
    entry_price: float
    target_price: float
    stop_loss: float
    
    # Additional ranking factors
    market_conditions: Dict[str, Any] = field(default_factory=dict)
    correlation_risk: float = 0.0
    portfolio_impact: float = 0.0

class SignalScoringEngine:
    """Advanced signal scoring and ranking engine"""
    
    def __init__(self):
        self.scoring_models = {}
        self.feature_scalers = {}
        self.historical_performance = {}
        self.market_regime_detector = None
        
        # Scoring weights for different components
        self.scoring_weights = {
            "technical_analysis": 0.25,
            "pattern_recognition": 0.20,
            "risk_management": 0.15,
            "market_conditions": 0.15,
            "timing": 0.10,
            "liquidity": 0.08,
            "volatility": 0.07
        }
        
        # Risk factor weights
        self.risk_weights = {
            "market_risk": 0.30,
            "strategy_risk": 0.25,
            "timing_risk": 0.20,
            "liquidity_risk": 0.15,
            "correlation_risk": 0.10
        }
        
        # Initialize ML models
        self._initialize_models()
    
    def score_signal(self, signal_data: Dict[str, Any], market_data: Dict[str, Any],
                    portfolio_data: Dict[str, Any] = None) -> SignalScore:
        """Score a trading signal comprehensively"""
        try:
            # Extract signal components
            technical_score = self._score_technical_factors(signal_data, market_data)
            fundamental_score = self._score_fundamental_factors(signal_data, market_data)
            market_score = self._score_market_conditions(market_data)
            timing_score = self._score_timing_factors(signal_data, market_data)
            risk_score = self._score_risk_factors(signal_data, market_data, portfolio_data)
            reward_score = self._score_reward_potential(signal_data, market_data)
            volatility_score = self._score_volatility_factors(signal_data, market_data)
            liquidity_score = self._score_liquidity_factors(signal_data, market_data)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(signal_data, market_data)
            
            # Calculate overall score using weighted average
            component_scores = {
                "technical": technical_score,
                "fundamental": fundamental_score,
                "market": market_score,
                "timing": timing_score,
                "risk": 100 - risk_score,  # Invert risk score
                "reward": reward_score,
                "volatility": volatility_score,
                "liquidity": liquidity_score
            }
            
            overall_score = self._calculate_weighted_score(component_scores)
            
            # Determine quality grade and priority
            quality_grade = self._determine_quality_grade(overall_score, confidence_score)
            priority = self._determine_priority(overall_score, risk_score, signal_data)
            
            # Detailed risk and confidence factors
            risk_factors = self._analyze_risk_factors(signal_data, market_data, portfolio_data)
            confidence_factors = self._analyze_confidence_factors(signal_data, market_data)
            
            return SignalScore(
                overall_score=overall_score,
                confidence_score=confidence_score,
                quality_grade=quality_grade,
                priority=priority,
                risk_score=risk_score,
                reward_score=reward_score,
                timing_score=timing_score,
                technical_score=technical_score,
                fundamental_score=fundamental_score,
                market_score=market_score,
                volatility_score=volatility_score,
                liquidity_score=liquidity_score,
                score_components=component_scores,
                risk_factors=risk_factors,
                confidence_factors=confidence_factors
            )
            
        except Exception as e:
            logger.error(f"Error scoring signal: {e}")
            return self._create_default_score()
    
    def rank_signals(self, signals: List[Dict[str, Any]], market_data: Dict[str, Any],
                    portfolio_data: Dict[str, Any] = None) -> List[RankedSignal]:
        """Rank multiple signals"""
        try:
            scored_signals = []
            
            # Score all signals
            for signal in signals:
                score = self.score_signal(signal, market_data, portfolio_data)
                
                ranked_signal = RankedSignal(
                    signal_id=signal.get("signal_id", ""),
                    symbol=signal.get("symbol", ""),
                    signal_type=signal.get("signal_type", ""),
                    strategy_name=signal.get("strategy_name", ""),
                    score=score,
                    rank=0,  # Will be set after sorting
                    percentile=0.0,  # Will be set after sorting
                    expected_return=signal.get("expected_return", 0.0),
                    max_risk=signal.get("max_risk", 0.0),
                    time_horizon=signal.get("time_horizon", 0),
                    entry_price=signal.get("entry_price", 0.0),
                    target_price=signal.get("target_price", 0.0),
                    stop_loss=signal.get("stop_loss", 0.0),
                    market_conditions=market_data.copy(),
                    correlation_risk=self._calculate_correlation_risk(signal, signals),
                    portfolio_impact=self._calculate_portfolio_impact(signal, portfolio_data)
                )
                
                scored_signals.append(ranked_signal)
            
            # Sort by overall score (descending)
            scored_signals.sort(key=lambda s: s.score.overall_score, reverse=True)
            
            # Assign ranks and percentiles
            total_signals = len(scored_signals)
            for i, signal in enumerate(scored_signals):
                signal.rank = i + 1
                signal.percentile = ((total_signals - i) / total_signals) * 100
            
            # Apply portfolio-level adjustments
            ranked_signals = self._apply_portfolio_adjustments(scored_signals, portfolio_data)
            
            return ranked_signals
            
        except Exception as e:
            logger.error(f"Error ranking signals: {e}")
            return []
    
    def _score_technical_factors(self, signal_data: Dict[str, Any], market_data: Dict[str, Any]) -> float:
        """Score technical analysis factors"""
        try:
            score = 0.0
            factors = 0
            
            # Technical indicators strength
            indicators = signal_data.get("technical_factors", {})
            
            # RSI scoring
            rsi = indicators.get("rsi", 50)
            if 30 <= rsi <= 70:  # Neutral zone
                rsi_score = 60
            elif rsi < 30 or rsi > 70:  # Oversold/overbought
                rsi_score = 80
            else:  # Extreme levels
                rsi_score = 90
            
            score += rsi_score
            factors += 1
            
            # MACD scoring
            macd_histogram = indicators.get("macd_histogram", 0)
            macd_score = min(100, abs(macd_histogram) * 1000 + 50)
            score += macd_score
            factors += 1
            
            # Volume confirmation
            volume_ratio = indicators.get("volume_ratio", 1.0)
            volume_score = min(100, volume_ratio * 30 + 40)
            score += volume_score
            factors += 1
            
            # Breakout strength
            breakout_strength = indicators.get("breakout_strength", 0)
            breakout_score = min(100, breakout_strength * 1000 + 50)
            score += breakout_score
            factors += 1
            
            # Pattern strength
            pattern_strength = signal_data.get("pattern_strength", 0.5)
            pattern_score = pattern_strength * 100
            score += pattern_score
            factors += 1
            
            return score / factors if factors > 0 else 50.0
            
        except Exception as e:
            logger.error(f"Error scoring technical factors: {e}")
            return 50.0
    
    def _score_fundamental_factors(self, signal_data: Dict[str, Any], market_data: Dict[str, Any]) -> float:
        """Score fundamental analysis factors"""
        try:
            score = 0.0
            factors = 0
            
            # Earnings proximity
            days_to_earnings = market_data.get("days_to_earnings", 999)
            if days_to_earnings <= 7:
                earnings_score = 90  # High impact
            elif days_to_earnings <= 14:
                earnings_score = 70  # Medium impact
            elif days_to_earnings <= 30:
                earnings_score = 60  # Low impact
            else:
                earnings_score = 50  # No impact
            
            score += earnings_score
            factors += 1
            
            # Market cap consideration
            market_cap = market_data.get("market_cap", 0)
            if market_cap > 100e9:  # Large cap
                cap_score = 80
            elif market_cap > 10e9:  # Mid cap
                cap_score = 70
            else:  # Small cap
                cap_score = 60
            
            score += cap_score
            factors += 1
            
            # Sector performance
            sector_performance = market_data.get("sector_performance", 0)
            sector_score = min(100, max(0, sector_performance * 100 + 50))
            score += sector_score
            factors += 1
            
            return score / factors if factors > 0 else 50.0
            
        except Exception as e:
            logger.error(f"Error scoring fundamental factors: {e}")
            return 50.0
    
    def _score_market_conditions(self, market_data: Dict[str, Any]) -> float:
        """Score overall market conditions"""
        try:
            score = 0.0
            factors = 0
            
            # VIX level
            vix = market_data.get("vix", 20)
            if vix < 15:  # Low volatility
                vix_score = 70
            elif vix < 25:  # Normal volatility
                vix_score = 80
            elif vix < 35:  # High volatility
                vix_score = 60
            else:  # Extreme volatility
                vix_score = 40
            
            score += vix_score
            factors += 1
            
            # Market trend
            market_trend = market_data.get("market_trend", 0)
            trend_score = min(100, max(0, market_trend * 50 + 50))
            score += trend_score
            factors += 1
            
            # Market breadth
            advance_decline = market_data.get("advance_decline_ratio", 1.0)
            breadth_score = min(100, advance_decline * 50)
            score += breadth_score
            factors += 1
            
            # Economic indicators
            economic_score = market_data.get("economic_score", 50)
            score += economic_score
            factors += 1
            
            return score / factors if factors > 0 else 50.0
            
        except Exception as e:
            logger.error(f"Error scoring market conditions: {e}")
            return 50.0
    
    def _score_timing_factors(self, signal_data: Dict[str, Any], market_data: Dict[str, Any]) -> float:
        """Score timing-related factors"""
        try:
            score = 0.0
            factors = 0
            
            # Time to expiration (for options)
            time_horizon = signal_data.get("time_horizon", 72)  # Hours
            days_to_expiry = time_horizon / 24
            
            if days_to_expiry <= 1:  # Very short term
                timing_score = 60
            elif days_to_expiry <= 7:  # Short term
                timing_score = 85
            elif days_to_expiry <= 30:  # Medium term
                timing_score = 75
            else:  # Long term
                timing_score = 65
            
            score += timing_score
            factors += 1
            
            # Market hours
            current_hour = datetime.now().hour
            if 9 <= current_hour <= 16:  # Market hours
                hours_score = 90
            elif 4 <= current_hour <= 9 or 16 <= current_hour <= 20:  # Pre/post market
                hours_score = 70
            else:  # Overnight
                hours_score = 50
            
            score += hours_score
            factors += 1
            
            # Day of week
            day_of_week = datetime.now().weekday()  # 0=Monday, 6=Sunday
            if day_of_week in [0, 1, 2]:  # Monday-Wednesday
                day_score = 85
            elif day_of_week == 3:  # Thursday
                day_score = 80
            elif day_of_week == 4:  # Friday
                day_score = 70
            else:  # Weekend
                day_score = 50
            
            score += day_score
            factors += 1
            
            return score / factors if factors > 0 else 50.0
            
        except Exception as e:
            logger.error(f"Error scoring timing factors: {e}")
            return 50.0
    
    def _score_risk_factors(self, signal_data: Dict[str, Any], market_data: Dict[str, Any],
                          portfolio_data: Dict[str, Any] = None) -> float:
        """Score risk factors (higher score = higher risk)"""
        try:
            risk_score = 0.0
            factors = 0
            
            # Position size risk
            position_size = signal_data.get("position_size", 0.02)
            size_risk = min(100, position_size * 2000)  # 5% = 100 risk points
            risk_score += size_risk
            factors += 1
            
            # Volatility risk
            volatility = market_data.get("implied_volatility", 0.25)
            vol_risk = min(100, volatility * 200)  # 50% IV = 100 risk points
            risk_score += vol_risk
            factors += 1
            
            # Liquidity risk
            avg_volume = market_data.get("avg_volume", 1000000)
            if avg_volume > 1000000:
                liquidity_risk = 10
            elif avg_volume > 500000:
                liquidity_risk = 30
            elif avg_volume > 100000:
                liquidity_risk = 60
            else:
                liquidity_risk = 90
            
            risk_score += liquidity_risk
            factors += 1
            
            # Time decay risk (for options)
            theta = signal_data.get("options_factors", {}).get("theta", 0)
            time_risk = min(100, abs(theta) * 1000)
            risk_score += time_risk
            factors += 1
            
            # Market risk
            beta = market_data.get("beta", 1.0)
            market_risk = min(100, abs(beta) * 50)
            risk_score += market_risk
            factors += 1
            
            # Correlation risk (if portfolio data available)
            if portfolio_data:
                correlation_risk = self._calculate_portfolio_correlation_risk(signal_data, portfolio_data)
                risk_score += correlation_risk
                factors += 1
            
            return risk_score / factors if factors > 0 else 50.0
            
        except Exception as e:
            logger.error(f"Error scoring risk factors: {e}")
            return 50.0
    
    def _score_reward_potential(self, signal_data: Dict[str, Any], market_data: Dict[str, Any]) -> float:
        """Score reward potential"""
        try:
            expected_return = signal_data.get("expected_return", 0.05)
            max_risk = signal_data.get("max_risk", 0.02)
            
            # Risk-reward ratio
            risk_reward_ratio = expected_return / max_risk if max_risk > 0 else 0
            
            # Score based on risk-reward ratio
            if risk_reward_ratio >= 3.0:
                rr_score = 95
            elif risk_reward_ratio >= 2.5:
                rr_score = 85
            elif risk_reward_ratio >= 2.0:
                rr_score = 75
            elif risk_reward_ratio >= 1.5:
                rr_score = 60
            else:
                rr_score = 40
            
            # Absolute return potential
            return_score = min(100, expected_return * 500)  # 20% return = 100 points
            
            # Probability of success
            confidence = signal_data.get("confidence", 0.6)
            prob_score = confidence * 100
            
            # Combined reward score
            reward_score = (rr_score * 0.4 + return_score * 0.3 + prob_score * 0.3)
            
            return min(100, reward_score)
            
        except Exception as e:
            logger.error(f"Error scoring reward potential: {e}")
            return 50.0
    
    def _score_volatility_factors(self, signal_data: Dict[str, Any], market_data: Dict[str, Any]) -> float:
        """Score volatility-related factors"""
        try:
            # Implied volatility percentile
            iv_percentile = market_data.get("iv_percentile", 50)
            
            # Strategy-specific volatility scoring
            strategy_type = signal_data.get("strategy_type", "")
            
            if "volatility" in strategy_type.lower() or "straddle" in strategy_type.lower():
                # Volatility strategies prefer low IV
                if iv_percentile < 20:
                    vol_score = 90
                elif iv_percentile < 40:
                    vol_score = 70
                else:
                    vol_score = 40
            else:
                # Directional strategies prefer moderate IV
                if 30 <= iv_percentile <= 70:
                    vol_score = 80
                else:
                    vol_score = 60
            
            # Historical vs implied volatility
            hv = market_data.get("historical_volatility", 0.25)
            iv = market_data.get("implied_volatility", 0.25)
            
            hv_iv_ratio = hv / iv if iv > 0 else 1
            
            if 0.8 <= hv_iv_ratio <= 1.2:  # Fair pricing
                hv_iv_score = 80
            elif hv_iv_ratio < 0.8:  # IV overpriced
                hv_iv_score = 60
            else:  # IV underpriced
                hv_iv_score = 90
            
            return (vol_score + hv_iv_score) / 2
            
        except Exception as e:
            logger.error(f"Error scoring volatility factors: {e}")
            return 50.0
    
    def _score_liquidity_factors(self, signal_data: Dict[str, Any], market_data: Dict[str, Any]) -> float:
        """Score liquidity factors"""
        try:
            # Average daily volume
            avg_volume = market_data.get("avg_volume", 0)
            
            if avg_volume > 5000000:
                volume_score = 95
            elif avg_volume > 1000000:
                volume_score = 85
            elif avg_volume > 500000:
                volume_score = 70
            elif avg_volume > 100000:
                volume_score = 50
            else:
                volume_score = 30
            
            # Bid-ask spread
            bid_ask_spread = market_data.get("bid_ask_spread", 0.01)
            spread_score = max(0, 100 - bid_ask_spread * 10000)  # 1% spread = 0 points
            
            # Options open interest (if applicable)
            open_interest = market_data.get("options_open_interest", 0)
            if open_interest > 10000:
                oi_score = 90
            elif open_interest > 1000:
                oi_score = 70
            elif open_interest > 100:
                oi_score = 50
            else:
                oi_score = 30
            
            # Market cap liquidity
            market_cap = market_data.get("market_cap", 0)
            if market_cap > 10e9:
                cap_score = 90
            elif market_cap > 1e9:
                cap_score = 70
            else:
                cap_score = 50
            
            return (volume_score + spread_score + oi_score + cap_score) / 4
            
        except Exception as e:
            logger.error(f"Error scoring liquidity factors: {e}")
            return 50.0
    
    def _calculate_confidence_score(self, signal_data: Dict[str, Any], market_data: Dict[str, Any]) -> float:
        """Calculate overall confidence score"""
        try:
            # Base confidence from signal
            base_confidence = signal_data.get("confidence", 0.6) * 100
            
            # Strategy historical performance
            strategy_name = signal_data.get("strategy_name", "")
            strategy_performance = self.historical_performance.get(strategy_name, {})
            win_rate = strategy_performance.get("win_rate", 0.6)
            
            # Market regime confidence
            market_regime = self._detect_market_regime(market_data)
            regime_confidence = self._get_regime_confidence(strategy_name, market_regime)
            
            # Signal strength factors
            signal_strength = signal_data.get("signal_strength", 0.5)
            pattern_strength = signal_data.get("pattern_strength", 0.5)
            
            # Combine confidence factors
            confidence_score = (
                base_confidence * 0.3 +
                win_rate * 100 * 0.25 +
                regime_confidence * 0.2 +
                signal_strength * 100 * 0.15 +
                pattern_strength * 100 * 0.1
            )
            
            return min(100, confidence_score)
            
        except Exception as e:
            logger.error(f"Error calculating confidence score: {e}")
            return 60.0
    
    def _calculate_weighted_score(self, component_scores: Dict[str, float]) -> float:
        """Calculate weighted overall score"""
        try:
            total_score = 0.0
            total_weight = 0.0
            
            score_mapping = {
                "technical": "technical_analysis",
                "fundamental": "technical_analysis",  # Use same weight
                "market": "market_conditions",
                "timing": "timing",
                "risk": "risk_management",
                "reward": "risk_management",  # Use same weight
                "volatility": "volatility",
                "liquidity": "liquidity"
            }
            
            for component, score in component_scores.items():
                weight_key = score_mapping.get(component, "technical_analysis")
                weight = self.scoring_weights.get(weight_key, 0.1)
                
                total_score += score * weight
                total_weight += weight
            
            return total_score / total_weight if total_weight > 0 else 50.0
            
        except Exception as e:
            logger.error(f"Error calculating weighted score: {e}")
            return 50.0
    
    def _determine_quality_grade(self, overall_score: float, confidence_score: float) -> SignalQuality:
        """Determine signal quality grade"""
        combined_score = (overall_score + confidence_score) / 2
        
        if combined_score >= 85:
            return SignalQuality.EXCELLENT
        elif combined_score >= 70:
            return SignalQuality.GOOD
        elif combined_score >= 55:
            return SignalQuality.FAIR
        else:
            return SignalQuality.POOR
    
    def _determine_priority(self, overall_score: float, risk_score: float, signal_data: Dict[str, Any]) -> SignalPriority:
        """Determine signal priority"""
        # High score, low risk = high priority
        priority_score = overall_score - risk_score
        
        # Adjust for expected return
        expected_return = signal_data.get("expected_return", 0.05)
        priority_score += expected_return * 500  # Boost for high returns
        
        # Adjust for time sensitivity
        time_horizon = signal_data.get("time_horizon", 72)
        if time_horizon <= 24:  # Very time sensitive
            priority_score += 20
        elif time_horizon <= 48:  # Time sensitive
            priority_score += 10
        
        if priority_score >= 90:
            return SignalPriority.CRITICAL
        elif priority_score >= 75:
            return SignalPriority.HIGH
        elif priority_score >= 60:
            return SignalPriority.MEDIUM
        else:
            return SignalPriority.LOW
    
    def _analyze_risk_factors(self, signal_data: Dict[str, Any], market_data: Dict[str, Any],
                            portfolio_data: Dict[str, Any] = None) -> Dict[str, float]:
        """Analyze detailed risk factors"""
        risk_factors = {}
        
        # Market risk
        beta = market_data.get("beta", 1.0)
        risk_factors["market_risk"] = min(100, abs(beta) * 50)
        
        # Volatility risk
        iv = market_data.get("implied_volatility", 0.25)
        risk_factors["volatility_risk"] = min(100, iv * 200)
        
        # Liquidity risk
        avg_volume = market_data.get("avg_volume", 1000000)
        risk_factors["liquidity_risk"] = max(0, 100 - avg_volume / 10000)
        
        # Time decay risk
        theta = signal_data.get("options_factors", {}).get("theta", 0)
        risk_factors["time_decay_risk"] = min(100, abs(theta) * 1000)
        
        # Concentration risk
        position_size = signal_data.get("position_size", 0.02)
        risk_factors["concentration_risk"] = min(100, position_size * 2000)
        
        return risk_factors
    
    def _analyze_confidence_factors(self, signal_data: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, float]:
        """Analyze detailed confidence factors"""
        confidence_factors = {}
        
        # Technical confidence
        signal_strength = signal_data.get("signal_strength", 0.5)
        confidence_factors["technical_confidence"] = signal_strength * 100
        
        # Pattern confidence
        pattern_strength = signal_data.get("pattern_strength", 0.5)
        confidence_factors["pattern_confidence"] = pattern_strength * 100
        
        # Volume confirmation
        volume_ratio = signal_data.get("technical_factors", {}).get("volume_ratio", 1.0)
        confidence_factors["volume_confirmation"] = min(100, volume_ratio * 50)
        
        # Market alignment
        market_trend = market_data.get("market_trend", 0)
        signal_direction = 1 if signal_data.get("direction", "bullish") == "bullish" else -1
        alignment = market_trend * signal_direction
        confidence_factors["market_alignment"] = max(0, alignment * 50 + 50)
        
        return confidence_factors
    
    def _calculate_correlation_risk(self, signal: Dict[str, Any], all_signals: List[Dict[str, Any]]) -> float:
        """Calculate correlation risk with other signals"""
        try:
            symbol = signal.get("symbol", "")
            sector = signal.get("sector", "")
            
            correlation_risk = 0.0
            
            for other_signal in all_signals:
                if other_signal.get("signal_id") == signal.get("signal_id"):
                    continue
                
                # Same symbol
                if other_signal.get("symbol") == symbol:
                    correlation_risk += 30
                
                # Same sector
                elif other_signal.get("sector") == sector:
                    correlation_risk += 10
                
                # Same strategy type
                elif other_signal.get("strategy_type") == signal.get("strategy_type"):
                    correlation_risk += 5
            
            return min(100, correlation_risk)
            
        except Exception as e:
            logger.error(f"Error calculating correlation risk: {e}")
            return 0.0
    
    def _calculate_portfolio_impact(self, signal: Dict[str, Any], portfolio_data: Dict[str, Any] = None) -> float:
        """Calculate portfolio impact score"""
        if not portfolio_data:
            return 0.0
        
        try:
            position_size = signal.get("position_size", 0.02)
            portfolio_value = portfolio_data.get("total_value", 100000)
            
            # Impact as percentage of portfolio
            impact = (position_size * portfolio_value) / portfolio_value * 100
            
            return min(100, impact * 20)  # 5% position = 100 impact points
            
        except Exception as e:
            logger.error(f"Error calculating portfolio impact: {e}")
            return 0.0
    
    def _calculate_portfolio_correlation_risk(self, signal_data: Dict[str, Any], portfolio_data: Dict[str, Any]) -> float:
        """Calculate portfolio correlation risk"""
        try:
            # Simplified correlation risk calculation
            symbol = signal_data.get("symbol", "")
            existing_positions = portfolio_data.get("positions", [])
            
            correlation_risk = 0.0
            
            for position in existing_positions:
                if position.get("symbol") == symbol:
                    correlation_risk += 40  # Same symbol
                elif position.get("sector") == signal_data.get("sector", ""):
                    correlation_risk += 15  # Same sector
            
            return min(100, correlation_risk)
            
        except Exception as e:
            logger.error(f"Error calculating portfolio correlation risk: {e}")
            return 0.0
    
    def _apply_portfolio_adjustments(self, ranked_signals: List[RankedSignal], 
                                   portfolio_data: Dict[str, Any] = None) -> List[RankedSignal]:
        """Apply portfolio-level adjustments to rankings"""
        if not portfolio_data:
            return ranked_signals
        
        try:
            # Adjust for portfolio concentration
            symbol_counts = {}
            for signal in ranked_signals:
                symbol = signal.symbol
                symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
            
            # Penalize over-concentration
            for signal in ranked_signals:
                if symbol_counts[signal.symbol] > 2:  # More than 2 signals for same symbol
                    penalty = (symbol_counts[signal.symbol] - 2) * 10
                    signal.score.overall_score = max(0, signal.score.overall_score - penalty)
            
            # Re-sort after adjustments
            ranked_signals.sort(key=lambda s: s.score.overall_score, reverse=True)
            
            # Update ranks
            for i, signal in enumerate(ranked_signals):
                signal.rank = i + 1
                signal.percentile = ((len(ranked_signals) - i) / len(ranked_signals)) * 100
            
            return ranked_signals
            
        except Exception as e:
            logger.error(f"Error applying portfolio adjustments: {e}")
            return ranked_signals
    
    def _detect_market_regime(self, market_data: Dict[str, Any]) -> str:
        """Detect current market regime"""
        try:
            vix = market_data.get("vix", 20)
            market_trend = market_data.get("market_trend", 0)
            
            if vix > 30:
                return "high_volatility"
            elif vix < 15:
                return "low_volatility"
            elif market_trend > 0.02:
                return "bull_market"
            elif market_trend < -0.02:
                return "bear_market"
            else:
                return "sideways_market"
                
        except Exception as e:
            logger.error(f"Error detecting market regime: {e}")
            return "unknown"
    
    def _get_regime_confidence(self, strategy_name: str, market_regime: str) -> float:
        """Get strategy confidence for market regime"""
        # Strategy performance by regime (simplified)
        regime_performance = {
            "momentum_breakout": {
                "bull_market": 85,
                "bear_market": 60,
                "sideways_market": 70,
                "high_volatility": 75,
                "low_volatility": 65
            },
            "mean_reversion": {
                "bull_market": 70,
                "bear_market": 70,
                "sideways_market": 85,
                "high_volatility": 60,
                "low_volatility": 80
            },
            "volatility": {
                "bull_market": 65,
                "bear_market": 65,
                "sideways_market": 60,
                "high_volatility": 90,
                "low_volatility": 40
            }
        }
        
        strategy_key = strategy_name.lower().replace(" ", "_")
        for key in regime_performance:
            if key in strategy_key:
                return regime_performance[key].get(market_regime, 70)
        
        return 70  # Default confidence
    
    def _initialize_models(self):
        """Initialize ML models for scoring"""
        try:
            # Initialize random forest for signal scoring
            self.scoring_models["random_forest"] = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
            # Initialize gradient boosting for confidence scoring
            self.scoring_models["gradient_boosting"] = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                random_state=42
            )
            
            # Initialize feature scalers
            self.feature_scalers["standard"] = StandardScaler()
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
    
    def _create_default_score(self) -> SignalScore:
        """Create default score for error cases"""
        return SignalScore(
            overall_score=50.0,
            confidence_score=50.0,
            quality_grade=SignalQuality.FAIR,
            priority=SignalPriority.MEDIUM,
            risk_score=50.0,
            reward_score=50.0,
            timing_score=50.0,
            technical_score=50.0,
            fundamental_score=50.0,
            market_score=50.0,
            volatility_score=50.0,
            liquidity_score=50.0
        )
    
    def update_performance(self, signal_id: str, actual_return: float, success: bool):
        """Update historical performance data"""
        try:
            # Update performance tracking
            # This would typically update a database
            pass
            
        except Exception as e:
            logger.error(f"Error updating performance: {e}")
    
    def get_scoring_statistics(self) -> Dict[str, Any]:
        """Get scoring engine statistics"""
        return {
            "total_signals_scored": len(self.historical_performance),
            "scoring_weights": self.scoring_weights,
            "risk_weights": self.risk_weights,
            "model_versions": {name: "1.0" for name in self.scoring_models.keys()}
        }

# Factory function
def create_scoring_engine() -> SignalScoringEngine:
    """Create signal scoring engine"""
    return SignalScoringEngine()

if __name__ == "__main__":
    # Example usage
    engine = create_scoring_engine()
    
    # Mock signal data
    signal_data = {
        "signal_id": "test_001",
        "symbol": "AAPL",
        "strategy_name": "Momentum Breakout",
        "strategy_type": "breakout",
        "confidence": 0.75,
        "signal_strength": 0.8,
        "pattern_strength": 0.7,
        "expected_return": 0.08,
        "max_risk": 0.03,
        "position_size": 0.02,
        "time_horizon": 48,
        "technical_factors": {
            "rsi": 65,
            "macd_histogram": 0.5,
            "volume_ratio": 1.8,
            "breakout_strength": 0.04
        },
        "options_factors": {
            "theta": -0.05
        }
    }
    
    # Mock market data
    market_data = {
        "vix": 22,
        "market_trend": 0.01,
        "beta": 1.2,
        "avg_volume": 2000000,
        "implied_volatility": 0.28,
        "historical_volatility": 0.25,
        "iv_percentile": 35,
        "market_cap": 2.5e12,
        "days_to_earnings": 10
    }
    
    # Score the signal
    score = engine.score_signal(signal_data, market_data)
    
    print(f"Signal Score: {score.overall_score:.1f}")
    print(f"Quality Grade: {score.quality_grade.value}")
    print(f"Priority: {score.priority.name}")
    print(f"Confidence: {score.confidence_score:.1f}")
    print(f"Risk Score: {score.risk_score:.1f}")
    
    # Test ranking multiple signals
    signals = [signal_data.copy() for _ in range(5)]
    for i, sig in enumerate(signals):
        sig["signal_id"] = f"test_{i:03d}"
        sig["confidence"] = 0.6 + i * 0.05
        sig["expected_return"] = 0.05 + i * 0.01
    
    ranked_signals = engine.rank_signals(signals, market_data)
    
    print(f"\nRanked {len(ranked_signals)} signals:")
    for signal in ranked_signals[:3]:  # Top 3
        print(f"Rank {signal.rank}: {signal.signal_id} - Score: {signal.score.overall_score:.1f}")

