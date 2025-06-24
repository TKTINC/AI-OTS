"""
Advanced Pattern Recognition Engine for AI Options Trading System
Sophisticated algorithms for identifying market structures and trading patterns
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging
import math
from scipy import stats, signal
from scipy.optimize import minimize
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import talib
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class PatternType(Enum):
    """Types of market patterns"""
    TREND = "trend"
    REVERSAL = "reversal"
    CONTINUATION = "continuation"
    CONSOLIDATION = "consolidation"
    BREAKOUT = "breakout"
    SUPPORT_RESISTANCE = "support_resistance"
    HARMONIC = "harmonic"
    CANDLESTICK = "candlestick"
    VOLUME = "volume"
    MOMENTUM = "momentum"

class PatternStrength(Enum):
    """Pattern strength levels"""
    WEAK = 1
    MODERATE = 2
    STRONG = 3
    VERY_STRONG = 4

@dataclass
class PatternSignal:
    """Pattern recognition signal"""
    pattern_name: str
    pattern_type: PatternType
    strength: PatternStrength
    confidence: float
    direction: str  # "bullish", "bearish", "neutral"
    entry_price: float
    target_price: float
    stop_loss: float
    probability: float
    time_horizon: int  # Hours
    description: str
    key_levels: Dict[str, float]
    pattern_data: Dict[str, Any]
    risk_reward_ratio: float

class PatternRecognitionEngine:
    """Advanced pattern recognition engine"""
    
    def __init__(self):
        self.patterns = {
            "trend_analysis": self._analyze_trends,
            "support_resistance": self._find_support_resistance,
            "breakout_patterns": self._detect_breakouts,
            "reversal_patterns": self._detect_reversals,
            "harmonic_patterns": self._detect_harmonic_patterns,
            "candlestick_patterns": self._detect_candlestick_patterns,
            "volume_patterns": self._analyze_volume_patterns,
            "momentum_divergence": self._detect_momentum_divergence,
            "consolidation_patterns": self._detect_consolidation,
            "fibonacci_patterns": self._analyze_fibonacci_levels
        }
        
        # Pattern weights for signal generation
        self.pattern_weights = {
            "trend_analysis": 0.15,
            "support_resistance": 0.12,
            "breakout_patterns": 0.15,
            "reversal_patterns": 0.12,
            "harmonic_patterns": 0.08,
            "candlestick_patterns": 0.10,
            "volume_patterns": 0.08,
            "momentum_divergence": 0.10,
            "consolidation_patterns": 0.05,
            "fibonacci_patterns": 0.05
        }
        
        # Pattern performance tracking
        self.pattern_performance = {}
    
    def analyze_patterns(self, symbol: str, price_data: pd.DataFrame, 
                        volume_data: pd.DataFrame, indicators: Dict[str, Any]) -> List[PatternSignal]:
        """Analyze all patterns for a symbol"""
        patterns = []
        
        for pattern_name, pattern_func in self.patterns.items():
            try:
                pattern_signals = pattern_func(symbol, price_data, volume_data, indicators)
                if pattern_signals:
                    if isinstance(pattern_signals, list):
                        patterns.extend(pattern_signals)
                    else:
                        patterns.append(pattern_signals)
            except Exception as e:
                logger.error(f"Error in pattern {pattern_name} for {symbol}: {e}")
        
        # Sort by confidence and strength
        patterns.sort(key=lambda p: p.confidence * p.strength.value, reverse=True)
        
        return patterns
    
    def _analyze_trends(self, symbol: str, price_data: pd.DataFrame, 
                       volume_data: pd.DataFrame, indicators: Dict[str, Any]) -> Optional[PatternSignal]:
        """Analyze trend patterns"""
        try:
            if len(price_data) < 50:
                return None
            
            close_prices = price_data['close'].values
            high_prices = price_data['high'].values
            low_prices = price_data['low'].values
            
            # Calculate trend using multiple timeframes
            short_trend = self._calculate_trend(close_prices[-20:])  # 20 periods
            medium_trend = self._calculate_trend(close_prices[-50:])  # 50 periods
            long_trend = self._calculate_trend(close_prices)  # All data
            
            # Trend strength calculation
            trend_alignment = sum([
                1 if short_trend > 0 else -1,
                1 if medium_trend > 0 else -1,
                1 if long_trend > 0 else -1
            ])
            
            trend_strength = abs(trend_alignment) / 3
            
            if trend_strength < 0.6:  # Weak trend
                return None
            
            # Determine trend direction
            if trend_alignment > 0:
                direction = "bullish"
                target_multiplier = 1.05
                stop_multiplier = 0.97
            else:
                direction = "bearish"
                target_multiplier = 0.95
                stop_multiplier = 1.03
            
            current_price = close_prices[-1]
            
            # Calculate trend line
            trend_line = self._calculate_trend_line(price_data)
            
            # Trend continuation probability
            trend_momentum = abs(short_trend)
            volume_confirmation = self._check_volume_confirmation(volume_data, direction)
            
            confidence = min(0.9, trend_strength * 0.6 + trend_momentum * 0.2 + volume_confirmation * 0.2)
            
            return PatternSignal(
                pattern_name="Trend Continuation",
                pattern_type=PatternType.TREND,
                strength=PatternStrength(min(4, int(trend_strength * 4) + 1)),
                confidence=confidence,
                direction=direction,
                entry_price=current_price,
                target_price=current_price * target_multiplier,
                stop_loss=current_price * stop_multiplier,
                probability=confidence,
                time_horizon=72,  # 3 days
                description=f"{direction.title()} trend with {trend_strength:.1%} alignment",
                key_levels={
                    "trend_line": trend_line,
                    "short_trend": short_trend,
                    "medium_trend": medium_trend,
                    "long_trend": long_trend
                },
                pattern_data={
                    "trend_alignment": trend_alignment,
                    "trend_strength": trend_strength,
                    "trend_momentum": trend_momentum,
                    "volume_confirmation": volume_confirmation
                },
                risk_reward_ratio=abs(target_multiplier - 1) / abs(stop_multiplier - 1)
            )
            
        except Exception as e:
            logger.error(f"Error in trend analysis: {e}")
            return None
    
    def _find_support_resistance(self, symbol: str, price_data: pd.DataFrame,
                               volume_data: pd.DataFrame, indicators: Dict[str, Any]) -> List[PatternSignal]:
        """Find support and resistance levels"""
        try:
            if len(price_data) < 100:
                return []
            
            high_prices = price_data['high'].values
            low_prices = price_data['low'].values
            close_prices = price_data['close'].values
            volume = volume_data['volume'].values
            
            # Find pivot points
            resistance_levels = self._find_resistance_levels(high_prices, volume)
            support_levels = self._find_support_levels(low_prices, volume)
            
            current_price = close_prices[-1]
            patterns = []
            
            # Analyze support levels
            for level, strength in support_levels:
                distance = (current_price - level) / current_price
                
                if 0 <= distance <= 0.02:  # Within 2% above support
                    confidence = min(0.85, strength * 0.7 + (0.02 - distance) * 25)
                    
                    patterns.append(PatternSignal(
                        pattern_name="Support Level",
                        pattern_type=PatternType.SUPPORT_RESISTANCE,
                        strength=PatternStrength(min(4, int(strength * 4) + 1)),
                        confidence=confidence,
                        direction="bullish",
                        entry_price=current_price,
                        target_price=level * 1.04,  # 4% above support
                        stop_loss=level * 0.99,     # 1% below support
                        probability=confidence,
                        time_horizon=48,
                        description=f"Strong support at ${level:.2f}",
                        key_levels={"support": level},
                        pattern_data={
                            "level_strength": strength,
                            "distance_to_level": distance,
                            "touches": self._count_level_touches(low_prices, level)
                        },
                        risk_reward_ratio=4.0
                    ))
            
            # Analyze resistance levels
            for level, strength in resistance_levels:
                distance = (level - current_price) / current_price
                
                if 0 <= distance <= 0.02:  # Within 2% below resistance
                    confidence = min(0.85, strength * 0.7 + (0.02 - distance) * 25)
                    
                    patterns.append(PatternSignal(
                        pattern_name="Resistance Level",
                        pattern_type=PatternType.SUPPORT_RESISTANCE,
                        strength=PatternStrength(min(4, int(strength * 4) + 1)),
                        confidence=confidence,
                        direction="bearish",
                        entry_price=current_price,
                        target_price=level * 0.96,  # 4% below resistance
                        stop_loss=level * 1.01,     # 1% above resistance
                        probability=confidence,
                        time_horizon=48,
                        description=f"Strong resistance at ${level:.2f}",
                        key_levels={"resistance": level},
                        pattern_data={
                            "level_strength": strength,
                            "distance_to_level": distance,
                            "touches": self._count_level_touches(high_prices, level)
                        },
                        risk_reward_ratio=4.0
                    ))
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error in support/resistance analysis: {e}")
            return []
    
    def _detect_breakouts(self, symbol: str, price_data: pd.DataFrame,
                         volume_data: pd.DataFrame, indicators: Dict[str, Any]) -> Optional[PatternSignal]:
        """Detect breakout patterns"""
        try:
            if len(price_data) < 50:
                return None
            
            close_prices = price_data['close'].values
            high_prices = price_data['high'].values
            low_prices = price_data['low'].values
            volume = volume_data['volume'].values
            
            # Calculate consolidation range
            lookback = 20
            recent_highs = high_prices[-lookback:]
            recent_lows = low_prices[-lookback:]
            
            range_high = np.max(recent_highs)
            range_low = np.min(recent_lows)
            range_width = (range_high - range_low) / close_prices[-1]
            
            # Look for tight consolidation
            if range_width > 0.05:  # Range too wide
                return None
            
            current_price = close_prices[-1]
            
            # Check for breakout
            breakout_threshold = 0.005  # 0.5% breakout
            
            if current_price > range_high * (1 + breakout_threshold):
                # Upward breakout
                direction = "bullish"
                breakout_strength = (current_price - range_high) / range_high
                
            elif current_price < range_low * (1 - breakout_threshold):
                # Downward breakout
                direction = "bearish"
                breakout_strength = (range_low - current_price) / range_low
                
            else:
                return None  # No breakout
            
            # Volume confirmation
            avg_volume = np.mean(volume[-20:])
            current_volume = volume[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            volume_confirmation = volume_ratio > 1.5  # 50% volume increase
            
            if not volume_confirmation:
                return None
            
            # Calculate confidence
            confidence = min(0.9, breakout_strength * 20 + volume_ratio * 0.2 + (1 - range_width) * 2)
            
            # Set targets
            if direction == "bullish":
                target_price = current_price * (1 + range_width * 2)
                stop_loss = range_high * 0.995
            else:
                target_price = current_price * (1 - range_width * 2)
                stop_loss = range_low * 1.005
            
            return PatternSignal(
                pattern_name="Breakout Pattern",
                pattern_type=PatternType.BREAKOUT,
                strength=PatternStrength(min(4, int(confidence * 4) + 1)),
                confidence=confidence,
                direction=direction,
                entry_price=current_price,
                target_price=target_price,
                stop_loss=stop_loss,
                probability=confidence,
                time_horizon=24,  # 1 day
                description=f"{direction.title()} breakout from {range_width:.1%} range",
                key_levels={
                    "range_high": range_high,
                    "range_low": range_low,
                    "breakout_level": range_high if direction == "bullish" else range_low
                },
                pattern_data={
                    "range_width": range_width,
                    "breakout_strength": breakout_strength,
                    "volume_ratio": volume_ratio,
                    "consolidation_days": lookback
                },
                risk_reward_ratio=abs(target_price - current_price) / abs(stop_loss - current_price)
            )
            
        except Exception as e:
            logger.error(f"Error in breakout detection: {e}")
            return None
    
    def _detect_reversals(self, symbol: str, price_data: pd.DataFrame,
                         volume_data: pd.DataFrame, indicators: Dict[str, Any]) -> List[PatternSignal]:
        """Detect reversal patterns"""
        try:
            if len(price_data) < 30:
                return []
            
            patterns = []
            
            # Double top/bottom patterns
            double_patterns = self._detect_double_patterns(price_data, volume_data)
            patterns.extend(double_patterns)
            
            # Head and shoulders patterns
            hs_patterns = self._detect_head_shoulders(price_data, volume_data)
            patterns.extend(hs_patterns)
            
            # Divergence patterns
            divergence_patterns = self._detect_price_divergence(price_data, indicators)
            patterns.extend(divergence_patterns)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error in reversal detection: {e}")
            return []
    
    def _detect_harmonic_patterns(self, symbol: str, price_data: pd.DataFrame,
                                volume_data: pd.DataFrame, indicators: Dict[str, Any]) -> List[PatternSignal]:
        """Detect harmonic patterns (Gartley, Butterfly, etc.)"""
        try:
            if len(price_data) < 100:
                return []
            
            patterns = []
            close_prices = price_data['close'].values
            high_prices = price_data['high'].values
            low_prices = price_data['low'].values
            
            # Find significant swing points
            swing_highs = self._find_swing_points(high_prices, 'high')
            swing_lows = self._find_swing_points(low_prices, 'low')
            
            # Combine and sort swing points
            all_swings = []
            for i, price in swing_highs:
                all_swings.append((i, price, 'high'))
            for i, price in swing_lows:
                all_swings.append((i, price, 'low'))
            
            all_swings.sort(key=lambda x: x[0])
            
            if len(all_swings) < 5:
                return patterns
            
            # Look for ABCD patterns
            for i in range(len(all_swings) - 4):
                points = all_swings[i:i+5]
                
                # Check for alternating high/low pattern
                if self._is_valid_abcd_pattern(points):
                    pattern = self._analyze_abcd_pattern(points, close_prices[-1])
                    if pattern:
                        patterns.append(pattern)
            
            # Look for Gartley patterns
            gartley_patterns = self._detect_gartley_patterns(all_swings, close_prices[-1])
            patterns.extend(gartley_patterns)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error in harmonic pattern detection: {e}")
            return []
    
    def _detect_candlestick_patterns(self, symbol: str, price_data: pd.DataFrame,
                                   volume_data: pd.DataFrame, indicators: Dict[str, Any]) -> List[PatternSignal]:
        """Detect candlestick patterns"""
        try:
            if len(price_data) < 10:
                return []
            
            patterns = []
            
            open_prices = price_data['open'].values
            high_prices = price_data['high'].values
            low_prices = price_data['low'].values
            close_prices = price_data['close'].values
            
            current_price = close_prices[-1]
            
            # Doji patterns
            doji = talib.CDLDOJI(open_prices, high_prices, low_prices, close_prices)
            if doji[-1] != 0:
                patterns.append(self._create_candlestick_signal(
                    "Doji", "neutral", current_price, 0.6, 24
                ))
            
            # Hammer patterns
            hammer = talib.CDLHAMMER(open_prices, high_prices, low_prices, close_prices)
            if hammer[-1] != 0:
                patterns.append(self._create_candlestick_signal(
                    "Hammer", "bullish", current_price, 0.7, 48
                ))
            
            # Shooting star
            shooting_star = talib.CDLSHOOTINGSTAR(open_prices, high_prices, low_prices, close_prices)
            if shooting_star[-1] != 0:
                patterns.append(self._create_candlestick_signal(
                    "Shooting Star", "bearish", current_price, 0.7, 48
                ))
            
            # Engulfing patterns
            engulfing = talib.CDLENGULFING(open_prices, high_prices, low_prices, close_prices)
            if engulfing[-1] > 0:
                patterns.append(self._create_candlestick_signal(
                    "Bullish Engulfing", "bullish", current_price, 0.75, 72
                ))
            elif engulfing[-1] < 0:
                patterns.append(self._create_candlestick_signal(
                    "Bearish Engulfing", "bearish", current_price, 0.75, 72
                ))
            
            # Morning/Evening star
            morning_star = talib.CDLMORNINGSTAR(open_prices, high_prices, low_prices, close_prices)
            evening_star = talib.CDLEVENINGSTAR(open_prices, high_prices, low_prices, close_prices)
            
            if morning_star[-1] != 0:
                patterns.append(self._create_candlestick_signal(
                    "Morning Star", "bullish", current_price, 0.8, 72
                ))
            
            if evening_star[-1] != 0:
                patterns.append(self._create_candlestick_signal(
                    "Evening Star", "bearish", current_price, 0.8, 72
                ))
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error in candlestick pattern detection: {e}")
            return []
    
    def _analyze_volume_patterns(self, symbol: str, price_data: pd.DataFrame,
                               volume_data: pd.DataFrame, indicators: Dict[str, Any]) -> Optional[PatternSignal]:
        """Analyze volume patterns"""
        try:
            if len(volume_data) < 20:
                return None
            
            volume = volume_data['volume'].values
            close_prices = price_data['close'].values
            
            # Volume trend analysis
            recent_volume = volume[-10:]
            avg_volume = np.mean(volume[-30:-10])
            
            volume_trend = np.polyfit(range(len(recent_volume)), recent_volume, 1)[0]
            volume_ratio = np.mean(recent_volume) / avg_volume if avg_volume > 0 else 1
            
            # Price-volume divergence
            price_change = (close_prices[-1] - close_prices[-10]) / close_prices[-10]
            volume_change = (np.mean(recent_volume) - avg_volume) / avg_volume if avg_volume > 0 else 0
            
            # Look for volume breakout
            if volume_ratio > 2.0 and volume_trend > 0:  # Strong volume increase
                direction = "bullish" if price_change > 0 else "bearish"
                confidence = min(0.8, volume_ratio * 0.3 + abs(price_change) * 10)
                
                current_price = close_prices[-1]
                
                if direction == "bullish":
                    target_price = current_price * 1.05
                    stop_loss = current_price * 0.98
                else:
                    target_price = current_price * 0.95
                    stop_loss = current_price * 1.02
                
                return PatternSignal(
                    pattern_name="Volume Breakout",
                    pattern_type=PatternType.VOLUME,
                    strength=PatternStrength(min(4, int(volume_ratio))),
                    confidence=confidence,
                    direction=direction,
                    entry_price=current_price,
                    target_price=target_price,
                    stop_loss=stop_loss,
                    probability=confidence,
                    time_horizon=48,
                    description=f"Volume surge ({volume_ratio:.1f}x) with {direction} bias",
                    key_levels={"avg_volume": avg_volume},
                    pattern_data={
                        "volume_ratio": volume_ratio,
                        "volume_trend": volume_trend,
                        "price_change": price_change,
                        "volume_change": volume_change
                    },
                    risk_reward_ratio=abs(target_price - current_price) / abs(stop_loss - current_price)
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error in volume pattern analysis: {e}")
            return None
    
    def _detect_momentum_divergence(self, symbol: str, price_data: pd.DataFrame,
                                  volume_data: pd.DataFrame, indicators: Dict[str, Any]) -> Optional[PatternSignal]:
        """Detect momentum divergence patterns"""
        try:
            if len(price_data) < 50:
                return None
            
            close_prices = price_data['close'].values
            
            # Get momentum indicators
            rsi = indicators.get('rsi', [])
            macd = indicators.get('macd', {})
            
            if len(rsi) < 20 or not macd:
                return None
            
            # Look for price vs RSI divergence
            price_peaks = self._find_peaks(close_prices[-20:])
            rsi_peaks = self._find_peaks(rsi[-20:])
            
            if len(price_peaks) >= 2 and len(rsi_peaks) >= 2:
                # Bearish divergence: higher price peaks, lower RSI peaks
                if (price_peaks[-1][1] > price_peaks[-2][1] and 
                    rsi_peaks[-1][1] < rsi_peaks[-2][1]):
                    
                    divergence_strength = (price_peaks[-1][1] - price_peaks[-2][1]) / price_peaks[-2][1]
                    rsi_divergence = (rsi_peaks[-2][1] - rsi_peaks[-1][1]) / rsi_peaks[-2][1]
                    
                    confidence = min(0.8, (divergence_strength + rsi_divergence) * 5)
                    
                    current_price = close_prices[-1]
                    
                    return PatternSignal(
                        pattern_name="Bearish Divergence",
                        pattern_type=PatternType.MOMENTUM,
                        strength=PatternStrength(min(4, int(confidence * 4) + 1)),
                        confidence=confidence,
                        direction="bearish",
                        entry_price=current_price,
                        target_price=current_price * 0.95,
                        stop_loss=current_price * 1.02,
                        probability=confidence,
                        time_horizon=72,
                        description="Price making higher highs while RSI making lower highs",
                        key_levels={
                            "price_peak_1": price_peaks[-2][1],
                            "price_peak_2": price_peaks[-1][1],
                            "rsi_peak_1": rsi_peaks[-2][1],
                            "rsi_peak_2": rsi_peaks[-1][1]
                        },
                        pattern_data={
                            "divergence_strength": divergence_strength,
                            "rsi_divergence": rsi_divergence
                        },
                        risk_reward_ratio=2.5
                    )
                
                # Bullish divergence: lower price peaks, higher RSI peaks
                elif (price_peaks[-1][1] < price_peaks[-2][1] and 
                      rsi_peaks[-1][1] > rsi_peaks[-2][1]):
                    
                    divergence_strength = (price_peaks[-2][1] - price_peaks[-1][1]) / price_peaks[-2][1]
                    rsi_divergence = (rsi_peaks[-1][1] - rsi_peaks[-2][1]) / rsi_peaks[-2][1]
                    
                    confidence = min(0.8, (divergence_strength + rsi_divergence) * 5)
                    
                    current_price = close_prices[-1]
                    
                    return PatternSignal(
                        pattern_name="Bullish Divergence",
                        pattern_type=PatternType.MOMENTUM,
                        strength=PatternStrength(min(4, int(confidence * 4) + 1)),
                        confidence=confidence,
                        direction="bullish",
                        entry_price=current_price,
                        target_price=current_price * 1.05,
                        stop_loss=current_price * 0.98,
                        probability=confidence,
                        time_horizon=72,
                        description="Price making lower highs while RSI making higher highs",
                        key_levels={
                            "price_peak_1": price_peaks[-2][1],
                            "price_peak_2": price_peaks[-1][1],
                            "rsi_peak_1": rsi_peaks[-2][1],
                            "rsi_peak_2": rsi_peaks[-1][1]
                        },
                        pattern_data={
                            "divergence_strength": divergence_strength,
                            "rsi_divergence": rsi_divergence
                        },
                        risk_reward_ratio=2.5
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Error in momentum divergence detection: {e}")
            return None
    
    def _detect_consolidation(self, symbol: str, price_data: pd.DataFrame,
                            volume_data: pd.DataFrame, indicators: Dict[str, Any]) -> Optional[PatternSignal]:
        """Detect consolidation patterns"""
        try:
            if len(price_data) < 30:
                return None
            
            close_prices = price_data['close'].values
            high_prices = price_data['high'].values
            low_prices = price_data['low'].values
            
            # Calculate price range over different periods
            periods = [10, 20, 30]
            ranges = []
            
            for period in periods:
                recent_high = np.max(high_prices[-period:])
                recent_low = np.min(low_prices[-period:])
                price_range = (recent_high - recent_low) / close_prices[-1]
                ranges.append(price_range)
            
            # Look for tight consolidation
            avg_range = np.mean(ranges)
            if avg_range > 0.04:  # Range too wide
                return None
            
            # Check for decreasing volatility
            volatility_trend = np.polyfit(range(len(ranges)), ranges, 1)[0]
            
            if volatility_trend > 0:  # Volatility increasing
                return None
            
            # Volume analysis during consolidation
            volume = volume_data['volume'].values
            avg_volume = np.mean(volume[-20:])
            
            current_price = close_prices[-1]
            confidence = min(0.7, (0.04 - avg_range) * 20 + abs(volatility_trend) * 50)
            
            return PatternSignal(
                pattern_name="Consolidation Pattern",
                pattern_type=PatternType.CONSOLIDATION,
                strength=PatternStrength(min(4, int(confidence * 4) + 1)),
                confidence=confidence,
                direction="neutral",
                entry_price=current_price,
                target_price=current_price,  # Neutral target
                stop_loss=current_price,     # Neutral stop
                probability=confidence,
                time_horizon=120,  # 5 days
                description=f"Tight consolidation ({avg_range:.1%} range) with decreasing volatility",
                key_levels={
                    "consolidation_high": np.max(high_prices[-20:]),
                    "consolidation_low": np.min(low_prices[-20:])
                },
                pattern_data={
                    "avg_range": avg_range,
                    "volatility_trend": volatility_trend,
                    "avg_volume": avg_volume
                },
                risk_reward_ratio=1.0  # Neutral for consolidation
            )
            
        except Exception as e:
            logger.error(f"Error in consolidation detection: {e}")
            return None
    
    def _analyze_fibonacci_levels(self, symbol: str, price_data: pd.DataFrame,
                                volume_data: pd.DataFrame, indicators: Dict[str, Any]) -> List[PatternSignal]:
        """Analyze Fibonacci retracement levels"""
        try:
            if len(price_data) < 50:
                return []
            
            patterns = []
            close_prices = price_data['close'].values
            high_prices = price_data['high'].values
            low_prices = price_data['low'].values
            
            # Find significant swing high and low
            lookback = 30
            swing_high = np.max(high_prices[-lookback:])
            swing_low = np.min(low_prices[-lookback:])
            
            # Calculate Fibonacci levels
            fib_levels = {
                0.236: swing_high - (swing_high - swing_low) * 0.236,
                0.382: swing_high - (swing_high - swing_low) * 0.382,
                0.500: swing_high - (swing_high - swing_low) * 0.500,
                0.618: swing_high - (swing_high - swing_low) * 0.618,
                0.786: swing_high - (swing_high - swing_low) * 0.786
            }
            
            current_price = close_prices[-1]
            
            # Check for price near Fibonacci levels
            for fib_ratio, fib_price in fib_levels.items():
                distance = abs(current_price - fib_price) / current_price
                
                if distance <= 0.01:  # Within 1% of Fibonacci level
                    # Determine direction based on trend
                    recent_trend = (close_prices[-1] - close_prices[-10]) / close_prices[-10]
                    
                    if recent_trend > 0:  # Uptrend - expect bounce from support
                        direction = "bullish"
                        target_price = current_price * 1.03
                        stop_loss = fib_price * 0.99
                    else:  # Downtrend - expect rejection at resistance
                        direction = "bearish"
                        target_price = current_price * 0.97
                        stop_loss = fib_price * 1.01
                    
                    confidence = min(0.75, (1 - distance * 100) * 0.5 + fib_ratio * 0.5)
                    
                    patterns.append(PatternSignal(
                        pattern_name=f"Fibonacci {fib_ratio:.1%} Level",
                        pattern_type=PatternType.SUPPORT_RESISTANCE,
                        strength=PatternStrength(min(4, int(fib_ratio * 4) + 1)),
                        confidence=confidence,
                        direction=direction,
                        entry_price=current_price,
                        target_price=target_price,
                        stop_loss=stop_loss,
                        probability=confidence,
                        time_horizon=48,
                        description=f"Price near {fib_ratio:.1%} Fibonacci level",
                        key_levels={
                            "fibonacci_level": fib_price,
                            "swing_high": swing_high,
                            "swing_low": swing_low
                        },
                        pattern_data={
                            "fib_ratio": fib_ratio,
                            "distance_to_level": distance,
                            "recent_trend": recent_trend
                        },
                        risk_reward_ratio=abs(target_price - current_price) / abs(stop_loss - current_price)
                    ))
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error in Fibonacci analysis: {e}")
            return []
    
    # Helper methods
    def _calculate_trend(self, prices: np.ndarray) -> float:
        """Calculate trend slope"""
        if len(prices) < 2:
            return 0
        
        x = np.arange(len(prices))
        slope, _ = np.polyfit(x, prices, 1)
        return slope / prices[-1]  # Normalize by current price
    
    def _calculate_trend_line(self, price_data: pd.DataFrame) -> float:
        """Calculate trend line value"""
        close_prices = price_data['close'].values
        x = np.arange(len(close_prices))
        slope, intercept = np.polyfit(x, close_prices, 1)
        return slope * (len(close_prices) - 1) + intercept
    
    def _check_volume_confirmation(self, volume_data: pd.DataFrame, direction: str) -> float:
        """Check volume confirmation for trend"""
        volume = volume_data['volume'].values
        if len(volume) < 10:
            return 0.5
        
        recent_volume = np.mean(volume[-5:])
        avg_volume = np.mean(volume[-20:-5])
        
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
        return min(1.0, volume_ratio / 2)  # Normalize to 0-1
    
    def _find_resistance_levels(self, high_prices: np.ndarray, volume: np.ndarray) -> List[Tuple[float, float]]:
        """Find resistance levels with strength"""
        peaks = signal.find_peaks(high_prices, distance=5)[0]
        
        levels = []
        for peak in peaks:
            price = high_prices[peak]
            
            # Count touches near this level
            touches = np.sum(np.abs(high_prices - price) / price < 0.01)
            
            # Volume at this level
            vol_strength = volume[peak] / np.mean(volume) if np.mean(volume) > 0 else 1
            
            # Combined strength
            strength = min(1.0, (touches * 0.3 + vol_strength * 0.7) / 5)
            
            if strength > 0.3:  # Minimum strength threshold
                levels.append((price, strength))
        
        # Remove duplicate levels
        levels = self._remove_duplicate_levels(levels)
        
        return sorted(levels, key=lambda x: x[1], reverse=True)[:5]  # Top 5 levels
    
    def _find_support_levels(self, low_prices: np.ndarray, volume: np.ndarray) -> List[Tuple[float, float]]:
        """Find support levels with strength"""
        troughs = signal.find_peaks(-low_prices, distance=5)[0]
        
        levels = []
        for trough in troughs:
            price = low_prices[trough]
            
            # Count touches near this level
            touches = np.sum(np.abs(low_prices - price) / price < 0.01)
            
            # Volume at this level
            vol_strength = volume[trough] / np.mean(volume) if np.mean(volume) > 0 else 1
            
            # Combined strength
            strength = min(1.0, (touches * 0.3 + vol_strength * 0.7) / 5)
            
            if strength > 0.3:  # Minimum strength threshold
                levels.append((price, strength))
        
        # Remove duplicate levels
        levels = self._remove_duplicate_levels(levels)
        
        return sorted(levels, key=lambda x: x[1], reverse=True)[:5]  # Top 5 levels
    
    def _count_level_touches(self, prices: np.ndarray, level: float) -> int:
        """Count how many times price touched a level"""
        tolerance = 0.01  # 1% tolerance
        return np.sum(np.abs(prices - level) / level < tolerance)
    
    def _remove_duplicate_levels(self, levels: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Remove duplicate levels within 1% of each other"""
        if not levels:
            return levels
        
        unique_levels = []
        for price, strength in levels:
            is_duplicate = False
            for existing_price, existing_strength in unique_levels:
                if abs(price - existing_price) / existing_price < 0.01:
                    is_duplicate = True
                    # Keep the stronger level
                    if strength > existing_strength:
                        unique_levels.remove((existing_price, existing_strength))
                        unique_levels.append((price, strength))
                    break
            
            if not is_duplicate:
                unique_levels.append((price, strength))
        
        return unique_levels
    
    def _detect_double_patterns(self, price_data: pd.DataFrame, volume_data: pd.DataFrame) -> List[PatternSignal]:
        """Detect double top/bottom patterns"""
        # Implementation for double top/bottom detection
        return []
    
    def _detect_head_shoulders(self, price_data: pd.DataFrame, volume_data: pd.DataFrame) -> List[PatternSignal]:
        """Detect head and shoulders patterns"""
        # Implementation for head and shoulders detection
        return []
    
    def _detect_price_divergence(self, price_data: pd.DataFrame, indicators: Dict[str, Any]) -> List[PatternSignal]:
        """Detect price divergence patterns"""
        # Implementation for price divergence detection
        return []
    
    def _find_swing_points(self, prices: np.ndarray, point_type: str) -> List[Tuple[int, float]]:
        """Find swing high/low points"""
        if point_type == 'high':
            peaks = signal.find_peaks(prices, distance=5)[0]
            return [(i, prices[i]) for i in peaks]
        else:
            troughs = signal.find_peaks(-prices, distance=5)[0]
            return [(i, prices[i]) for i in troughs]
    
    def _is_valid_abcd_pattern(self, points: List[Tuple[int, float, str]]) -> bool:
        """Check if points form a valid ABCD pattern"""
        if len(points) != 5:
            return False
        
        # Check alternating high/low pattern
        types = [point[2] for point in points]
        return (types == ['high', 'low', 'high', 'low', 'high'] or 
                types == ['low', 'high', 'low', 'high', 'low'])
    
    def _analyze_abcd_pattern(self, points: List[Tuple[int, float, str]], current_price: float) -> Optional[PatternSignal]:
        """Analyze ABCD pattern"""
        # Implementation for ABCD pattern analysis
        return None
    
    def _detect_gartley_patterns(self, swings: List[Tuple[int, float, str]], current_price: float) -> List[PatternSignal]:
        """Detect Gartley patterns"""
        # Implementation for Gartley pattern detection
        return []
    
    def _create_candlestick_signal(self, pattern_name: str, direction: str, 
                                 current_price: float, confidence: float, time_horizon: int) -> PatternSignal:
        """Create candlestick pattern signal"""
        if direction == "bullish":
            target_price = current_price * 1.03
            stop_loss = current_price * 0.98
        elif direction == "bearish":
            target_price = current_price * 0.97
            stop_loss = current_price * 1.02
        else:  # neutral
            target_price = current_price
            stop_loss = current_price
        
        return PatternSignal(
            pattern_name=pattern_name,
            pattern_type=PatternType.CANDLESTICK,
            strength=PatternStrength(min(4, int(confidence * 4) + 1)),
            confidence=confidence,
            direction=direction,
            entry_price=current_price,
            target_price=target_price,
            stop_loss=stop_loss,
            probability=confidence,
            time_horizon=time_horizon,
            description=f"{pattern_name} candlestick pattern",
            key_levels={},
            pattern_data={"pattern_type": "candlestick"},
            risk_reward_ratio=abs(target_price - current_price) / abs(stop_loss - current_price) if stop_loss != current_price else 1.0
        )
    
    def _find_peaks(self, data: np.ndarray) -> List[Tuple[int, float]]:
        """Find peaks in data"""
        peaks = signal.find_peaks(data, distance=3)[0]
        return [(i, data[i]) for i in peaks]

# Factory function
def create_pattern_engine() -> PatternRecognitionEngine:
    """Create pattern recognition engine"""
    return PatternRecognitionEngine()

if __name__ == "__main__":
    # Example usage
    engine = create_pattern_engine()
    
    # Mock data for testing
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    # Generate sample price data
    price_data = pd.DataFrame({
        'open': 100 + np.cumsum(np.random.randn(100) * 0.5),
        'high': 100 + np.cumsum(np.random.randn(100) * 0.5) + np.random.rand(100) * 2,
        'low': 100 + np.cumsum(np.random.randn(100) * 0.5) - np.random.rand(100) * 2,
        'close': 100 + np.cumsum(np.random.randn(100) * 0.5)
    }, index=dates)
    
    # Ensure high >= close >= low and high >= open >= low
    price_data['high'] = np.maximum(price_data['high'], np.maximum(price_data['open'], price_data['close']))
    price_data['low'] = np.minimum(price_data['low'], np.minimum(price_data['open'], price_data['close']))
    
    volume_data = pd.DataFrame({
        'volume': np.random.randint(100000, 1000000, 100)
    }, index=dates)
    
    indicators = {
        'rsi': np.random.rand(100) * 100,
        'macd': {
            'macd': np.random.randn(100) * 0.5,
            'signal': np.random.randn(100) * 0.3
        }
    }
    
    patterns = engine.analyze_patterns("AAPL", price_data, volume_data, indicators)
    
    print(f"Detected {len(patterns)} patterns:")
    for pattern in patterns:
        print(f"- {pattern.pattern_name}: {pattern.confidence:.2f} confidence, {pattern.direction} direction")

