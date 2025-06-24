"""
Signal Generation Service for AI Options Trading System
Core service for generating, managing, and distributing trading signals
"""

from flask import Flask, request, jsonify, g
from flask_cors import CORS
import logging
import time
import json
import asyncio
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
import psycopg2
from psycopg2.extras import RealDictCursor
import redis
import numpy as np
import pandas as pd
import requests
import os
from enum import Enum
import uuid
import websocket
import queue

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SignalType(Enum):
    """Signal types for options trading"""
    BUY_CALL = "BUY_CALL"
    BUY_PUT = "BUY_PUT"
    SELL_CALL = "SELL_CALL"
    SELL_PUT = "SELL_PUT"
    STRADDLE = "STRADDLE"
    STRANGLE = "STRANGLE"
    IRON_CONDOR = "IRON_CONDOR"
    BUTTERFLY = "BUTTERFLY"
    HOLD = "HOLD"

class SignalPriority(Enum):
    """Signal priority levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class TradingSignal:
    """Trading signal data structure"""
    id: str
    symbol: str
    signal_type: SignalType
    confidence: float
    priority: SignalPriority
    target_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    expiration: datetime
    reasoning: str
    technical_indicators: Dict[str, Any]
    options_data: Dict[str, Any]
    risk_metrics: Dict[str, Any]
    created_at: datetime
    expires_at: datetime
    model_version: str = "1.0.0"
    strategy_name: str = ""
    market_conditions: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to dictionary"""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "signal_type": self.signal_type.value,
            "confidence": self.confidence,
            "priority": self.priority.value,
            "target_price": self.target_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "position_size": self.position_size,
            "expiration": self.expiration.isoformat(),
            "reasoning": self.reasoning,
            "technical_indicators": self.technical_indicators,
            "options_data": self.options_data,
            "risk_metrics": self.risk_metrics,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "model_version": self.model_version,
            "strategy_name": self.strategy_name,
            "market_conditions": self.market_conditions or {}
        }

@dataclass
class SignalConfig:
    """Configuration for signal generation service"""
    # Database settings
    db_host: str = os.getenv("DB_HOST", "localhost")
    db_port: int = int(os.getenv("DB_PORT", "5432"))
    db_name: str = os.getenv("DB_NAME", "trading_db")
    db_user: str = os.getenv("DB_USER", "trading_admin")
    db_password: str = os.getenv("DB_PASSWORD", "")
    
    # Redis settings
    redis_host: str = os.getenv("REDIS_HOST", "localhost")
    redis_port: int = int(os.getenv("REDIS_PORT", "6379"))
    redis_db: int = int(os.getenv("REDIS_DB", "0"))
    redis_password: str = os.getenv("REDIS_PASSWORD", "")
    
    # Service settings
    service_host: str = os.getenv("SIGNALS_SERVICE_HOST", "0.0.0.0")
    service_port: int = int(os.getenv("SIGNALS_SERVICE_PORT", "8004"))
    
    # Signal generation settings
    min_confidence: float = float(os.getenv("MIN_SIGNAL_CONFIDENCE", "0.6"))
    max_signals_per_symbol: int = int(os.getenv("MAX_SIGNALS_PER_SYMBOL", "3"))
    signal_expiry_hours: int = int(os.getenv("SIGNAL_EXPIRY_HOURS", "24"))
    generation_interval: int = int(os.getenv("SIGNAL_GENERATION_INTERVAL", "60"))
    
    # External service URLs
    analytics_service_url: str = os.getenv("ANALYTICS_SERVICE_URL", "http://localhost:8003")
    data_service_url: str = os.getenv("DATA_SERVICE_URL", "http://localhost:8002")
    cache_service_url: str = os.getenv("CACHE_SERVICE_URL", "http://localhost:8001")
    
    # Target symbols
    target_symbols: List[str] = None
    
    def __post_init__(self):
        if self.target_symbols is None:
            self.target_symbols = [
                "AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA", "META",
                "SPY", "QQQ"
            ]

class SignalGenerator:
    """Core signal generation engine"""
    
    def __init__(self, config: SignalConfig):
        self.config = config
        self.active_signals = {}
        self.signal_history = []
        self.generation_stats = {
            "signals_generated": 0,
            "signals_executed": 0,
            "total_profit": 0.0,
            "win_rate": 0.0,
            "last_generation": None
        }
    
    def generate_signal(self, symbol: str, market_data: Dict[str, Any], 
                       analytics_data: Dict[str, Any]) -> Optional[TradingSignal]:
        """Generate a trading signal for a symbol"""
        try:
            # Extract key data
            current_price = market_data.get("current_price", 0)
            volume = market_data.get("volume", 0)
            
            # Get technical indicators
            indicators = analytics_data.get("indicators", {})
            patterns = analytics_data.get("patterns", {})
            
            # Analyze market conditions
            market_conditions = self._analyze_market_conditions(market_data, analytics_data)
            
            # Generate signal based on strategy
            signal_data = self._apply_trading_strategies(symbol, current_price, indicators, patterns, market_conditions)
            
            if not signal_data:
                return None
            
            # Create signal object
            signal = TradingSignal(
                id=str(uuid.uuid4()),
                symbol=symbol,
                signal_type=SignalType(signal_data["signal_type"]),
                confidence=signal_data["confidence"],
                priority=SignalPriority(signal_data["priority"]),
                target_price=signal_data["target_price"],
                stop_loss=signal_data["stop_loss"],
                take_profit=signal_data["take_profit"],
                position_size=signal_data["position_size"],
                expiration=signal_data["expiration"],
                reasoning=signal_data["reasoning"],
                technical_indicators=indicators,
                options_data=signal_data.get("options_data", {}),
                risk_metrics=signal_data.get("risk_metrics", {}),
                created_at=datetime.now(timezone.utc),
                expires_at=datetime.now(timezone.utc) + timedelta(hours=self.config.signal_expiry_hours),
                strategy_name=signal_data.get("strategy_name", ""),
                market_conditions=market_conditions
            )
            
            # Validate signal
            if self._validate_signal(signal):
                self.generation_stats["signals_generated"] += 1
                self.generation_stats["last_generation"] = datetime.now(timezone.utc).isoformat()
                return signal
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return None
    
    def _analyze_market_conditions(self, market_data: Dict[str, Any], 
                                 analytics_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze current market conditions"""
        conditions = {
            "volatility": "normal",
            "trend": "neutral",
            "momentum": "neutral",
            "volume": "normal",
            "market_sentiment": "neutral"
        }
        
        try:
            # Analyze volatility
            atr = analytics_data.get("indicators", {}).get("atr")
            if atr:
                if atr > 2.0:
                    conditions["volatility"] = "high"
                elif atr < 0.5:
                    conditions["volatility"] = "low"
            
            # Analyze trend
            trend_data = analytics_data.get("patterns", {}).get("trend", {})
            if trend_data:
                conditions["trend"] = trend_data.get("trend", "neutral")
            
            # Analyze momentum
            rsi = analytics_data.get("indicators", {}).get("rsi")
            if rsi:
                if rsi > 70:
                    conditions["momentum"] = "overbought"
                elif rsi < 30:
                    conditions["momentum"] = "oversold"
            
            # Analyze volume
            current_volume = market_data.get("volume", 0)
            avg_volume = market_data.get("avg_volume", current_volume)
            if avg_volume > 0:
                volume_ratio = current_volume / avg_volume
                if volume_ratio > 1.5:
                    conditions["volume"] = "high"
                elif volume_ratio < 0.5:
                    conditions["volume"] = "low"
            
        except Exception as e:
            logger.warning(f"Error analyzing market conditions: {e}")
        
        return conditions
    
    def _apply_trading_strategies(self, symbol: str, current_price: float,
                                indicators: Dict[str, Any], patterns: Dict[str, Any],
                                market_conditions: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Apply trading strategies to generate signals"""
        
        # Strategy 1: RSI Reversal Strategy
        rsi_signal = self._rsi_reversal_strategy(symbol, current_price, indicators, market_conditions)
        if rsi_signal:
            return rsi_signal
        
        # Strategy 2: Breakout Strategy
        breakout_signal = self._breakout_strategy(symbol, current_price, patterns, market_conditions)
        if breakout_signal:
            return breakout_signal
        
        # Strategy 3: Mean Reversion Strategy
        mean_reversion_signal = self._mean_reversion_strategy(symbol, current_price, indicators, market_conditions)
        if mean_reversion_signal:
            return mean_reversion_signal
        
        # Strategy 4: Momentum Strategy
        momentum_signal = self._momentum_strategy(symbol, current_price, indicators, market_conditions)
        if momentum_signal:
            return momentum_signal
        
        return None
    
    def _rsi_reversal_strategy(self, symbol: str, current_price: float,
                             indicators: Dict[str, Any], market_conditions: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """RSI-based reversal strategy for options trading"""
        rsi = indicators.get("rsi")
        if not rsi:
            return None
        
        # Look for oversold conditions for call options
        if rsi < 30 and market_conditions.get("trend") != "downtrend":
            confidence = min(0.9, (30 - rsi) / 30 + 0.6)
            
            return {
                "signal_type": "BUY_CALL",
                "confidence": confidence,
                "priority": 3 if confidence > 0.8 else 2,
                "target_price": current_price * 1.05,  # 5% target
                "stop_loss": current_price * 0.98,     # 2% stop loss
                "take_profit": current_price * 1.08,   # 8% take profit
                "position_size": min(0.02, confidence * 0.03),  # Risk-adjusted position size
                "expiration": datetime.now(timezone.utc) + timedelta(days=7),
                "reasoning": f"RSI oversold at {rsi:.2f}, expecting reversal",
                "strategy_name": "RSI_Reversal_Call",
                "risk_metrics": {
                    "max_loss": 0.02,
                    "reward_risk_ratio": 4.0,
                    "probability_of_profit": confidence
                }
            }
        
        # Look for overbought conditions for put options
        elif rsi > 70 and market_conditions.get("trend") != "uptrend":
            confidence = min(0.9, (rsi - 70) / 30 + 0.6)
            
            return {
                "signal_type": "BUY_PUT",
                "confidence": confidence,
                "priority": 3 if confidence > 0.8 else 2,
                "target_price": current_price * 0.95,  # 5% target down
                "stop_loss": current_price * 1.02,     # 2% stop loss
                "take_profit": current_price * 0.92,   # 8% take profit
                "position_size": min(0.02, confidence * 0.03),
                "expiration": datetime.now(timezone.utc) + timedelta(days=7),
                "reasoning": f"RSI overbought at {rsi:.2f}, expecting reversal",
                "strategy_name": "RSI_Reversal_Put",
                "risk_metrics": {
                    "max_loss": 0.02,
                    "reward_risk_ratio": 4.0,
                    "probability_of_profit": confidence
                }
            }
        
        return None
    
    def _breakout_strategy(self, symbol: str, current_price: float,
                         patterns: Dict[str, Any], market_conditions: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Breakout strategy for options trading"""
        breakout_data = patterns.get("breakout", {})
        if not breakout_data.get("breakout"):
            return None
        
        direction = breakout_data.get("direction")
        strength = breakout_data.get("strength", 0)
        volume_confirmation = breakout_data.get("volume_confirmation", False)
        
        if not volume_confirmation or strength < 0.02:  # Minimum 2% breakout
            return None
        
        confidence = min(0.9, strength * 10 + 0.5)  # Scale strength to confidence
        
        if direction == "upward":
            return {
                "signal_type": "BUY_CALL",
                "confidence": confidence,
                "priority": 4 if confidence > 0.8 else 3,
                "target_price": current_price * (1 + strength * 2),
                "stop_loss": current_price * 0.97,
                "take_profit": current_price * (1 + strength * 3),
                "position_size": min(0.03, confidence * 0.04),
                "expiration": datetime.now(timezone.utc) + timedelta(days=5),
                "reasoning": f"Upward breakout detected with {strength:.1%} strength",
                "strategy_name": "Breakout_Call",
                "risk_metrics": {
                    "max_loss": 0.03,
                    "reward_risk_ratio": strength * 100,
                    "probability_of_profit": confidence
                }
            }
        
        elif direction == "downward":
            return {
                "signal_type": "BUY_PUT",
                "confidence": confidence,
                "priority": 4 if confidence > 0.8 else 3,
                "target_price": current_price * (1 - strength * 2),
                "stop_loss": current_price * 1.03,
                "take_profit": current_price * (1 - strength * 3),
                "position_size": min(0.03, confidence * 0.04),
                "expiration": datetime.now(timezone.utc) + timedelta(days=5),
                "reasoning": f"Downward breakout detected with {strength:.1%} strength",
                "strategy_name": "Breakout_Put",
                "risk_metrics": {
                    "max_loss": 0.03,
                    "reward_risk_ratio": strength * 100,
                    "probability_of_profit": confidence
                }
            }
        
        return None
    
    def _mean_reversion_strategy(self, symbol: str, current_price: float,
                               indicators: Dict[str, Any], market_conditions: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Mean reversion strategy using Bollinger Bands"""
        bollinger = indicators.get("bollinger_bands", {})
        if not bollinger:
            return None
        
        upper_band = bollinger.get("upper")
        lower_band = bollinger.get("lower")
        middle_band = bollinger.get("middle")
        
        if not all([upper_band, lower_band, middle_band]):
            return None
        
        # Price near lower band - expect reversion up
        if current_price <= lower_band * 1.01:  # Within 1% of lower band
            distance_from_band = (lower_band - current_price) / current_price
            confidence = min(0.85, distance_from_band * 20 + 0.6)
            
            return {
                "signal_type": "BUY_CALL",
                "confidence": confidence,
                "priority": 2,
                "target_price": middle_band,
                "stop_loss": current_price * 0.95,
                "take_profit": upper_band * 0.95,
                "position_size": min(0.015, confidence * 0.02),
                "expiration": datetime.now(timezone.utc) + timedelta(days=10),
                "reasoning": f"Price near lower Bollinger Band, expecting mean reversion",
                "strategy_name": "Mean_Reversion_Call",
                "risk_metrics": {
                    "max_loss": 0.05,
                    "reward_risk_ratio": 3.0,
                    "probability_of_profit": confidence
                }
            }
        
        # Price near upper band - expect reversion down
        elif current_price >= upper_band * 0.99:  # Within 1% of upper band
            distance_from_band = (current_price - upper_band) / current_price
            confidence = min(0.85, distance_from_band * 20 + 0.6)
            
            return {
                "signal_type": "BUY_PUT",
                "confidence": confidence,
                "priority": 2,
                "target_price": middle_band,
                "stop_loss": current_price * 1.05,
                "take_profit": lower_band * 1.05,
                "position_size": min(0.015, confidence * 0.02),
                "expiration": datetime.now(timezone.utc) + timedelta(days=10),
                "reasoning": f"Price near upper Bollinger Band, expecting mean reversion",
                "strategy_name": "Mean_Reversion_Put",
                "risk_metrics": {
                    "max_loss": 0.05,
                    "reward_risk_ratio": 3.0,
                    "probability_of_profit": confidence
                }
            }
        
        return None
    
    def _momentum_strategy(self, symbol: str, current_price: float,
                         indicators: Dict[str, Any], market_conditions: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Momentum strategy using MACD"""
        macd_data = indicators.get("macd", {})
        if not macd_data:
            return None
        
        macd = macd_data.get("macd")
        signal = macd_data.get("signal")
        histogram = macd_data.get("histogram")
        
        if not all([macd, signal, histogram]):
            return None
        
        # MACD bullish crossover
        if macd > signal and histogram > 0 and market_conditions.get("volume") == "high":
            confidence = min(0.8, abs(histogram) * 100 + 0.6)
            
            return {
                "signal_type": "BUY_CALL",
                "confidence": confidence,
                "priority": 3,
                "target_price": current_price * 1.06,
                "stop_loss": current_price * 0.97,
                "take_profit": current_price * 1.10,
                "position_size": min(0.025, confidence * 0.03),
                "expiration": datetime.now(timezone.utc) + timedelta(days=7),
                "reasoning": f"MACD bullish crossover with strong momentum",
                "strategy_name": "MACD_Momentum_Call",
                "risk_metrics": {
                    "max_loss": 0.03,
                    "reward_risk_ratio": 3.33,
                    "probability_of_profit": confidence
                }
            }
        
        # MACD bearish crossover
        elif macd < signal and histogram < 0 and market_conditions.get("volume") == "high":
            confidence = min(0.8, abs(histogram) * 100 + 0.6)
            
            return {
                "signal_type": "BUY_PUT",
                "confidence": confidence,
                "priority": 3,
                "target_price": current_price * 0.94,
                "stop_loss": current_price * 1.03,
                "take_profit": current_price * 0.90,
                "position_size": min(0.025, confidence * 0.03),
                "expiration": datetime.now(timezone.utc) + timedelta(days=7),
                "reasoning": f"MACD bearish crossover with strong momentum",
                "strategy_name": "MACD_Momentum_Put",
                "risk_metrics": {
                    "max_loss": 0.03,
                    "reward_risk_ratio": 3.33,
                    "probability_of_profit": confidence
                }
            }
        
        return None
    
    def _validate_signal(self, signal: TradingSignal) -> bool:
        """Validate signal meets quality criteria"""
        try:
            # Check minimum confidence
            if signal.confidence < self.config.min_confidence:
                return False
            
            # Check risk/reward ratio
            if signal.signal_type in [SignalType.BUY_CALL, SignalType.BUY_PUT]:
                risk = abs(signal.target_price - signal.stop_loss) / signal.target_price
                reward = abs(signal.take_profit - signal.target_price) / signal.target_price
                
                if reward / risk < 2.0:  # Minimum 2:1 reward/risk ratio
                    return False
            
            # Check position size is reasonable
            if signal.position_size > 0.05:  # Maximum 5% position size
                return False
            
            # Check expiration is reasonable
            time_to_expiry = (signal.expiration - datetime.now(timezone.utc)).total_seconds()
            if time_to_expiry < 3600 or time_to_expiry > 30 * 24 * 3600:  # 1 hour to 30 days
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating signal: {e}")
            return False

class SignalService:
    """Main signal generation service"""
    
    def __init__(self, config: SignalConfig = None):
        self.config = config or SignalConfig()
        self.app = Flask(__name__)
        CORS(self.app, origins="*")
        
        # Initialize components
        self.db_pool = None
        self.redis_client = None
        self.signal_generator = SignalGenerator(self.config)
        self.active_signals = {}
        self.signal_queue = queue.Queue()
        
        # Background processing
        self.executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="signals")
        self.generation_active = False
        
        # Initialize connections
        self._init_database()
        self._init_redis()
        self._setup_routes()
        
        # Start background signal generation
        self._start_signal_generation()
    
    def _init_database(self):
        """Initialize database connection"""
        try:
            self.db_pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=20,
                host=self.config.db_host,
                port=self.config.db_port,
                database=self.config.db_name,
                user=self.config.db_user,
                password=self.config.db_password
            )
            logger.info("Database connection pool initialized")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            self.db_pool = None
    
    def _init_redis(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                password=self.config.redis_password if self.config.redis_password else None,
                decode_responses=True
            )
            self.redis_client.ping()
            logger.info("Redis connection initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {e}")
            self.redis_client = None
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.before_request
        def before_request():
            g.start_time = time.time()
            g.request_id = request.headers.get('X-Request-ID', f"req_{int(time.time() * 1000)}")
        
        @self.app.after_request
        def after_request(response):
            duration = time.time() - g.start_time
            response.headers['X-Request-ID'] = g.request_id
            response.headers['X-Response-Time'] = f"{duration * 1000:.2f}ms"
            return response
        
        @self.app.errorhandler(Exception)
        def handle_exception(e):
            logger.error(f"Unhandled exception: {str(e)}")
            return jsonify({
                "error": "Internal server error",
                "request_id": g.request_id,
                "timestamp": datetime.now().isoformat()
            }), 500
        
        # Health check
        @self.app.route('/health', methods=['GET'])
        def health_check():
            health_status = {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "service": "signals",
                "version": "1.0.0",
                "components": {
                    "database": "healthy" if self.db_pool else "unhealthy",
                    "redis": "healthy" if self.redis_client else "unhealthy",
                    "signal_generation": "active" if self.generation_active else "inactive"
                },
                "stats": self.signal_generator.generation_stats
            }
            
            if not all(status == "healthy" for status in health_status["components"].values() if status != "active"):
                health_status["status"] = "degraded"
            
            return jsonify(health_status)
        
        # Signal generation endpoints
        @self.app.route('/api/v1/signals/generate', methods=['POST'])
        def generate_signals():
            """Manually trigger signal generation"""
            try:
                data = request.get_json() or {}
                symbols = data.get('symbols', self.config.target_symbols)
                
                # Generate signals for specified symbols
                generated_signals = []
                for symbol in symbols:
                    signal = self._generate_signal_for_symbol(symbol)
                    if signal:
                        generated_signals.append(signal.to_dict())
                
                return jsonify({
                    "success": True,
                    "signals_generated": len(generated_signals),
                    "signals": generated_signals,
                    "timestamp": datetime.now().isoformat()
                })
            
            except Exception as e:
                logger.error(f"Failed to generate signals: {e}")
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 500
        
        @self.app.route('/api/v1/signals/active', methods=['GET'])
        def get_active_signals():
            """Get all active signals"""
            try:
                symbol = request.args.get('symbol')
                priority = request.args.get('priority')
                limit = int(request.args.get('limit', 50))
                
                signals = self._get_active_signals(symbol, priority, limit)
                
                return jsonify({
                    "success": True,
                    "count": len(signals),
                    "signals": signals,
                    "timestamp": datetime.now().isoformat()
                })
            
            except Exception as e:
                logger.error(f"Failed to get active signals: {e}")
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 500
        
        @self.app.route('/api/v1/signals/<signal_id>', methods=['GET'])
        def get_signal(signal_id):
            """Get specific signal by ID"""
            try:
                signal = self._get_signal_by_id(signal_id)
                
                if signal:
                    return jsonify({
                        "success": True,
                        "signal": signal,
                        "timestamp": datetime.now().isoformat()
                    })
                else:
                    return jsonify({
                        "success": False,
                        "error": "Signal not found"
                    }), 404
            
            except Exception as e:
                logger.error(f"Failed to get signal {signal_id}: {e}")
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 500
        
        @self.app.route('/api/v1/signals/<signal_id>/execute', methods=['POST'])
        def execute_signal(signal_id):
            """Mark signal as executed"""
            try:
                data = request.get_json() or {}
                execution_price = data.get('execution_price')
                execution_time = data.get('execution_time', datetime.now().isoformat())
                
                success = self._execute_signal(signal_id, execution_price, execution_time)
                
                if success:
                    return jsonify({
                        "success": True,
                        "message": "Signal marked as executed",
                        "timestamp": datetime.now().isoformat()
                    })
                else:
                    return jsonify({
                        "success": False,
                        "error": "Failed to execute signal"
                    }), 400
            
            except Exception as e:
                logger.error(f"Failed to execute signal {signal_id}: {e}")
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 500
        
        @self.app.route('/api/v1/signals/stats', methods=['GET'])
        def get_signal_stats():
            """Get signal generation statistics"""
            try:
                stats = self._get_signal_statistics()
                
                return jsonify({
                    "success": True,
                    "stats": stats,
                    "timestamp": datetime.now().isoformat()
                })
            
            except Exception as e:
                logger.error(f"Failed to get signal stats: {e}")
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 500
        
        @self.app.route('/api/v1/signals/performance', methods=['GET'])
        def get_signal_performance():
            """Get signal performance metrics"""
            try:
                days = int(request.args.get('days', 30))
                symbol = request.args.get('symbol')
                
                performance = self._get_signal_performance(days, symbol)
                
                return jsonify({
                    "success": True,
                    "performance": performance,
                    "timestamp": datetime.now().isoformat()
                })
            
            except Exception as e:
                logger.error(f"Failed to get signal performance: {e}")
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 500
    
    def _generate_signal_for_symbol(self, symbol: str) -> Optional[TradingSignal]:
        """Generate signal for a specific symbol"""
        try:
            # Get market data
            market_data = self._get_market_data(symbol)
            if not market_data:
                return None
            
            # Get analytics data
            analytics_data = self._get_analytics_data(symbol)
            if not analytics_data:
                return None
            
            # Generate signal
            signal = self.signal_generator.generate_signal(symbol, market_data, analytics_data)
            
            if signal:
                # Store signal
                self._store_signal(signal)
                
                # Cache signal
                self._cache_signal(signal)
                
                # Add to active signals
                self.active_signals[signal.id] = signal
                
                logger.info(f"Generated signal {signal.id} for {symbol}: {signal.signal_type.value}")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return None
    
    def _get_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current market data for symbol"""
        try:
            # Try cache first
            if self.redis_client:
                cached_data = self.redis_client.get(f"market_data:{symbol}")
                if cached_data:
                    return json.loads(cached_data)
            
            # Get from data service
            response = requests.get(
                f"{self.config.data_service_url}/api/v1/data/current/{symbol}",
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("data", {})
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            return None
    
    def _get_analytics_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get analytics data for symbol"""
        try:
            # Get technical indicators
            indicators_response = requests.get(
                f"{self.config.analytics_service_url}/api/v1/analytics/indicators/{symbol}",
                timeout=5
            )
            
            # Get patterns
            patterns_response = requests.get(
                f"{self.config.analytics_service_url}/api/v1/analytics/patterns/{symbol}",
                timeout=5
            )
            
            analytics_data = {}
            
            if indicators_response.status_code == 200:
                indicators_data = indicators_response.json()
                analytics_data["indicators"] = indicators_data.get("data", {})
            
            if patterns_response.status_code == 200:
                patterns_data = patterns_response.json()
                analytics_data["patterns"] = patterns_data.get("data", {})
            
            return analytics_data if analytics_data else None
            
        except Exception as e:
            logger.error(f"Error getting analytics data for {symbol}: {e}")
            return None
    
    def _store_signal(self, signal: TradingSignal):
        """Store signal in database"""
        if not self.db_pool:
            return
        
        conn = None
        try:
            conn = self.db_pool.getconn()
            cursor = conn.cursor()
            
            query = """
                INSERT INTO trading_signals (
                    id, symbol, signal_type, confidence, priority, target_price,
                    stop_loss, take_profit, position_size, expiration, reasoning,
                    technical_indicators, options_data, risk_metrics, created_at,
                    expires_at, model_version, strategy_name, market_conditions
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            cursor.execute(query, (
                signal.id, signal.symbol, signal.signal_type.value, signal.confidence,
                signal.priority.value, signal.target_price, signal.stop_loss,
                signal.take_profit, signal.position_size, signal.expiration,
                signal.reasoning, json.dumps(signal.technical_indicators),
                json.dumps(signal.options_data), json.dumps(signal.risk_metrics),
                signal.created_at, signal.expires_at, signal.model_version,
                signal.strategy_name, json.dumps(signal.market_conditions)
            ))
            
            conn.commit()
            
        except Exception as e:
            logger.error(f"Failed to store signal: {e}")
            if conn:
                conn.rollback()
        
        finally:
            if conn:
                self.db_pool.putconn(conn)
    
    def _cache_signal(self, signal: TradingSignal):
        """Cache signal in Redis"""
        if not self.redis_client:
            return
        
        try:
            # Cache individual signal
            self.redis_client.setex(
                f"signal:{signal.id}",
                3600,  # 1 hour TTL
                json.dumps(signal.to_dict(), default=str)
            )
            
            # Add to active signals list
            self.redis_client.zadd(
                f"active_signals:{signal.symbol}",
                {signal.id: signal.confidence}
            )
            
            # Keep only top signals per symbol
            self.redis_client.zremrangebyrank(
                f"active_signals:{signal.symbol}",
                0, -(self.config.max_signals_per_symbol + 1)
            )
            
        except Exception as e:
            logger.error(f"Failed to cache signal: {e}")
    
    def _start_signal_generation(self):
        """Start background signal generation"""
        def generation_loop():
            self.generation_active = True
            logger.info("Started background signal generation")
            
            while self.generation_active:
                try:
                    for symbol in self.config.target_symbols:
                        if not self.generation_active:
                            break
                        
                        # Check if we need more signals for this symbol
                        active_count = len(self._get_active_signals(symbol))
                        if active_count < self.config.max_signals_per_symbol:
                            signal = self._generate_signal_for_symbol(symbol)
                            if signal:
                                logger.info(f"Background generated signal for {symbol}")
                    
                    # Wait before next generation cycle
                    time.sleep(self.config.generation_interval)
                    
                except Exception as e:
                    logger.error(f"Error in signal generation loop: {e}")
                    time.sleep(30)  # Wait before retrying
            
            logger.info("Stopped background signal generation")
        
        # Start generation in background thread
        self.executor.submit(generation_loop)
    
    def _get_active_signals(self, symbol: str = None, priority: str = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Get active signals from database"""
        if not self.db_pool:
            return []
        
        conn = None
        try:
            conn = self.db_pool.getconn()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            query = """
                SELECT * FROM trading_signals
                WHERE expires_at > NOW() AND NOT executed
            """
            params = []
            
            if symbol:
                query += " AND symbol = %s"
                params.append(symbol)
            
            if priority:
                query += " AND priority = %s"
                params.append(int(priority))
            
            query += " ORDER BY confidence DESC, created_at DESC LIMIT %s"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            return [dict(row) for row in rows]
            
        except Exception as e:
            logger.error(f"Failed to get active signals: {e}")
            return []
        
        finally:
            if conn:
                self.db_pool.putconn(conn)
    
    def _get_signal_by_id(self, signal_id: str) -> Optional[Dict[str, Any]]:
        """Get signal by ID"""
        # Try cache first
        if self.redis_client:
            try:
                cached_signal = self.redis_client.get(f"signal:{signal_id}")
                if cached_signal:
                    return json.loads(cached_signal)
            except Exception as e:
                logger.warning(f"Cache error getting signal {signal_id}: {e}")
        
        # Get from database
        if not self.db_pool:
            return None
        
        conn = None
        try:
            conn = self.db_pool.getconn()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("SELECT * FROM trading_signals WHERE id = %s", (signal_id,))
            row = cursor.fetchone()
            
            return dict(row) if row else None
            
        except Exception as e:
            logger.error(f"Failed to get signal {signal_id}: {e}")
            return None
        
        finally:
            if conn:
                self.db_pool.putconn(conn)
    
    def _execute_signal(self, signal_id: str, execution_price: float, execution_time: str) -> bool:
        """Mark signal as executed"""
        if not self.db_pool:
            return False
        
        conn = None
        try:
            conn = self.db_pool.getconn()
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE trading_signals
                SET executed = TRUE, execution_price = %s, execution_time = %s
                WHERE id = %s
            """, (execution_price, execution_time, signal_id))
            
            conn.commit()
            
            # Remove from active signals cache
            if self.redis_client:
                self.redis_client.delete(f"signal:{signal_id}")
            
            return cursor.rowcount > 0
            
        except Exception as e:
            logger.error(f"Failed to execute signal {signal_id}: {e}")
            if conn:
                conn.rollback()
            return False
        
        finally:
            if conn:
                self.db_pool.putconn(conn)
    
    def _get_signal_statistics(self) -> Dict[str, Any]:
        """Get signal generation statistics"""
        stats = self.signal_generator.generation_stats.copy()
        
        # Add database stats
        if self.db_pool:
            conn = None
            try:
                conn = self.db_pool.getconn()
                cursor = conn.cursor()
                
                # Total signals
                cursor.execute("SELECT COUNT(*) FROM trading_signals")
                stats["total_signals"] = cursor.fetchone()[0]
                
                # Active signals
                cursor.execute("SELECT COUNT(*) FROM trading_signals WHERE expires_at > NOW() AND NOT executed")
                stats["active_signals"] = cursor.fetchone()[0]
                
                # Executed signals
                cursor.execute("SELECT COUNT(*) FROM trading_signals WHERE executed")
                stats["executed_signals"] = cursor.fetchone()[0]
                
                # Win rate calculation
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total,
                        SUM(CASE WHEN performance > 0 THEN 1 ELSE 0 END) as wins
                    FROM trading_signals 
                    WHERE executed AND performance IS NOT NULL
                """)
                result = cursor.fetchone()
                if result and result[0] > 0:
                    stats["win_rate"] = result[1] / result[0]
                
            except Exception as e:
                logger.error(f"Error getting signal statistics: {e}")
            
            finally:
                if conn:
                    self.db_pool.putconn(conn)
        
        return stats
    
    def _get_signal_performance(self, days: int, symbol: str = None) -> Dict[str, Any]:
        """Get signal performance metrics"""
        if not self.db_pool:
            return {}
        
        conn = None
        try:
            conn = self.db_pool.getconn()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            query = """
                SELECT 
                    symbol,
                    signal_type,
                    AVG(performance) as avg_performance,
                    COUNT(*) as total_signals,
                    SUM(CASE WHEN performance > 0 THEN 1 ELSE 0 END) as winning_signals,
                    MAX(performance) as best_performance,
                    MIN(performance) as worst_performance
                FROM trading_signals
                WHERE executed AND performance IS NOT NULL
                AND created_at >= NOW() - INTERVAL '%s days'
            """
            params = [days]
            
            if symbol:
                query += " AND symbol = %s"
                params.append(symbol)
            
            query += " GROUP BY symbol, signal_type ORDER BY avg_performance DESC"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            performance_data = [dict(row) for row in rows]
            
            # Calculate overall metrics
            if performance_data:
                total_signals = sum(row["total_signals"] for row in performance_data)
                total_wins = sum(row["winning_signals"] for row in performance_data)
                avg_performance = sum(row["avg_performance"] * row["total_signals"] for row in performance_data) / total_signals
                
                overall_metrics = {
                    "total_signals": total_signals,
                    "win_rate": total_wins / total_signals if total_signals > 0 else 0,
                    "average_performance": avg_performance,
                    "best_performance": max(row["best_performance"] for row in performance_data),
                    "worst_performance": min(row["worst_performance"] for row in performance_data)
                }
            else:
                overall_metrics = {
                    "total_signals": 0,
                    "win_rate": 0,
                    "average_performance": 0,
                    "best_performance": 0,
                    "worst_performance": 0
                }
            
            return {
                "overall": overall_metrics,
                "by_symbol_and_type": performance_data,
                "period_days": days
            }
            
        except Exception as e:
            logger.error(f"Failed to get signal performance: {e}")
            return {}
        
        finally:
            if conn:
                self.db_pool.putconn(conn)
    
    def run(self):
        """Run the signal generation service"""
        logger.info(f"Starting Signal Generation Service on {self.config.service_host}:{self.config.service_port}")
        self.app.run(
            host=self.config.service_host,
            port=self.config.service_port,
            debug=False,
            threaded=True
        )
    
    def stop(self):
        """Stop the signal generation service"""
        self.generation_active = False
        logger.info("Signal generation service stopped")

# Factory function
def create_signal_service(config: SignalConfig = None) -> SignalService:
    """Create signal generation service"""
    return SignalService(config)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Options Trading System Signal Generation Service")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8004, help="Port to bind to")
    parser.add_argument("--min-confidence", type=float, default=0.6, help="Minimum signal confidence")
    
    args = parser.parse_args()
    
    # Create configuration
    config = SignalConfig()
    config.service_host = args.host
    config.service_port = args.port
    config.min_confidence = args.min_confidence
    
    # Create and run service
    service = create_signal_service(config)
    
    try:
        service.run()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
        service.stop()

