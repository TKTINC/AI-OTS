"""
Analytics Service for AI Options Trading System
Provides market data analysis, technical indicators, and pattern recognition
"""

from flask import Flask, request, jsonify, g
from flask_cors import CORS
import logging
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import threading
from dataclasses import dataclass, asdict
import psycopg2
from psycopg2.extras import RealDictCursor
import redis
import os
from scipy import stats
import talib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AnalyticsConfig:
    """Configuration for analytics service"""
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
    service_host: str = os.getenv("ANALYTICS_SERVICE_HOST", "0.0.0.0")
    service_port: int = int(os.getenv("ANALYTICS_SERVICE_PORT", "8003"))
    
    # Analytics settings
    cache_ttl: int = int(os.getenv("ANALYTICS_CACHE_TTL", "300"))  # 5 minutes
    max_lookback_days: int = int(os.getenv("MAX_LOOKBACK_DAYS", "30"))
    min_data_points: int = int(os.getenv("MIN_DATA_POINTS", "20"))

class TechnicalIndicators:
    """Technical analysis indicators"""
    
    @staticmethod
    def sma(prices: np.ndarray, period: int) -> np.ndarray:
        """Simple Moving Average"""
        return talib.SMA(prices, timeperiod=period)
    
    @staticmethod
    def ema(prices: np.ndarray, period: int) -> np.ndarray:
        """Exponential Moving Average"""
        return talib.EMA(prices, timeperiod=period)
    
    @staticmethod
    def rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Relative Strength Index"""
        return talib.RSI(prices, timeperiod=period)
    
    @staticmethod
    def macd(prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """MACD indicator"""
        return talib.MACD(prices, fastperiod=fast, slowperiod=slow, signalperiod=signal)
    
    @staticmethod
    def bollinger_bands(prices: np.ndarray, period: int = 20, std_dev: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bollinger Bands"""
        return talib.BBANDS(prices, timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev)
    
    @staticmethod
    def stochastic(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                  k_period: int = 14, d_period: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """Stochastic Oscillator"""
        return talib.STOCH(high, low, close, fastk_period=k_period, slowk_period=d_period, slowd_period=d_period)
    
    @staticmethod
    def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """Average True Range"""
        return talib.ATR(high, low, close, timeperiod=period)
    
    @staticmethod
    def volume_profile(prices: np.ndarray, volumes: np.ndarray, bins: int = 20) -> Dict[str, Any]:
        """Volume Profile analysis"""
        price_min, price_max = np.min(prices), np.max(prices)
        price_bins = np.linspace(price_min, price_max, bins + 1)
        
        volume_profile = np.zeros(bins)
        
        for i in range(len(prices)):
            bin_idx = np.digitize(prices[i], price_bins) - 1
            bin_idx = max(0, min(bins - 1, bin_idx))
            volume_profile[bin_idx] += volumes[i]
        
        # Find Point of Control (POC) - price level with highest volume
        poc_idx = np.argmax(volume_profile)
        poc_price = (price_bins[poc_idx] + price_bins[poc_idx + 1]) / 2
        
        # Calculate Value Area (70% of volume)
        total_volume = np.sum(volume_profile)
        target_volume = total_volume * 0.7
        
        # Find value area around POC
        value_area_volume = volume_profile[poc_idx]
        value_area_low = poc_idx
        value_area_high = poc_idx
        
        while value_area_volume < target_volume and (value_area_low > 0 or value_area_high < bins - 1):
            left_volume = volume_profile[value_area_low - 1] if value_area_low > 0 else 0
            right_volume = volume_profile[value_area_high + 1] if value_area_high < bins - 1 else 0
            
            if left_volume >= right_volume and value_area_low > 0:
                value_area_low -= 1
                value_area_volume += left_volume
            elif value_area_high < bins - 1:
                value_area_high += 1
                value_area_volume += right_volume
            else:
                break
        
        return {
            "poc_price": poc_price,
            "value_area_low": (price_bins[value_area_low] + price_bins[value_area_low + 1]) / 2,
            "value_area_high": (price_bins[value_area_high] + price_bins[value_area_high + 1]) / 2,
            "volume_profile": volume_profile.tolist(),
            "price_bins": price_bins.tolist()
        }

class PatternRecognition:
    """Chart pattern recognition"""
    
    @staticmethod
    def detect_support_resistance(prices: np.ndarray, window: int = 20, min_touches: int = 2) -> Dict[str, List[float]]:
        """Detect support and resistance levels"""
        highs = []
        lows = []
        
        # Find local highs and lows
        for i in range(window, len(prices) - window):
            if all(prices[i] >= prices[i-j] for j in range(1, window+1)) and \
               all(prices[i] >= prices[i+j] for j in range(1, window+1)):
                highs.append(prices[i])
            
            if all(prices[i] <= prices[i-j] for j in range(1, window+1)) and \
               all(prices[i] <= prices[i+j] for j in range(1, window+1)):
                lows.append(prices[i])
        
        # Cluster similar levels
        def cluster_levels(levels, tolerance=0.02):
            if not levels:
                return []
            
            levels = sorted(levels)
            clusters = []
            current_cluster = [levels[0]]
            
            for level in levels[1:]:
                if abs(level - current_cluster[-1]) / current_cluster[-1] <= tolerance:
                    current_cluster.append(level)
                else:
                    if len(current_cluster) >= min_touches:
                        clusters.append(np.mean(current_cluster))
                    current_cluster = [level]
            
            if len(current_cluster) >= min_touches:
                clusters.append(np.mean(current_cluster))
            
            return clusters
        
        resistance_levels = cluster_levels(highs)
        support_levels = cluster_levels(lows)
        
        return {
            "support": support_levels,
            "resistance": resistance_levels
        }
    
    @staticmethod
    def detect_trend(prices: np.ndarray, window: int = 20) -> Dict[str, Any]:
        """Detect trend direction and strength"""
        if len(prices) < window:
            return {"trend": "unknown", "strength": 0, "slope": 0}
        
        # Linear regression on recent prices
        x = np.arange(len(prices))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x[-window:], prices[-window:])
        
        # Determine trend
        if slope > 0.001:
            trend = "uptrend"
        elif slope < -0.001:
            trend = "downtrend"
        else:
            trend = "sideways"
        
        # Trend strength based on R-squared
        strength = r_value ** 2
        
        return {
            "trend": trend,
            "strength": strength,
            "slope": slope,
            "r_squared": r_value ** 2,
            "p_value": p_value
        }
    
    @staticmethod
    def detect_breakout(prices: np.ndarray, volumes: np.ndarray, window: int = 20) -> Dict[str, Any]:
        """Detect price breakouts"""
        if len(prices) < window * 2:
            return {"breakout": False, "direction": None, "strength": 0}
        
        # Calculate recent high/low
        recent_high = np.max(prices[-window:])
        recent_low = np.min(prices[-window:])
        
        # Calculate historical range
        historical_high = np.max(prices[:-window])
        historical_low = np.min(prices[:-window])
        
        current_price = prices[-1]
        avg_volume = np.mean(volumes[-window:])
        historical_avg_volume = np.mean(volumes[:-window])
        
        # Check for breakout
        breakout = False
        direction = None
        strength = 0
        
        if current_price > historical_high and avg_volume > historical_avg_volume * 1.5:
            breakout = True
            direction = "upward"
            strength = (current_price - historical_high) / historical_high
        elif current_price < historical_low and avg_volume > historical_avg_volume * 1.5:
            breakout = True
            direction = "downward"
            strength = (historical_low - current_price) / historical_low
        
        return {
            "breakout": breakout,
            "direction": direction,
            "strength": strength,
            "volume_confirmation": avg_volume > historical_avg_volume * 1.5
        }

class OptionsAnalytics:
    """Options-specific analytics"""
    
    @staticmethod
    def calculate_implied_volatility_surface(options_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate implied volatility surface"""
        # Mock implementation - replace with real IV calculation
        strikes = sorted(set(opt['strike_price'] for opt in options_data))
        expiries = sorted(set(opt['expiration'] for opt in options_data))
        
        iv_surface = {}
        for expiry in expiries:
            iv_surface[expiry] = {}
            for strike in strikes:
                # Mock IV calculation
                iv_surface[expiry][strike] = 0.20 + np.random.normal(0, 0.05)
        
        return {
            "surface": iv_surface,
            "strikes": strikes,
            "expiries": expiries,
            "timestamp": datetime.now().isoformat()
        }
    
    @staticmethod
    def calculate_greeks(option_price: float, underlying_price: float, strike: float,
                        time_to_expiry: float, risk_free_rate: float = 0.05,
                        volatility: float = 0.20) -> Dict[str, float]:
        """Calculate option Greeks (simplified Black-Scholes)"""
        # Simplified Greeks calculation for mock data
        # In production, use proper options pricing library
        
        moneyness = underlying_price / strike
        
        # Mock Greeks based on moneyness and time
        delta = 0.5 + (moneyness - 1) * 0.3
        gamma = 0.1 * np.exp(-abs(moneyness - 1) * 2)
        theta = -option_price * 0.1 / (time_to_expiry + 0.01)
        vega = underlying_price * 0.1 * np.sqrt(time_to_expiry)
        rho = strike * time_to_expiry * 0.01
        
        return {
            "delta": delta,
            "gamma": gamma,
            "theta": theta,
            "vega": vega,
            "rho": rho
        }

class AnalyticsService:
    """Analytics service for market data analysis"""
    
    def __init__(self, config: AnalyticsConfig = None):
        """Initialize analytics service"""
        self.config = config or AnalyticsConfig()
        self.app = Flask(__name__)
        CORS(self.app, origins="*")
        
        # Database connection
        self.db_pool = None
        self.redis_client = None
        
        # Analytics components
        self.indicators = TechnicalIndicators()
        self.patterns = PatternRecognition()
        self.options_analytics = OptionsAnalytics()
        
        # Background executor
        self.executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="analytics")
        
        # Initialize components
        self._init_database()
        self._init_redis()
        self._setup_routes()
    
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
            
            # Test connection
            self.redis_client.ping()
            logger.info("Redis connection initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {e}")
            self.redis_client = None
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.before_request
        def before_request():
            """Before request middleware"""
            g.start_time = time.time()
            g.request_id = request.headers.get('X-Request-ID', f"req_{int(time.time() * 1000)}")
        
        @self.app.after_request
        def after_request(response):
            """After request middleware"""
            duration = time.time() - g.start_time
            response.headers['X-Request-ID'] = g.request_id
            response.headers['X-Response-Time'] = f"{duration * 1000:.2f}ms"
            return response
        
        @self.app.errorhandler(Exception)
        def handle_exception(e):
            """Global exception handler"""
            logger.error(f"Unhandled exception: {str(e)}")
            return jsonify({
                "error": "Internal server error",
                "request_id": g.request_id,
                "timestamp": datetime.now().isoformat()
            }), 500
        
        # Health check
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Service health check"""
            health_status = {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "service": "analytics",
                "version": "1.0.0",
                "components": {
                    "database": "healthy" if self.db_pool else "unhealthy",
                    "redis": "healthy" if self.redis_client else "unhealthy"
                }
            }
            
            if not all(status == "healthy" for status in health_status["components"].values()):
                health_status["status"] = "degraded"
            
            return jsonify(health_status)
        
        # Technical indicators endpoints
        @self.app.route('/api/v1/analytics/indicators/<symbol>', methods=['GET'])
        def get_technical_indicators(symbol):
            """Get technical indicators for a symbol"""
            try:
                period = int(request.args.get('period', 20))
                indicators_list = request.args.get('indicators', 'sma,ema,rsi,macd').split(',')
                
                # Check cache first
                cache_key = f"indicators:{symbol}:{period}:{','.join(sorted(indicators_list))}"
                cached_result = self._get_from_cache(cache_key)
                
                if cached_result:
                    return jsonify(cached_result)
                
                # Get price data
                price_data = self._get_price_data(symbol, days=max(period * 2, 50))
                
                if not price_data:
                    return jsonify({
                        "success": False,
                        "error": "No price data available"
                    }), 404
                
                # Calculate indicators
                result = self._calculate_indicators(price_data, indicators_list, period)
                result.update({
                    "success": True,
                    "symbol": symbol,
                    "period": period,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Cache result
                self._cache_result(cache_key, result)
                
                return jsonify(result)
            
            except Exception as e:
                logger.error(f"Failed to get technical indicators: {e}")
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 500
        
        # Pattern recognition endpoints
        @self.app.route('/api/v1/analytics/patterns/<symbol>', methods=['GET'])
        def get_patterns(symbol):
            """Get chart patterns for a symbol"""
            try:
                days = int(request.args.get('days', 30))
                
                # Check cache first
                cache_key = f"patterns:{symbol}:{days}"
                cached_result = self._get_from_cache(cache_key)
                
                if cached_result:
                    return jsonify(cached_result)
                
                # Get price data
                price_data = self._get_price_data(symbol, days=days)
                
                if not price_data:
                    return jsonify({
                        "success": False,
                        "error": "No price data available"
                    }), 404
                
                # Analyze patterns
                result = self._analyze_patterns(price_data)
                result.update({
                    "success": True,
                    "symbol": symbol,
                    "days": days,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Cache result
                self._cache_result(cache_key, result)
                
                return jsonify(result)
            
            except Exception as e:
                logger.error(f"Failed to get patterns: {e}")
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 500
        
        # Options analytics endpoints
        @self.app.route('/api/v1/analytics/options/<symbol>', methods=['GET'])
        def get_options_analytics(symbol):
            """Get options analytics for a symbol"""
            try:
                expiry = request.args.get('expiry')
                
                # Check cache first
                cache_key = f"options:{symbol}:{expiry or 'all'}"
                cached_result = self._get_from_cache(cache_key)
                
                if cached_result:
                    return jsonify(cached_result)
                
                # Get options data
                options_data = self._get_options_data(symbol, expiry)
                
                if not options_data:
                    return jsonify({
                        "success": False,
                        "error": "No options data available"
                    }), 404
                
                # Calculate options analytics
                result = self._calculate_options_analytics(options_data)
                result.update({
                    "success": True,
                    "symbol": symbol,
                    "expiry": expiry,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Cache result
                self._cache_result(cache_key, result)
                
                return jsonify(result)
            
            except Exception as e:
                logger.error(f"Failed to get options analytics: {e}")
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 500
        
        # Market analysis endpoints
        @self.app.route('/api/v1/analytics/market-analysis', methods=['GET'])
        def get_market_analysis():
            """Get overall market analysis"""
            try:
                symbols = request.args.get('symbols', 'AAPL,GOOGL,MSFT').split(',')
                
                # Check cache first
                cache_key = f"market_analysis:{','.join(sorted(symbols))}"
                cached_result = self._get_from_cache(cache_key)
                
                if cached_result:
                    return jsonify(cached_result)
                
                # Analyze each symbol
                analysis = {}
                for symbol in symbols:
                    symbol_analysis = self._analyze_symbol(symbol.strip())
                    if symbol_analysis:
                        analysis[symbol.strip()] = symbol_analysis
                
                # Calculate market sentiment
                market_sentiment = self._calculate_market_sentiment(analysis)
                
                result = {
                    "success": True,
                    "symbols": analysis,
                    "market_sentiment": market_sentiment,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Cache result
                self._cache_result(cache_key, result)
                
                return jsonify(result)
            
            except Exception as e:
                logger.error(f"Failed to get market analysis: {e}")
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 500
        
        # Custom analytics endpoints
        @self.app.route('/api/v1/analytics/custom', methods=['POST'])
        def run_custom_analysis():
            """Run custom analytics"""
            try:
                data = request.get_json()
                
                if not data or 'analysis_type' not in data:
                    return jsonify({
                        "success": False,
                        "error": "analysis_type is required"
                    }), 400
                
                analysis_type = data['analysis_type']
                parameters = data.get('parameters', {})
                
                # Run custom analysis
                result = self._run_custom_analysis(analysis_type, parameters)
                
                return jsonify({
                    "success": True,
                    "analysis_type": analysis_type,
                    "result": result,
                    "timestamp": datetime.now().isoformat()
                })
            
            except Exception as e:
                logger.error(f"Failed to run custom analysis: {e}")
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 500
    
    def _get_from_cache(self, key: str) -> Optional[Dict[str, Any]]:
        """Get result from cache"""
        if not self.redis_client:
            return None
        
        try:
            cached_data = self.redis_client.get(key)
            if cached_data:
                return json.loads(cached_data)
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
        
        return None
    
    def _cache_result(self, key: str, result: Dict[str, Any]):
        """Cache analysis result"""
        if not self.redis_client:
            return
        
        try:
            self.redis_client.setex(key, self.config.cache_ttl, json.dumps(result, default=str))
        except Exception as e:
            logger.warning(f"Cache set error: {e}")
    
    def _get_price_data(self, symbol: str, days: int = 30) -> Optional[pd.DataFrame]:
        """Get price data from database"""
        if not self.db_pool:
            return None
        
        conn = None
        try:
            conn = self.db_pool.getconn()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Get OHLCV data
            end_time = int(time.time() * 1e9)
            start_time = end_time - (days * 24 * 60 * 60 * 1e9)
            
            query = """
                SELECT ts_event, open, high, low, close, volume
                FROM stock_ohlcv
                WHERE symbol = %s
                AND ts_event >= %s AND ts_event <= %s
                ORDER BY ts_event
            """
            
            cursor.execute(query, (symbol, start_time, end_time))
            rows = cursor.fetchall()
            
            if not rows:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame([dict(row) for row in rows])
            df['timestamp'] = pd.to_datetime(df['ts_event'], unit='ns')
            df.set_index('timestamp', inplace=True)
            
            # Convert fixed-point prices to float
            for col in ['open', 'high', 'low', 'close']:
                df[col] = df[col] / 1e9
            
            return df
        
        except Exception as e:
            logger.error(f"Failed to get price data: {e}")
            return None
        
        finally:
            if conn:
                self.db_pool.putconn(conn)
    
    def _calculate_indicators(self, price_data: pd.DataFrame, indicators: List[str], period: int) -> Dict[str, Any]:
        """Calculate technical indicators"""
        result = {}
        
        prices = price_data['close'].values
        highs = price_data['high'].values
        lows = price_data['low'].values
        volumes = price_data['volume'].values
        
        for indicator in indicators:
            try:
                if indicator == 'sma':
                    result['sma'] = self.indicators.sma(prices, period)[-1] if len(prices) >= period else None
                elif indicator == 'ema':
                    result['ema'] = self.indicators.ema(prices, period)[-1] if len(prices) >= period else None
                elif indicator == 'rsi':
                    rsi_values = self.indicators.rsi(prices, period)
                    result['rsi'] = rsi_values[-1] if len(rsi_values) > 0 and not np.isnan(rsi_values[-1]) else None
                elif indicator == 'macd':
                    macd, signal, histogram = self.indicators.macd(prices)
                    result['macd'] = {
                        'macd': macd[-1] if len(macd) > 0 and not np.isnan(macd[-1]) else None,
                        'signal': signal[-1] if len(signal) > 0 and not np.isnan(signal[-1]) else None,
                        'histogram': histogram[-1] if len(histogram) > 0 and not np.isnan(histogram[-1]) else None
                    }
                elif indicator == 'bollinger':
                    upper, middle, lower = self.indicators.bollinger_bands(prices, period)
                    result['bollinger_bands'] = {
                        'upper': upper[-1] if len(upper) > 0 and not np.isnan(upper[-1]) else None,
                        'middle': middle[-1] if len(middle) > 0 and not np.isnan(middle[-1]) else None,
                        'lower': lower[-1] if len(lower) > 0 and not np.isnan(lower[-1]) else None
                    }
                elif indicator == 'atr':
                    atr_values = self.indicators.atr(highs, lows, prices, period)
                    result['atr'] = atr_values[-1] if len(atr_values) > 0 and not np.isnan(atr_values[-1]) else None
                elif indicator == 'volume_profile':
                    result['volume_profile'] = self.indicators.volume_profile(prices, volumes)
            
            except Exception as e:
                logger.warning(f"Failed to calculate {indicator}: {e}")
                result[indicator] = None
        
        return result
    
    def _analyze_patterns(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze chart patterns"""
        prices = price_data['close'].values
        volumes = price_data['volume'].values
        
        # Support/Resistance
        support_resistance = self.patterns.detect_support_resistance(prices)
        
        # Trend analysis
        trend_analysis = self.patterns.detect_trend(prices)
        
        # Breakout detection
        breakout_analysis = self.patterns.detect_breakout(prices, volumes)
        
        return {
            "support_resistance": support_resistance,
            "trend": trend_analysis,
            "breakout": breakout_analysis
        }
    
    def _get_options_data(self, symbol: str, expiry: str = None) -> List[Dict[str, Any]]:
        """Get options data from database"""
        # Mock options data for now
        return [
            {
                "symbol": f"{symbol}231215C00150000",
                "underlying_symbol": symbol,
                "strike_price": 150.0,
                "expiration": "2023-12-15",
                "option_type": "CALL",
                "bid_price": 2.50,
                "ask_price": 2.55,
                "last_price": 2.52,
                "volume": 1000,
                "open_interest": 5000,
                "implied_volatility": 0.25
            }
        ]
    
    def _calculate_options_analytics(self, options_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate options analytics"""
        # IV Surface
        iv_surface = self.options_analytics.calculate_implied_volatility_surface(options_data)
        
        # Greeks for each option
        greeks = []
        for option in options_data:
            option_greeks = self.options_analytics.calculate_greeks(
                option['last_price'],
                150.0,  # Mock underlying price
                option['strike_price'],
                0.1,  # Mock time to expiry
                0.05,  # Risk-free rate
                option['implied_volatility']
            )
            option_greeks['symbol'] = option['symbol']
            greeks.append(option_greeks)
        
        return {
            "iv_surface": iv_surface,
            "greeks": greeks,
            "total_volume": sum(opt['volume'] for opt in options_data),
            "total_open_interest": sum(opt['open_interest'] for opt in options_data)
        }
    
    def _analyze_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Analyze a single symbol"""
        try:
            # Get price data
            price_data = self._get_price_data(symbol, days=30)
            if not price_data:
                return None
            
            # Calculate key indicators
            indicators = self._calculate_indicators(price_data, ['sma', 'rsi', 'macd'], 20)
            
            # Analyze patterns
            patterns = self._analyze_patterns(price_data)
            
            # Current price info
            current_price = price_data['close'].iloc[-1]
            price_change = current_price - price_data['close'].iloc[-2] if len(price_data) > 1 else 0
            price_change_pct = (price_change / price_data['close'].iloc[-2] * 100) if len(price_data) > 1 else 0
            
            return {
                "current_price": current_price,
                "price_change": price_change,
                "price_change_pct": price_change_pct,
                "indicators": indicators,
                "patterns": patterns,
                "volume": price_data['volume'].iloc[-1],
                "avg_volume": price_data['volume'].mean()
            }
        
        except Exception as e:
            logger.error(f"Failed to analyze symbol {symbol}: {e}")
            return None
    
    def _calculate_market_sentiment(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall market sentiment"""
        if not analysis:
            return {"sentiment": "neutral", "confidence": 0}
        
        bullish_signals = 0
        bearish_signals = 0
        total_signals = 0
        
        for symbol, data in analysis.items():
            if not data:
                continue
            
            # Price momentum
            if data.get('price_change_pct', 0) > 1:
                bullish_signals += 1
            elif data.get('price_change_pct', 0) < -1:
                bearish_signals += 1
            total_signals += 1
            
            # RSI signals
            rsi = data.get('indicators', {}).get('rsi')
            if rsi:
                if rsi < 30:
                    bullish_signals += 1  # Oversold
                elif rsi > 70:
                    bearish_signals += 1  # Overbought
                total_signals += 1
            
            # Trend signals
            trend = data.get('patterns', {}).get('trend', {}).get('trend')
            if trend == 'uptrend':
                bullish_signals += 1
            elif trend == 'downtrend':
                bearish_signals += 1
            total_signals += 1
        
        if total_signals == 0:
            return {"sentiment": "neutral", "confidence": 0}
        
        bullish_ratio = bullish_signals / total_signals
        bearish_ratio = bearish_signals / total_signals
        
        if bullish_ratio > 0.6:
            sentiment = "bullish"
            confidence = bullish_ratio
        elif bearish_ratio > 0.6:
            sentiment = "bearish"
            confidence = bearish_ratio
        else:
            sentiment = "neutral"
            confidence = 1 - abs(bullish_ratio - bearish_ratio)
        
        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "bullish_signals": bullish_signals,
            "bearish_signals": bearish_signals,
            "total_signals": total_signals
        }
    
    def _run_custom_analysis(self, analysis_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run custom analysis"""
        # Mock custom analysis - extend as needed
        if analysis_type == "correlation_analysis":
            symbols = parameters.get('symbols', ['AAPL', 'GOOGL'])
            return {
                "correlation_matrix": [[1.0, 0.75], [0.75, 1.0]],
                "symbols": symbols
            }
        elif analysis_type == "volatility_analysis":
            symbol = parameters.get('symbol', 'AAPL')
            return {
                "historical_volatility": 0.25,
                "implied_volatility": 0.30,
                "volatility_rank": 65
            }
        else:
            return {"error": f"Unknown analysis type: {analysis_type}"}
    
    def run(self):
        """Run the analytics service"""
        logger.info(f"Starting Analytics Service on {self.config.service_host}:{self.config.service_port}")
        self.app.run(
            host=self.config.service_host,
            port=self.config.service_port,
            debug=False,
            threaded=True
        )

# Factory function
def create_analytics_service(config: AnalyticsConfig = None) -> AnalyticsService:
    """Create analytics service"""
    return AnalyticsService(config)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Options Trading System Analytics Service")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8003, help="Port to bind to")
    
    args = parser.parse_args()
    
    # Create configuration
    config = AnalyticsConfig()
    config.service_host = args.host
    config.service_port = args.port
    
    # Create and run service
    service = create_analytics_service(config)
    service.run()

