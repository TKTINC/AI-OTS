"""
Data Ingestion Service for AI Options Trading System
Handles real-time and historical data collection from Databento and other sources
"""

from flask import Flask, request, jsonify, g
from flask_cors import CORS
import logging
import time
import json
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import threading
from dataclasses import dataclass, asdict
import psycopg2
from psycopg2.extras import RealDictCursor
import redis
import os

# Import our custom modules
from databento_client import create_databento_client, SubscriptionRequest, DatabentoMode
from mock_data_generator import MockDataGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DataIngestionConfig:
    """Configuration for data ingestion service"""
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
    
    # Databento settings
    databento_api_key: str = os.getenv("DATABENTO_API_KEY", "mock_api_key")
    databento_mode: str = os.getenv("DATABENTO_MODE", "mock")
    
    # Service settings
    service_host: str = os.getenv("DATA_SERVICE_HOST", "0.0.0.0")
    service_port: int = int(os.getenv("DATA_SERVICE_PORT", "8002"))
    
    # Data collection settings
    target_symbols: List[str] = None
    collection_interval: int = int(os.getenv("COLLECTION_INTERVAL", "60"))  # seconds
    batch_size: int = int(os.getenv("BATCH_SIZE", "1000"))
    
    def __post_init__(self):
        if self.target_symbols is None:
            self.target_symbols = [
                "AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA", "META"
            ]

class DataIngestionService:
    """Data ingestion service for market data"""
    
    def __init__(self, config: DataIngestionConfig = None):
        """Initialize data ingestion service"""
        self.config = config or DataIngestionConfig()
        self.app = Flask(__name__)
        CORS(self.app, origins="*")
        
        # Database connection
        self.db_pool = None
        self.redis_client = None
        
        # Databento client
        self.databento_client = None
        
        # Data collection state
        self.collection_active = False
        self.collection_stats = {
            "records_collected": 0,
            "last_collection": None,
            "errors": 0,
            "active_subscriptions": []
        }
        
        # Background executor
        self.executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="data-ingestion")
        
        # Initialize components
        self._init_database()
        self._init_redis()
        self._init_databento()
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
    
    def _init_databento(self):
        """Initialize Databento client"""
        try:
            self.databento_client = create_databento_client(
                api_key=self.config.databento_api_key,
                mode=self.config.databento_mode,
                symbols=self.config.target_symbols
            )
            logger.info(f"Databento client initialized in {self.config.databento_mode} mode")
        except Exception as e:
            logger.error(f"Failed to initialize Databento client: {e}")
            self.databento_client = None
    
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
                "service": "data-ingestion",
                "version": "1.0.0",
                "components": {
                    "database": "healthy" if self.db_pool else "unhealthy",
                    "redis": "healthy" if self.redis_client else "unhealthy",
                    "databento": "healthy" if self.databento_client else "unhealthy"
                },
                "collection_stats": self.collection_stats
            }
            
            # Determine overall status
            if not all(status == "healthy" for status in health_status["components"].values()):
                health_status["status"] = "degraded"
            
            return jsonify(health_status)
        
        # Data collection endpoints
        @self.app.route('/api/v1/data/collection/start', methods=['POST'])
        def start_collection():
            """Start data collection"""
            try:
                data = request.get_json() or {}
                symbols = data.get('symbols', self.config.target_symbols)
                schemas = data.get('schemas', ['trades', 'tbbo', 'ohlcv-1m'])
                
                if self.collection_active:
                    return jsonify({
                        "success": False,
                        "message": "Data collection already active"
                    }), 400
                
                # Start collection in background
                self.executor.submit(self._start_data_collection, symbols, schemas)
                
                return jsonify({
                    "success": True,
                    "message": "Data collection started",
                    "symbols": symbols,
                    "schemas": schemas,
                    "timestamp": datetime.now().isoformat()
                })
            
            except Exception as e:
                logger.error(f"Failed to start collection: {e}")
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 500
        
        @self.app.route('/api/v1/data/collection/stop', methods=['POST'])
        def stop_collection():
            """Stop data collection"""
            try:
                self.collection_active = False
                
                return jsonify({
                    "success": True,
                    "message": "Data collection stopped",
                    "timestamp": datetime.now().isoformat()
                })
            
            except Exception as e:
                logger.error(f"Failed to stop collection: {e}")
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 500
        
        @self.app.route('/api/v1/data/collection/status', methods=['GET'])
        def collection_status():
            """Get collection status"""
            return jsonify({
                "active": self.collection_active,
                "stats": self.collection_stats,
                "config": {
                    "symbols": self.config.target_symbols,
                    "interval": self.config.collection_interval,
                    "batch_size": self.config.batch_size
                },
                "timestamp": datetime.now().isoformat()
            })
        
        # Historical data endpoints
        @self.app.route('/api/v1/data/historical', methods=['GET'])
        def get_historical_data():
            """Get historical market data"""
            try:
                symbol = request.args.get('symbol')
                schema = request.args.get('schema', 'trades')
                start_date = request.args.get('start')
                end_date = request.args.get('end')
                limit = int(request.args.get('limit', 1000))
                
                if not symbol:
                    return jsonify({
                        "success": False,
                        "error": "Symbol is required"
                    }), 400
                
                # Get data from database
                data = self._get_historical_data_from_db(symbol, schema, start_date, end_date, limit)
                
                return jsonify({
                    "success": True,
                    "data": data,
                    "count": len(data),
                    "symbol": symbol,
                    "schema": schema,
                    "timestamp": datetime.now().isoformat()
                })
            
            except Exception as e:
                logger.error(f"Failed to get historical data: {e}")
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 500
        
        # Real-time data endpoints
        @self.app.route('/api/v1/data/realtime/subscribe', methods=['POST'])
        def subscribe_realtime():
            """Subscribe to real-time data"""
            try:
                data = request.get_json()
                
                if not data or 'symbols' not in data:
                    return jsonify({
                        "success": False,
                        "error": "Symbols are required"
                    }), 400
                
                symbols = data['symbols']
                schema = data.get('schema', 'trades')
                callback_url = data.get('callback_url')
                
                # Create subscription
                subscription_id = self._create_realtime_subscription(symbols, schema, callback_url)
                
                return jsonify({
                    "success": True,
                    "subscription_id": subscription_id,
                    "symbols": symbols,
                    "schema": schema,
                    "timestamp": datetime.now().isoformat()
                })
            
            except Exception as e:
                logger.error(f"Failed to create subscription: {e}")
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 500
        
        # Market status endpoint
        @self.app.route('/api/v1/data/market-status', methods=['GET'])
        def get_market_status():
            """Get current market status"""
            try:
                if self.databento_client:
                    # This would be async in real implementation
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        status = loop.run_until_complete(self.databento_client.get_market_status())
                    finally:
                        loop.close()
                else:
                    # Mock status
                    status = {
                        "is_open": True,
                        "session": "regular",
                        "next_event": "close",
                        "next_time": "2023-12-01T21:00:00Z"
                    }
                
                return jsonify({
                    "success": True,
                    "data": status,
                    "timestamp": datetime.now().isoformat()
                })
            
            except Exception as e:
                logger.error(f"Failed to get market status: {e}")
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 500
        
        # Data quality endpoints
        @self.app.route('/api/v1/data/quality/check', methods=['POST'])
        def check_data_quality():
            """Check data quality for a symbol"""
            try:
                data = request.get_json()
                symbol = data.get('symbol')
                date = data.get('date', datetime.now().strftime('%Y-%m-%d'))
                
                if not symbol:
                    return jsonify({
                        "success": False,
                        "error": "Symbol is required"
                    }), 400
                
                quality_report = self._check_data_quality(symbol, date)
                
                return jsonify({
                    "success": True,
                    "data": quality_report,
                    "symbol": symbol,
                    "date": date,
                    "timestamp": datetime.now().isoformat()
                })
            
            except Exception as e:
                logger.error(f"Failed to check data quality: {e}")
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 500
    
    def _start_data_collection(self, symbols: List[str], schemas: List[str]):
        """Start background data collection"""
        self.collection_active = True
        self.collection_stats["active_subscriptions"] = []
        
        logger.info(f"Starting data collection for {len(symbols)} symbols")
        
        try:
            # Start Databento client
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            async def collection_loop():
                await self.databento_client.start()
                
                # Subscribe to each schema for each symbol
                for schema in schemas:
                    request = SubscriptionRequest(
                        dataset="XNAS.ITCH",
                        schema=schema,
                        symbols=symbols
                    )
                    
                    await self.databento_client.subscribe_live(
                        request,
                        lambda data: self._handle_market_data(data)
                    )
                    
                    self.collection_stats["active_subscriptions"].append(f"{schema}:{','.join(symbols)}")
                
                # Keep running while collection is active
                while self.collection_active:
                    await asyncio.sleep(1)
                
                await self.databento_client.stop()
            
            loop.run_until_complete(collection_loop())
        
        except Exception as e:
            logger.error(f"Data collection error: {e}")
            self.collection_active = False
        finally:
            self.collection_stats["last_collection"] = datetime.now().isoformat()
    
    def _handle_market_data(self, data: Dict[str, Any]):
        """Handle incoming market data"""
        try:
            # Store in database
            self._store_market_data(data)
            
            # Cache in Redis
            self._cache_market_data(data)
            
            # Update stats
            self.collection_stats["records_collected"] += 1
            
        except Exception as e:
            logger.error(f"Failed to handle market data: {e}")
            self.collection_stats["errors"] += 1
    
    def _store_market_data(self, data: Dict[str, Any]):
        """Store market data in database"""
        if not self.db_pool:
            return
        
        conn = None
        try:
            conn = self.db_pool.getconn()
            cursor = conn.cursor()
            
            # Determine table based on data type
            if 'action' in data and data['action'] == 'T':  # Trade
                table = 'stock_trades'
                query = """
                    INSERT INTO stock_trades (
                        ts_event, ts_recv, instrument_id, symbol, price, size, side, sequence
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (ts_event, instrument_id, sequence) DO NOTHING
                """
                values = (
                    data.get('ts_event'),
                    data.get('ts_recv'),
                    data.get('instrument_id'),
                    data.get('symbol'),
                    data.get('price'),
                    data.get('size'),
                    data.get('side'),
                    data.get('sequence')
                )
            
            elif 'levels' in data:  # Quote
                table = 'stock_quotes'
                query = """
                    INSERT INTO stock_quotes (
                        ts_event, ts_recv, instrument_id, symbol, side, price, size, sequence
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (ts_event, instrument_id, side, sequence) DO NOTHING
                """
                values = (
                    data.get('ts_event'),
                    data.get('ts_recv'),
                    data.get('instrument_id'),
                    data.get('symbol'),
                    data.get('side'),
                    data.get('price'),
                    data.get('size'),
                    data.get('sequence')
                )
            
            elif 'open' in data:  # OHLCV
                table = 'stock_ohlcv'
                query = """
                    INSERT INTO stock_ohlcv (
                        ts_event, instrument_id, symbol, open, high, low, close, volume
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (ts_event, instrument_id) DO UPDATE SET
                        open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume
                """
                values = (
                    data.get('ts_event'),
                    data.get('instrument_id'),
                    data.get('symbol'),
                    data.get('open'),
                    data.get('high'),
                    data.get('low'),
                    data.get('close'),
                    data.get('volume')
                )
            
            else:
                return  # Unknown data type
            
            cursor.execute(query, values)
            conn.commit()
        
        except Exception as e:
            logger.error(f"Failed to store market data: {e}")
            if conn:
                conn.rollback()
        
        finally:
            if conn:
                self.db_pool.putconn(conn)
    
    def _cache_market_data(self, data: Dict[str, Any]):
        """Cache market data in Redis"""
        if not self.redis_client:
            return
        
        try:
            symbol = data.get('symbol')
            if not symbol:
                return
            
            # Cache latest price
            if 'price' in data:
                price_key = f"latest_price:{symbol}"
                price_data = {
                    "price": data['price'],
                    "timestamp": data.get('ts_event'),
                    "volume": data.get('size', 0)
                }
                self.redis_client.setex(price_key, 300, json.dumps(price_data))  # 5 min TTL
            
            # Cache in time series
            ts_key = f"timeseries:{symbol}:{data.get('action', 'unknown')}"
            self.redis_client.zadd(ts_key, {json.dumps(data): data.get('ts_event', time.time())})
            
            # Keep only recent data (last hour)
            cutoff = time.time() - 3600
            self.redis_client.zremrangebyscore(ts_key, 0, cutoff)
        
        except Exception as e:
            logger.error(f"Failed to cache market data: {e}")
    
    def _get_historical_data_from_db(self, symbol: str, schema: str, 
                                   start_date: str, end_date: str, limit: int) -> List[Dict[str, Any]]:
        """Get historical data from database"""
        if not self.db_pool:
            return []
        
        conn = None
        try:
            conn = self.db_pool.getconn()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Determine table and query based on schema
            if schema == 'trades':
                table = 'stock_trades'
                query = f"""
                    SELECT * FROM {table}
                    WHERE symbol = %s
                    AND ts_event >= %s AND ts_event <= %s
                    ORDER BY ts_event DESC
                    LIMIT %s
                """
            elif schema == 'tbbo':
                table = 'stock_quotes'
                query = f"""
                    SELECT * FROM {table}
                    WHERE symbol = %s
                    AND ts_event >= %s AND ts_event <= %s
                    ORDER BY ts_event DESC
                    LIMIT %s
                """
            elif schema.startswith('ohlcv'):
                table = 'stock_ohlcv'
                query = f"""
                    SELECT * FROM {table}
                    WHERE symbol = %s
                    AND ts_event >= %s AND ts_event <= %s
                    ORDER BY ts_event DESC
                    LIMIT %s
                """
            else:
                return []
            
            # Convert dates to timestamps if provided
            start_ts = int(datetime.fromisoformat(start_date).timestamp() * 1e9) if start_date else 0
            end_ts = int(datetime.fromisoformat(end_date).timestamp() * 1e9) if end_date else int(time.time() * 1e9)
            
            cursor.execute(query, (symbol, start_ts, end_ts, limit))
            rows = cursor.fetchall()
            
            return [dict(row) for row in rows]
        
        except Exception as e:
            logger.error(f"Failed to get historical data: {e}")
            return []
        
        finally:
            if conn:
                self.db_pool.putconn(conn)
    
    def _create_realtime_subscription(self, symbols: List[str], schema: str, callback_url: str) -> str:
        """Create real-time data subscription"""
        subscription_id = f"sub_{int(time.time() * 1000)}"
        
        # Store subscription info in Redis
        if self.redis_client:
            subscription_data = {
                "id": subscription_id,
                "symbols": symbols,
                "schema": schema,
                "callback_url": callback_url,
                "created_at": datetime.now().isoformat()
            }
            
            self.redis_client.setex(
                f"subscription:{subscription_id}",
                3600,  # 1 hour TTL
                json.dumps(subscription_data)
            )
        
        return subscription_id
    
    def _check_data_quality(self, symbol: str, date: str) -> Dict[str, Any]:
        """Check data quality for a symbol on a specific date"""
        # Mock data quality check - replace with real implementation
        return {
            "symbol": symbol,
            "date": date,
            "completeness": 98.5,
            "accuracy": 99.2,
            "timeliness": 97.8,
            "consistency": 99.1,
            "issues": [
                "Minor gaps in data between 14:30-14:32",
                "Delayed quotes during market open"
            ],
            "overall_score": 98.7,
            "status": "good"
        }
    
    def run(self):
        """Run the data ingestion service"""
        logger.info(f"Starting Data Ingestion Service on {self.config.service_host}:{self.config.service_port}")
        self.app.run(
            host=self.config.service_host,
            port=self.config.service_port,
            debug=False,
            threaded=True
        )

# Factory function
def create_data_ingestion_service(config: DataIngestionConfig = None) -> DataIngestionService:
    """Create data ingestion service"""
    return DataIngestionService(config)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Options Trading System Data Ingestion Service")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8002, help="Port to bind to")
    parser.add_argument("--mode", default="mock", help="Databento mode (live/historical/mock)")
    
    args = parser.parse_args()
    
    # Create configuration
    config = DataIngestionConfig()
    config.service_host = args.host
    config.service_port = args.port
    config.databento_mode = args.mode
    
    # Create and run service
    service = create_data_ingestion_service(config)
    service.run()

