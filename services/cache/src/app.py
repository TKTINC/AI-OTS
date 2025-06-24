"""
Cache Service API for AI Options Trading System
Flask-based REST API for cache operations
"""

from flask import Flask, request, jsonify, g
from flask_cors import CORS
import logging
import time
import json
from datetime import datetime
from typing import Dict, Any, Optional
import traceback
from functools import wraps

from config import get_config
from redis_manager import create_redis_manager
from cache_strategies import UnifiedCacheManager

# Initialize Flask app
app = Flask(__name__)

# Global variables
cache_manager = None
unified_cache = None
config = None

def create_app():
    """Application factory"""
    global cache_manager, unified_cache, config
    
    # Load configuration
    config = get_config()
    
    # Setup CORS
    if config.service.enable_cors:
        CORS(app, origins=config.service.cors_origins.split(","))
    
    # Initialize Redis manager
    cache_manager = create_redis_manager(
        host=config.redis.host,
        port=config.redis.port,
        password=config.redis.password,
        db=config.redis.db,
        ssl=config.redis.ssl,
        socket_timeout=config.redis.socket_timeout,
        socket_connect_timeout=config.redis.socket_connect_timeout,
        max_connections=config.redis.max_connections,
        ttl_stock_prices=config.redis.ttl_stock_prices,
        ttl_options_data=config.redis.ttl_options_data,
        ttl_signals=config.redis.ttl_signals,
        ttl_user_sessions=config.redis.ttl_user_sessions,
        ttl_rate_limits=config.redis.ttl_rate_limits,
        ttl_analytics=config.redis.ttl_analytics,
        ttl_market_status=config.redis.ttl_market_status,
        compress_threshold=config.redis.compress_threshold,
        compression_level=config.redis.compression_level
    )
    
    # Initialize unified cache manager
    unified_cache = UnifiedCacheManager(cache_manager)
    
    return app

# Middleware and decorators
def rate_limit_check():
    """Check rate limits for requests"""
    if not config.service.rate_limit_enabled:
        return True
    
    # Get client identifier (IP address or API key)
    client_id = request.headers.get('X-API-Key') or request.remote_addr
    
    # Check rate limit
    rate_limit_result = cache_manager.check_rate_limit(
        f"api:{client_id}",
        config.service.rate_limit_requests,
        config.service.rate_limit_window
    )
    
    if not rate_limit_result["allowed"]:
        return False, rate_limit_result
    
    return True, rate_limit_result

def require_rate_limit(f):
    """Decorator to enforce rate limiting"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        allowed, rate_info = rate_limit_check()
        
        if not allowed:
            response = jsonify({
                "error": "Rate limit exceeded",
                "limit": rate_info["limit"],
                "remaining": rate_info["remaining"],
                "reset_time": rate_info["reset_time"]
            })
            response.status_code = 429
            response.headers["X-RateLimit-Limit"] = str(rate_info["limit"])
            response.headers["X-RateLimit-Remaining"] = str(rate_info["remaining"])
            response.headers["X-RateLimit-Reset"] = str(rate_info["reset_time"])
            return response
        
        # Add rate limit headers to successful responses
        response = f(*args, **kwargs)
        if hasattr(response, 'headers'):
            response.headers["X-RateLimit-Limit"] = str(rate_info["limit"])
            response.headers["X-RateLimit-Remaining"] = str(rate_info["remaining"])
            response.headers["X-RateLimit-Reset"] = str(rate_info["reset_time"])
        
        return response
    
    return decorated_function

@app.before_request
def before_request():
    """Before request middleware"""
    g.start_time = time.time()
    g.request_id = request.headers.get('X-Request-ID', f"req_{int(time.time() * 1000)}")

@app.after_request
def after_request(response):
    """After request middleware"""
    # Add request ID to response
    response.headers['X-Request-ID'] = g.request_id
    
    # Log request
    duration_ms = (time.time() - g.start_time) * 1000
    logging.info(
        f"Request {g.request_id}: {request.method} {request.path} "
        f"-> {response.status_code} ({duration_ms:.2f}ms)"
    )
    
    return response

@app.errorhandler(Exception)
def handle_exception(e):
    """Global exception handler"""
    logging.error(f"Unhandled exception in request {g.request_id}: {str(e)}")
    logging.error(traceback.format_exc())
    
    return jsonify({
        "error": "Internal server error",
        "request_id": g.request_id,
        "timestamp": datetime.now().isoformat()
    }), 500

# Health check endpoints
@app.route('/health', methods=['GET'])
def health_check():
    """Basic health check"""
    try:
        # Quick Redis ping
        cache_manager.redis_client.ping()
        
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "service": "cache-service",
            "version": "1.0.0"
        })
    
    except Exception as e:
        logging.error(f"Health check failed: {e}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 503

@app.route('/health/detailed', methods=['GET'])
def detailed_health_check():
    """Detailed health check with metrics"""
    try:
        health_data = unified_cache.health_check()
        
        return jsonify({
            "status": health_data["status"],
            "timestamp": datetime.now().isoformat(),
            "service": "cache-service",
            "version": "1.0.0",
            "details": health_data
        })
    
    except Exception as e:
        logging.error(f"Detailed health check failed: {e}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 503

# Stock price endpoints
@app.route('/api/v1/cache/stock-prices/<symbol>', methods=['GET'])
@require_rate_limit
def get_stock_price(symbol):
    """Get cached stock price"""
    try:
        price_data = unified_cache.stock_prices.get_price(symbol.upper())
        
        if price_data:
            return jsonify({
                "success": True,
                "data": price_data,
                "cached": True,
                "timestamp": datetime.now().isoformat()
            })
        else:
            return jsonify({
                "success": False,
                "error": "Stock price not found in cache",
                "symbol": symbol.upper()
            }), 404
    
    except Exception as e:
        logging.error(f"Error getting stock price for {symbol}: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/v1/cache/stock-prices/<symbol>', methods=['POST'])
@require_rate_limit
def set_stock_price(symbol):
    """Cache stock price"""
    try:
        price_data = request.get_json()
        
        if not price_data:
            return jsonify({
                "success": False,
                "error": "No price data provided"
            }), 400
        
        # Add symbol to data if not present
        price_data["symbol"] = symbol.upper()
        
        success = unified_cache.stock_prices.set_price(symbol.upper(), price_data)
        
        return jsonify({
            "success": success,
            "message": "Stock price cached successfully" if success else "Failed to cache stock price",
            "symbol": symbol.upper(),
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        logging.error(f"Error setting stock price for {symbol}: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/v1/cache/stock-prices/latest', methods=['GET'])
@require_rate_limit
def get_latest_prices():
    """Get latest prices for multiple symbols"""
    try:
        symbols = request.args.get('symbols', '').split(',')
        symbols = [s.strip().upper() for s in symbols if s.strip()]
        
        if symbols:
            prices = cache_manager.get_latest_prices(symbols)
        else:
            prices = cache_manager.get_latest_prices()
        
        return jsonify({
            "success": True,
            "data": prices,
            "count": len(prices),
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        logging.error(f"Error getting latest prices: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/v1/cache/stock-prices/bulk', methods=['POST'])
@require_rate_limit
def bulk_update_prices():
    """Bulk update stock prices"""
    try:
        price_updates = request.get_json()
        
        if not isinstance(price_updates, list):
            return jsonify({
                "success": False,
                "error": "Expected list of price updates"
            }), 400
        
        success_count = unified_cache.stock_prices.bulk_update_prices(price_updates)
        
        return jsonify({
            "success": True,
            "message": f"Successfully updated {success_count} prices",
            "updated_count": success_count,
            "total_count": len(price_updates),
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        logging.error(f"Error in bulk price update: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# Options endpoints
@app.route('/api/v1/cache/options/<symbol>/<expiry>', methods=['GET'])
@require_rate_limit
def get_options_chain(symbol, expiry):
    """Get cached options chain"""
    try:
        options_data = unified_cache.options.get_options_chain(symbol.upper(), expiry)
        
        if options_data:
            return jsonify({
                "success": True,
                "data": options_data,
                "symbol": symbol.upper(),
                "expiry": expiry,
                "count": len(options_data),
                "timestamp": datetime.now().isoformat()
            })
        else:
            return jsonify({
                "success": False,
                "error": "Options chain not found in cache",
                "symbol": symbol.upper(),
                "expiry": expiry
            }), 404
    
    except Exception as e:
        logging.error(f"Error getting options chain for {symbol}: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/v1/cache/options/<symbol>/<expiry>', methods=['POST'])
@require_rate_limit
def set_options_chain(symbol, expiry):
    """Cache options chain"""
    try:
        options_data = request.get_json()
        
        if not isinstance(options_data, list):
            return jsonify({
                "success": False,
                "error": "Expected list of options data"
            }), 400
        
        success = unified_cache.options.set_options_chain(symbol.upper(), expiry, options_data)
        
        return jsonify({
            "success": success,
            "message": "Options chain cached successfully" if success else "Failed to cache options chain",
            "symbol": symbol.upper(),
            "expiry": expiry,
            "count": len(options_data),
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        logging.error(f"Error setting options chain for {symbol}: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/v1/cache/options/<symbol>/summary', methods=['GET'])
@require_rate_limit
def get_options_summary(symbol):
    """Get options summary"""
    try:
        summary_data = unified_cache.options.get_options_summary(symbol.upper())
        
        if summary_data:
            return jsonify({
                "success": True,
                "data": summary_data,
                "symbol": symbol.upper(),
                "timestamp": datetime.now().isoformat()
            })
        else:
            return jsonify({
                "success": False,
                "error": "Options summary not found",
                "symbol": symbol.upper()
            }), 404
    
    except Exception as e:
        logging.error(f"Error getting options summary for {symbol}: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# Signal endpoints
@app.route('/api/v1/cache/signals/<signal_id>', methods=['GET'])
@require_rate_limit
def get_signal(signal_id):
    """Get cached signal"""
    try:
        signal_data = unified_cache.signals.get_signal(signal_id)
        
        if signal_data:
            return jsonify({
                "success": True,
                "data": signal_data,
                "signal_id": signal_id,
                "timestamp": datetime.now().isoformat()
            })
        else:
            return jsonify({
                "success": False,
                "error": "Signal not found in cache",
                "signal_id": signal_id
            }), 404
    
    except Exception as e:
        logging.error(f"Error getting signal {signal_id}: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/v1/cache/signals/<signal_id>', methods=['POST'])
@require_rate_limit
def set_signal(signal_id):
    """Cache signal"""
    try:
        signal_data = request.get_json()
        
        if not signal_data:
            return jsonify({
                "success": False,
                "error": "No signal data provided"
            }), 400
        
        success = unified_cache.signals.set_signal(signal_id, signal_data)
        
        return jsonify({
            "success": success,
            "message": "Signal cached successfully" if success else "Failed to cache signal",
            "signal_id": signal_id,
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        logging.error(f"Error setting signal {signal_id}: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/v1/cache/signals/active', methods=['GET'])
@require_rate_limit
def get_active_signals():
    """Get active signals"""
    try:
        symbol = request.args.get('symbol')
        
        if symbol:
            signals = unified_cache.signals.get_active_signals(symbol.upper())
        else:
            signals = unified_cache.signals.get_active_signals()
        
        return jsonify({
            "success": True,
            "data": signals,
            "count": len(signals),
            "symbol": symbol.upper() if symbol else None,
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        logging.error(f"Error getting active signals: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# Session endpoints
@app.route('/api/v1/cache/sessions', methods=['POST'])
@require_rate_limit
def create_session():
    """Create user session"""
    try:
        session_data = request.get_json()
        
        if not session_data or 'user_id' not in session_data:
            return jsonify({
                "success": False,
                "error": "user_id is required"
            }), 400
        
        user_id = session_data.pop('user_id')
        session_id = unified_cache.sessions.create_session(user_id, session_data)
        
        if session_id:
            return jsonify({
                "success": True,
                "session_id": session_id,
                "user_id": user_id,
                "timestamp": datetime.now().isoformat()
            })
        else:
            return jsonify({
                "success": False,
                "error": "Failed to create session"
            }), 500
    
    except Exception as e:
        logging.error(f"Error creating session: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/v1/cache/sessions/<session_id>', methods=['GET'])
@require_rate_limit
def get_session(session_id):
    """Get session data"""
    try:
        session_data = unified_cache.sessions.get_session(session_id)
        
        if session_data:
            return jsonify({
                "success": True,
                "data": session_data,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat()
            })
        else:
            return jsonify({
                "success": False,
                "error": "Session not found or expired",
                "session_id": session_id
            }), 404
    
    except Exception as e:
        logging.error(f"Error getting session {session_id}: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/v1/cache/sessions/<session_id>', methods=['DELETE'])
@require_rate_limit
def delete_session(session_id):
    """Delete session"""
    try:
        success = unified_cache.sessions.delete_session(session_id)
        
        return jsonify({
            "success": success,
            "message": "Session deleted successfully" if success else "Session not found",
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        logging.error(f"Error deleting session {session_id}: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# Analytics endpoints
@app.route('/api/v1/cache/analytics/<key>', methods=['GET'])
@require_rate_limit
def get_analytics(key):
    """Get cached analytics data"""
    try:
        analytics_data = cache_manager.get_analytics(key)
        
        if analytics_data:
            return jsonify({
                "success": True,
                "data": analytics_data,
                "key": key,
                "timestamp": datetime.now().isoformat()
            })
        else:
            return jsonify({
                "success": False,
                "error": "Analytics data not found",
                "key": key
            }), 404
    
    except Exception as e:
        logging.error(f"Error getting analytics {key}: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/v1/cache/analytics/<key>', methods=['POST'])
@require_rate_limit
def set_analytics(key):
    """Cache analytics data"""
    try:
        analytics_data = request.get_json()
        ttl = request.args.get('ttl', type=int)
        
        if not analytics_data:
            return jsonify({
                "success": False,
                "error": "No analytics data provided"
            }), 400
        
        success = cache_manager.cache_analytics(key, analytics_data, ttl)
        
        return jsonify({
            "success": success,
            "message": "Analytics data cached successfully" if success else "Failed to cache analytics data",
            "key": key,
            "ttl": ttl,
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        logging.error(f"Error setting analytics {key}: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# Market status endpoints
@app.route('/api/v1/cache/market-status', methods=['GET'])
@require_rate_limit
def get_market_status():
    """Get market status"""
    try:
        status_data = cache_manager.get_market_status()
        
        if status_data:
            return jsonify({
                "success": True,
                "data": status_data,
                "timestamp": datetime.now().isoformat()
            })
        else:
            return jsonify({
                "success": False,
                "error": "Market status not found"
            }), 404
    
    except Exception as e:
        logging.error(f"Error getting market status: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/v1/cache/market-status', methods=['POST'])
@require_rate_limit
def set_market_status():
    """Set market status"""
    try:
        status_data = request.get_json()
        
        if not status_data:
            return jsonify({
                "success": False,
                "error": "No status data provided"
            }), 400
        
        success = cache_manager.set_market_status(status_data)
        
        return jsonify({
            "success": success,
            "message": "Market status updated successfully" if success else "Failed to update market status",
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        logging.error(f"Error setting market status: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# Cache management endpoints
@app.route('/api/v1/cache/stats', methods=['GET'])
@require_rate_limit
def get_cache_stats():
    """Get cache statistics"""
    try:
        stats = unified_cache.get_overall_metrics()
        
        return jsonify({
            "success": True,
            "data": stats,
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        logging.error(f"Error getting cache stats: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/v1/cache/invalidate', methods=['POST'])
@require_rate_limit
def invalidate_cache():
    """Invalidate cache keys by pattern"""
    try:
        data = request.get_json()
        pattern = data.get('pattern') if data else None
        
        if not pattern:
            return jsonify({
                "success": False,
                "error": "Pattern is required"
            }), 400
        
        count = cache_manager.invalidate_pattern(pattern)
        
        return jsonify({
            "success": True,
            "message": f"Invalidated {count} keys",
            "pattern": pattern,
            "count": count,
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        logging.error(f"Error invalidating cache: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# Initialize app
if __name__ == '__main__':
    app = create_app()
    
    # Run development server
    app.run(
        host=config.service.host,
        port=config.service.port,
        debug=config.service.debug
    )

