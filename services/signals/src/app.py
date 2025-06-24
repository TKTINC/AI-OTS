"""
Main Flask Application for Signal Generation Service
AI Options Trading System - Week 2 Implementation
"""

import os
import sys
import logging
from datetime import datetime, timezone
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO
import redis
import json
from threading import Thread
import time

# Add src to path for imports
sys.path.insert(0, os.path.dirname(__file__))

# Import service components
from core.signal_service import SignalService
from strategies.options_strategies import OptionsStrategies
from patterns.pattern_recognition import PatternRecognition
from scoring.signal_scoring import SignalScoring
from broadcasting.signal_broadcaster import SignalBroadcaster, NotificationPreferences, NotificationChannel
from history.signal_tracker import SignalTracker
from integration.service_integrator import ServiceIntegrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'signal_service_secret_key')

# Enable CORS for all routes
CORS(app, origins="*")

# Initialize SocketIO
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Initialize service components
signal_service = SignalService()
options_strategies = OptionsStrategies()
pattern_recognition = PatternRecognition()
signal_scoring = SignalScoring()
signal_tracker = SignalTracker()
service_integrator = ServiceIntegrator()

# Initialize broadcaster with SocketIO integration
broadcaster = SignalBroadcaster()

# Global configuration
TARGET_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "SPY", "QQQ"]
SIGNAL_GENERATION_INTERVAL = 300  # 5 minutes

class SignalGenerationApp:
    """Main application class for signal generation service"""
    
    def __init__(self):
        self.running = False
        self.generation_thread = None
        
    def start_background_generation(self):
        """Start background signal generation"""
        if not self.running:
            self.running = True
            self.generation_thread = Thread(target=self._background_signal_generation, daemon=True)
            self.generation_thread.start()
            logger.info("Background signal generation started")
    
    def stop_background_generation(self):
        """Stop background signal generation"""
        self.running = False
        if self.generation_thread:
            self.generation_thread.join(timeout=5)
        logger.info("Background signal generation stopped")
    
    def _background_signal_generation(self):
        """Background thread for continuous signal generation"""
        while self.running:
            try:
                for symbol in TARGET_SYMBOLS:
                    self._generate_signals_for_symbol(symbol)
                
                # Wait for next interval
                time.sleep(SIGNAL_GENERATION_INTERVAL)
                
            except Exception as e:
                logger.error(f"Error in background signal generation: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
    
    def _generate_signals_for_symbol(self, symbol: str):
        """Generate signals for a specific symbol"""
        try:
            # Get comprehensive market analysis
            analysis = service_integrator.get_comprehensive_market_analysis(symbol)
            
            if analysis.get("error"):
                logger.warning(f"Failed to get analysis for {symbol}: {analysis['error']}")
                return
            
            # Extract data components
            market_data = analysis.get("market_data", {})
            technical_analysis = analysis.get("technical_analysis", {})
            pattern_analysis = analysis.get("pattern_analysis", {})
            options_data = analysis.get("options_data", {})
            options_analytics = analysis.get("options_analytics", {})
            
            # Generate signals using different strategies
            strategies = [
                ("Momentum Breakout", options_strategies.momentum_breakout_strategy),
                ("Volatility Squeeze", options_strategies.volatility_squeeze_strategy),
                ("Gamma Scalping", options_strategies.gamma_scalping_strategy),
                ("Delta Neutral Straddle", options_strategies.delta_neutral_straddle_strategy),
                ("Iron Condor Range", options_strategies.iron_condor_range_strategy)
            ]
            
            for strategy_name, strategy_func in strategies:
                try:
                    signal_data = strategy_func(market_data, technical_analysis, options_analytics)
                    
                    if signal_data and signal_data.get("confidence", 0) > 0.6:
                        # Score the signal
                        score = signal_scoring.score_signal(signal_data, market_data)
                        signal_data["score"] = score
                        
                        # Only proceed with high-quality signals
                        if score.get("overall_score", 0) > 70:
                            # Create and validate signal
                            signal = signal_service.create_signal(signal_data)
                            
                            if signal:
                                # Track signal generation
                                signal_tracker.track_signal_generation(signal_data)
                                
                                # Broadcast signal
                                broadcaster.broadcast_signal(signal_data)
                                
                                logger.info(f"Generated {strategy_name} signal for {symbol}: {signal.signal_id}")
                
                except Exception as e:
                    logger.error(f"Error generating {strategy_name} signal for {symbol}: {e}")
            
        except Exception as e:
            logger.error(f"Error generating signals for {symbol}: {e}")

# Initialize app instance
signal_app = SignalGenerationApp()

# API Routes
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Check service health
        health_status = service_integrator.check_service_health()
        
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "service": "signal-generation",
            "version": "2.0.0",
            "dependencies": health_status
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }), 500

@app.route('/api/v1/signals/generate', methods=['POST'])
def generate_signal():
    """Generate a signal manually"""
    try:
        data = request.get_json()
        symbol = data.get('symbol')
        strategy = data.get('strategy', 'auto')
        
        if not symbol:
            return jsonify({"error": "Symbol is required"}), 400
        
        # Get market analysis
        analysis = service_integrator.get_comprehensive_market_analysis(symbol)
        
        if analysis.get("error"):
            return jsonify({"error": f"Failed to get market data: {analysis['error']}"}), 500
        
        # Generate signal based on strategy
        if strategy == 'auto' or strategy == 'momentum_breakout':
            signal_data = options_strategies.momentum_breakout_strategy(
                analysis.get("market_data", {}),
                analysis.get("technical_analysis", {}),
                analysis.get("options_analytics", {})
            )
        elif strategy == 'volatility_squeeze':
            signal_data = options_strategies.volatility_squeeze_strategy(
                analysis.get("market_data", {}),
                analysis.get("technical_analysis", {}),
                analysis.get("options_analytics", {})
            )
        else:
            return jsonify({"error": f"Unknown strategy: {strategy}"}), 400
        
        if not signal_data:
            return jsonify({"error": "No signal generated"}), 404
        
        # Score the signal
        score = signal_scoring.score_signal(signal_data, analysis.get("market_data", {}))
        signal_data["score"] = score
        
        # Create signal
        signal = signal_service.create_signal(signal_data)
        
        if signal:
            # Track and broadcast
            signal_tracker.track_signal_generation(signal_data)
            broadcaster.broadcast_signal(signal_data)
            
            return jsonify({
                "success": True,
                "signal": signal_data,
                "signal_id": signal.signal_id
            })
        else:
            return jsonify({"error": "Failed to create signal"}), 500
            
    except Exception as e:
        logger.error(f"Error generating signal: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/v1/signals/active', methods=['GET'])
def get_active_signals():
    """Get active signals"""
    try:
        symbol = request.args.get('symbol')
        strategy = request.args.get('strategy')
        min_confidence = float(request.args.get('min_confidence', 0.0))
        limit = int(request.args.get('limit', 20))
        
        signals = signal_service.get_active_signals(
            symbol=symbol,
            strategy=strategy,
            min_confidence=min_confidence,
            limit=limit
        )
        
        return jsonify({
            "success": True,
            "signals": [signal.to_dict() for signal in signals],
            "count": len(signals)
        })
        
    except Exception as e:
        logger.error(f"Error getting active signals: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/v1/signals/<signal_id>', methods=['GET'])
def get_signal(signal_id):
    """Get specific signal details"""
    try:
        signal = signal_service.get_signal(signal_id)
        
        if signal:
            return jsonify({
                "success": True,
                "signal": signal.to_dict()
            })
        else:
            return jsonify({"error": "Signal not found"}), 404
            
    except Exception as e:
        logger.error(f"Error getting signal {signal_id}: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/v1/signals/<signal_id>/execute', methods=['POST'])
def execute_signal(signal_id):
    """Mark signal as executed"""
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        execution_price = data.get('execution_price')
        position_size = data.get('position_size')
        
        if not all([user_id, execution_price, position_size]):
            return jsonify({"error": "Missing required fields"}), 400
        
        # Get signal details
        signal = signal_service.get_signal(signal_id)
        if not signal:
            return jsonify({"error": "Signal not found"}), 404
        
        # Create execution record
        from history.signal_tracker import SignalExecution
        execution = SignalExecution(
            signal_id=signal_id,
            execution_id=f"exec_{signal_id}_{int(time.time())}",
            user_id=user_id,
            symbol=signal.symbol,
            strategy_name=signal.strategy_name,
            signal_type=signal.signal_type.value,
            executed_at=datetime.now(timezone.utc),
            execution_price=float(execution_price),
            position_size=float(position_size),
            original_entry_price=signal.entry_price,
            original_target_price=signal.target_price,
            original_stop_loss=signal.stop_loss,
            original_confidence=signal.confidence,
            original_expected_return=signal.expected_return
        )
        
        # Track execution
        result = signal_tracker.track_signal_execution(execution)
        
        if result:
            return jsonify({
                "success": True,
                "execution_id": execution.execution_id,
                "message": "Signal execution tracked"
            })
        else:
            return jsonify({"error": "Failed to track execution"}), 500
            
    except Exception as e:
        logger.error(f"Error executing signal {signal_id}: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/v1/signals/performance', methods=['GET'])
def get_performance():
    """Get performance analytics"""
    try:
        user_id = request.args.get('user_id')
        days = int(request.args.get('days', 30))
        
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)
        
        report = signal_tracker.generate_performance_report(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date
        )
        
        return jsonify({
            "success": True,
            "performance": {
                "period_days": days,
                "total_signals": report.total_signals,
                "executed_signals": report.executed_signals,
                "win_rate": report.win_rate,
                "average_return": report.average_return,
                "total_return": report.total_return,
                "sharpe_ratio": report.sharpe_ratio,
                "max_drawdown": report.max_drawdown,
                "strategy_performance": report.strategy_performance
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting performance: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/v1/signals/subscribe', methods=['POST'])
def subscribe_notifications():
    """Subscribe to signal notifications"""
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        
        if not user_id:
            return jsonify({"error": "User ID is required"}), 400
        
        preferences = NotificationPreferences(
            user_id=user_id,
            channels=[NotificationChannel(ch) for ch in data.get('channels', ['websocket'])],
            min_priority=data.get('min_priority', 2),
            symbols=data.get('symbols'),
            strategies=data.get('strategies'),
            custom_filters=data.get('custom_filters'),
            email=data.get('email'),
            phone=data.get('phone'),
            slack_webhook=data.get('slack_webhook'),
            discord_webhook=data.get('discord_webhook')
        )
        
        result = broadcaster.subscribe_user(user_id, preferences)
        
        return jsonify({
            "success": result,
            "message": "Subscription successful" if result else "Subscription failed"
        })
        
    except Exception as e:
        logger.error(f"Error subscribing to notifications: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/v1/signals/stats', methods=['GET'])
def get_signal_stats():
    """Get signal generation statistics"""
    try:
        stats = signal_service.get_generation_stats()
        
        return jsonify({
            "success": True,
            "stats": stats
        })
        
    except Exception as e:
        logger.error(f"Error getting signal stats: {e}")
        return jsonify({"error": str(e)}), 500

# SocketIO Events
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info(f"Client connected: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info(f"Client disconnected: {request.sid}")

@socketio.on('join_signals')
def handle_join_signals(data):
    """Join signal notifications room"""
    user_id = data.get('user_id')
    if user_id:
        join_room(f"signals_{user_id}")
        logger.info(f"Client {request.sid} joined signals room for user {user_id}")

# Application startup
def create_app():
    """Application factory"""
    return app

if __name__ == '__main__':
    try:
        # Start background signal generation
        signal_app.start_background_generation()
        
        # Start the Flask-SocketIO server
        logger.info("Starting Signal Generation Service on port 8004")
        socketio.run(
            app,
            host='0.0.0.0',
            port=8004,
            debug=False,
            use_reloader=False
        )
        
    except KeyboardInterrupt:
        logger.info("Shutting down Signal Generation Service")
        signal_app.stop_background_generation()
        service_integrator.shutdown()
    except Exception as e:
        logger.error(f"Error starting Signal Generation Service: {e}")
        signal_app.stop_background_generation()
        service_integrator.shutdown()
        sys.exit(1)

