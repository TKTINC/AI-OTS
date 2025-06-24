"""
Real-time Signal Broadcasting and Notification System for AI Options Trading System
WebSocket-based real-time signal distribution with multi-channel notifications
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Any, Optional, Set, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timezone, timedelta
import websockets
import redis
from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit, join_room, leave_room
from threading import Thread, Lock
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import requests
import uuid
from queue import Queue, Empty
import sqlite3
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class NotificationChannel(Enum):
    """Notification delivery channels"""
    WEBSOCKET = "websocket"
    EMAIL = "email"
    SMS = "sms"
    SLACK = "slack"
    DISCORD = "discord"
    WEBHOOK = "webhook"
    PUSH = "push"

class SignalPriority(Enum):
    """Signal priority levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class SubscriptionType(Enum):
    """Subscription types"""
    ALL_SIGNALS = "all_signals"
    HIGH_PRIORITY = "high_priority"
    SYMBOL_SPECIFIC = "symbol_specific"
    STRATEGY_SPECIFIC = "strategy_specific"
    CUSTOM_FILTER = "custom_filter"

@dataclass
class NotificationPreferences:
    """User notification preferences"""
    user_id: str
    channels: List[NotificationChannel]
    min_priority: SignalPriority
    symbols: List[str] = None
    strategies: List[str] = None
    custom_filters: Dict[str, Any] = None
    quiet_hours: Dict[str, str] = None  # {"start": "22:00", "end": "06:00"}
    max_notifications_per_hour: int = 10
    email: str = None
    phone: str = None
    slack_webhook: str = None
    discord_webhook: str = None
    custom_webhook: str = None

@dataclass
class SignalBroadcast:
    """Signal broadcast message"""
    signal_id: str
    timestamp: datetime
    symbol: str
    strategy_name: str
    signal_type: str
    priority: SignalPriority
    confidence: float
    expected_return: float
    max_risk: float
    entry_price: float
    target_price: float
    stop_loss: float
    time_horizon: int
    reasoning: str
    technical_factors: Dict[str, Any]
    options_factors: Dict[str, Any]
    risk_factors: Dict[str, Any]
    score: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['priority'] = self.priority.name
        return data

class SignalBroadcaster:
    """Real-time signal broadcasting system"""
    
    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.subscribers: Dict[str, NotificationPreferences] = {}
        self.active_connections: Dict[str, Set[str]] = {}  # user_id -> set of connection_ids
        self.signal_queue = Queue()
        self.notification_history = {}
        self.rate_limiters = {}
        self.lock = Lock()
        
        # Initialize Flask-SocketIO for WebSocket support
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'signal_broadcaster_secret'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*", async_mode='threading')
        
        # Setup routes and event handlers
        self._setup_routes()
        self._setup_socketio_events()
        
        # Start background workers
        self.running = True
        self.broadcast_thread = Thread(target=self._broadcast_worker, daemon=True)
        self.cleanup_thread = Thread(target=self._cleanup_worker, daemon=True)
        
        # Initialize database for persistent storage
        self._init_database()
    
    def start(self):
        """Start the broadcasting system"""
        try:
            self.broadcast_thread.start()
            self.cleanup_thread.start()
            logger.info("Signal broadcaster started successfully")
        except Exception as e:
            logger.error(f"Error starting signal broadcaster: {e}")
    
    def stop(self):
        """Stop the broadcasting system"""
        self.running = False
        logger.info("Signal broadcaster stopped")
    
    def broadcast_signal(self, signal_data: Dict[str, Any]) -> bool:
        """Broadcast a new signal to all subscribers"""
        try:
            # Create broadcast message
            broadcast = SignalBroadcast(
                signal_id=signal_data.get("signal_id", str(uuid.uuid4())),
                timestamp=datetime.now(timezone.utc),
                symbol=signal_data.get("symbol", ""),
                strategy_name=signal_data.get("strategy_name", ""),
                signal_type=signal_data.get("signal_type", ""),
                priority=SignalPriority(signal_data.get("priority", 2)),
                confidence=signal_data.get("confidence", 0.0),
                expected_return=signal_data.get("expected_return", 0.0),
                max_risk=signal_data.get("max_risk", 0.0),
                entry_price=signal_data.get("entry_price", 0.0),
                target_price=signal_data.get("target_price", 0.0),
                stop_loss=signal_data.get("stop_loss", 0.0),
                time_horizon=signal_data.get("time_horizon", 0),
                reasoning=signal_data.get("reasoning", ""),
                technical_factors=signal_data.get("technical_factors", {}),
                options_factors=signal_data.get("options_factors", {}),
                risk_factors=signal_data.get("risk_factors", {}),
                score=signal_data.get("score", {})
            )
            
            # Add to broadcast queue
            self.signal_queue.put(broadcast)
            
            # Store in Redis for real-time access
            self.redis_client.lpush("recent_signals", json.dumps(broadcast.to_dict()))
            self.redis_client.ltrim("recent_signals", 0, 99)  # Keep last 100 signals
            
            logger.info(f"Signal {broadcast.signal_id} queued for broadcast")
            return True
            
        except Exception as e:
            logger.error(f"Error broadcasting signal: {e}")
            return False
    
    def subscribe_user(self, user_id: str, preferences: NotificationPreferences) -> bool:
        """Subscribe a user to signal notifications"""
        try:
            with self.lock:
                self.subscribers[user_id] = preferences
                self.rate_limiters[user_id] = {
                    "count": 0,
                    "reset_time": time.time() + 3600  # Reset every hour
                }
            
            # Store preferences in database
            self._store_user_preferences(user_id, preferences)
            
            logger.info(f"User {user_id} subscribed to signal notifications")
            return True
            
        except Exception as e:
            logger.error(f"Error subscribing user {user_id}: {e}")
            return False
    
    def unsubscribe_user(self, user_id: str) -> bool:
        """Unsubscribe a user from signal notifications"""
        try:
            with self.lock:
                if user_id in self.subscribers:
                    del self.subscribers[user_id]
                if user_id in self.rate_limiters:
                    del self.rate_limiters[user_id]
                if user_id in self.active_connections:
                    del self.active_connections[user_id]
            
            # Remove from database
            self._remove_user_preferences(user_id)
            
            logger.info(f"User {user_id} unsubscribed from signal notifications")
            return True
            
        except Exception as e:
            logger.error(f"Error unsubscribing user {user_id}: {e}")
            return False
    
    def get_recent_signals(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent signals from Redis"""
        try:
            signals = self.redis_client.lrange("recent_signals", 0, limit - 1)
            return [json.loads(signal) for signal in signals]
        except Exception as e:
            logger.error(f"Error getting recent signals: {e}")
            return []
    
    def _broadcast_worker(self):
        """Background worker for broadcasting signals"""
        while self.running:
            try:
                # Get signal from queue (blocking with timeout)
                try:
                    broadcast = self.signal_queue.get(timeout=1.0)
                except Empty:
                    continue
                
                # Broadcast to all eligible subscribers
                self._process_broadcast(broadcast)
                
                # Mark task as done
                self.signal_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in broadcast worker: {e}")
                time.sleep(1)
    
    def _process_broadcast(self, broadcast: SignalBroadcast):
        """Process and send broadcast to eligible subscribers"""
        try:
            eligible_users = self._get_eligible_users(broadcast)
            
            for user_id in eligible_users:
                preferences = self.subscribers.get(user_id)
                if not preferences:
                    continue
                
                # Check rate limiting
                if not self._check_rate_limit(user_id):
                    continue
                
                # Check quiet hours
                if self._is_quiet_hours(preferences):
                    continue
                
                # Send notifications through preferred channels
                self._send_notifications(user_id, preferences, broadcast)
                
                # Update rate limiter
                self._update_rate_limiter(user_id)
            
            logger.info(f"Broadcast {broadcast.signal_id} sent to {len(eligible_users)} users")
            
        except Exception as e:
            logger.error(f"Error processing broadcast: {e}")
    
    def _get_eligible_users(self, broadcast: SignalBroadcast) -> List[str]:
        """Get list of users eligible to receive this broadcast"""
        eligible_users = []
        
        with self.lock:
            for user_id, preferences in self.subscribers.items():
                if self._is_user_eligible(preferences, broadcast):
                    eligible_users.append(user_id)
        
        return eligible_users
    
    def _is_user_eligible(self, preferences: NotificationPreferences, broadcast: SignalBroadcast) -> bool:
        """Check if user is eligible to receive this broadcast"""
        try:
            # Check priority threshold
            if broadcast.priority.value < preferences.min_priority.value:
                return False
            
            # Check symbol filter
            if preferences.symbols and broadcast.symbol not in preferences.symbols:
                return False
            
            # Check strategy filter
            if preferences.strategies and broadcast.strategy_name not in preferences.strategies:
                return False
            
            # Check custom filters
            if preferences.custom_filters:
                if not self._apply_custom_filters(preferences.custom_filters, broadcast):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking user eligibility: {e}")
            return False
    
    def _apply_custom_filters(self, filters: Dict[str, Any], broadcast: SignalBroadcast) -> bool:
        """Apply custom filters to broadcast"""
        try:
            # Min confidence filter
            if "min_confidence" in filters:
                if broadcast.confidence < filters["min_confidence"]:
                    return False
            
            # Min expected return filter
            if "min_expected_return" in filters:
                if broadcast.expected_return < filters["min_expected_return"]:
                    return False
            
            # Max risk filter
            if "max_risk" in filters:
                if broadcast.max_risk > filters["max_risk"]:
                    return False
            
            # Signal type filter
            if "signal_types" in filters:
                if broadcast.signal_type not in filters["signal_types"]:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error applying custom filters: {e}")
            return True  # Default to allowing signal
    
    def _check_rate_limit(self, user_id: str) -> bool:
        """Check if user has exceeded rate limit"""
        try:
            rate_limiter = self.rate_limiters.get(user_id)
            if not rate_limiter:
                return True
            
            current_time = time.time()
            
            # Reset counter if hour has passed
            if current_time > rate_limiter["reset_time"]:
                rate_limiter["count"] = 0
                rate_limiter["reset_time"] = current_time + 3600
            
            # Check if under limit
            preferences = self.subscribers.get(user_id)
            max_notifications = preferences.max_notifications_per_hour if preferences else 10
            
            return rate_limiter["count"] < max_notifications
            
        except Exception as e:
            logger.error(f"Error checking rate limit for user {user_id}: {e}")
            return True
    
    def _update_rate_limiter(self, user_id: str):
        """Update rate limiter for user"""
        try:
            if user_id in self.rate_limiters:
                self.rate_limiters[user_id]["count"] += 1
        except Exception as e:
            logger.error(f"Error updating rate limiter for user {user_id}: {e}")
    
    def _is_quiet_hours(self, preferences: NotificationPreferences) -> bool:
        """Check if current time is within user's quiet hours"""
        try:
            if not preferences.quiet_hours:
                return False
            
            current_time = datetime.now().time()
            start_time = datetime.strptime(preferences.quiet_hours["start"], "%H:%M").time()
            end_time = datetime.strptime(preferences.quiet_hours["end"], "%H:%M").time()
            
            if start_time <= end_time:
                return start_time <= current_time <= end_time
            else:  # Quiet hours span midnight
                return current_time >= start_time or current_time <= end_time
                
        except Exception as e:
            logger.error(f"Error checking quiet hours: {e}")
            return False
    
    def _send_notifications(self, user_id: str, preferences: NotificationPreferences, broadcast: SignalBroadcast):
        """Send notifications through all preferred channels"""
        for channel in preferences.channels:
            try:
                if channel == NotificationChannel.WEBSOCKET:
                    self._send_websocket_notification(user_id, broadcast)
                elif channel == NotificationChannel.EMAIL:
                    self._send_email_notification(preferences.email, broadcast)
                elif channel == NotificationChannel.SMS:
                    self._send_sms_notification(preferences.phone, broadcast)
                elif channel == NotificationChannel.SLACK:
                    self._send_slack_notification(preferences.slack_webhook, broadcast)
                elif channel == NotificationChannel.DISCORD:
                    self._send_discord_notification(preferences.discord_webhook, broadcast)
                elif channel == NotificationChannel.WEBHOOK:
                    self._send_webhook_notification(preferences.custom_webhook, broadcast)
                
            except Exception as e:
                logger.error(f"Error sending {channel.value} notification to user {user_id}: {e}")
    
    def _send_websocket_notification(self, user_id: str, broadcast: SignalBroadcast):
        """Send WebSocket notification"""
        try:
            if user_id in self.active_connections:
                self.socketio.emit('new_signal', broadcast.to_dict(), room=user_id)
                logger.debug(f"WebSocket notification sent to user {user_id}")
        except Exception as e:
            logger.error(f"Error sending WebSocket notification: {e}")
    
    def _send_email_notification(self, email: str, broadcast: SignalBroadcast):
        """Send email notification"""
        if not email:
            return
        
        try:
            # Create email content
            subject = f"🚨 Trading Signal: {broadcast.symbol} - {broadcast.strategy_name}"
            
            html_content = f"""
            <html>
            <body>
                <h2>New Trading Signal Alert</h2>
                <table border="1" cellpadding="5" cellspacing="0">
                    <tr><td><strong>Symbol:</strong></td><td>{broadcast.symbol}</td></tr>
                    <tr><td><strong>Strategy:</strong></td><td>{broadcast.strategy_name}</td></tr>
                    <tr><td><strong>Signal Type:</strong></td><td>{broadcast.signal_type}</td></tr>
                    <tr><td><strong>Priority:</strong></td><td>{broadcast.priority.name}</td></tr>
                    <tr><td><strong>Confidence:</strong></td><td>{broadcast.confidence:.1%}</td></tr>
                    <tr><td><strong>Expected Return:</strong></td><td>{broadcast.expected_return:.1%}</td></tr>
                    <tr><td><strong>Max Risk:</strong></td><td>{broadcast.max_risk:.1%}</td></tr>
                    <tr><td><strong>Entry Price:</strong></td><td>${broadcast.entry_price:.2f}</td></tr>
                    <tr><td><strong>Target Price:</strong></td><td>${broadcast.target_price:.2f}</td></tr>
                    <tr><td><strong>Stop Loss:</strong></td><td>${broadcast.stop_loss:.2f}</td></tr>
                    <tr><td><strong>Time Horizon:</strong></td><td>{broadcast.time_horizon} hours</td></tr>
                </table>
                <p><strong>Reasoning:</strong> {broadcast.reasoning}</p>
                <p><em>Generated at {broadcast.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}</em></p>
            </body>
            </html>
            """
            
            # This would integrate with actual email service
            logger.info(f"Email notification prepared for {email}")
            
        except Exception as e:
            logger.error(f"Error sending email notification: {e}")
    
    def _send_sms_notification(self, phone: str, broadcast: SignalBroadcast):
        """Send SMS notification"""
        if not phone:
            return
        
        try:
            message = (f"🚨 {broadcast.symbol} Signal: {broadcast.strategy_name} "
                      f"({broadcast.confidence:.0%} confidence, {broadcast.expected_return:.1%} return)")
            
            # This would integrate with SMS service (Twilio, etc.)
            logger.info(f"SMS notification prepared for {phone}")
            
        except Exception as e:
            logger.error(f"Error sending SMS notification: {e}")
    
    def _send_slack_notification(self, webhook_url: str, broadcast: SignalBroadcast):
        """Send Slack notification"""
        if not webhook_url:
            return
        
        try:
            payload = {
                "text": f"🚨 New Trading Signal: {broadcast.symbol}",
                "attachments": [
                    {
                        "color": "good" if broadcast.priority.value >= 3 else "warning",
                        "fields": [
                            {"title": "Symbol", "value": broadcast.symbol, "short": True},
                            {"title": "Strategy", "value": broadcast.strategy_name, "short": True},
                            {"title": "Confidence", "value": f"{broadcast.confidence:.1%}", "short": True},
                            {"title": "Expected Return", "value": f"{broadcast.expected_return:.1%}", "short": True},
                            {"title": "Entry Price", "value": f"${broadcast.entry_price:.2f}", "short": True},
                            {"title": "Target Price", "value": f"${broadcast.target_price:.2f}", "short": True}
                        ]
                    }
                ]
            }
            
            response = requests.post(webhook_url, json=payload, timeout=10)
            if response.status_code == 200:
                logger.debug("Slack notification sent successfully")
            else:
                logger.error(f"Slack notification failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error sending Slack notification: {e}")
    
    def _send_discord_notification(self, webhook_url: str, broadcast: SignalBroadcast):
        """Send Discord notification"""
        if not webhook_url:
            return
        
        try:
            embed = {
                "title": f"🚨 Trading Signal: {broadcast.symbol}",
                "description": broadcast.reasoning,
                "color": 0x00ff00 if broadcast.priority.value >= 3 else 0xffaa00,
                "fields": [
                    {"name": "Strategy", "value": broadcast.strategy_name, "inline": True},
                    {"name": "Confidence", "value": f"{broadcast.confidence:.1%}", "inline": True},
                    {"name": "Expected Return", "value": f"{broadcast.expected_return:.1%}", "inline": True},
                    {"name": "Entry Price", "value": f"${broadcast.entry_price:.2f}", "inline": True},
                    {"name": "Target Price", "value": f"${broadcast.target_price:.2f}", "inline": True},
                    {"name": "Stop Loss", "value": f"${broadcast.stop_loss:.2f}", "inline": True}
                ],
                "timestamp": broadcast.timestamp.isoformat()
            }
            
            payload = {"embeds": [embed]}
            
            response = requests.post(webhook_url, json=payload, timeout=10)
            if response.status_code == 204:
                logger.debug("Discord notification sent successfully")
            else:
                logger.error(f"Discord notification failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error sending Discord notification: {e}")
    
    def _send_webhook_notification(self, webhook_url: str, broadcast: SignalBroadcast):
        """Send custom webhook notification"""
        if not webhook_url:
            return
        
        try:
            payload = broadcast.to_dict()
            
            response = requests.post(webhook_url, json=payload, timeout=10)
            if response.status_code == 200:
                logger.debug("Webhook notification sent successfully")
            else:
                logger.error(f"Webhook notification failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error sending webhook notification: {e}")
    
    def _cleanup_worker(self):
        """Background worker for cleanup tasks"""
        while self.running:
            try:
                # Clean up old notification history
                current_time = time.time()
                cutoff_time = current_time - 86400  # 24 hours ago
                
                # Clean up rate limiters
                with self.lock:
                    for user_id, rate_limiter in list(self.rate_limiters.items()):
                        if current_time > rate_limiter["reset_time"]:
                            rate_limiter["count"] = 0
                            rate_limiter["reset_time"] = current_time + 3600
                
                # Sleep for 5 minutes before next cleanup
                time.sleep(300)
                
            except Exception as e:
                logger.error(f"Error in cleanup worker: {e}")
                time.sleep(60)
    
    def _setup_routes(self):
        """Setup Flask routes"""
        @self.app.route('/api/v1/signals/subscribe', methods=['POST'])
        def subscribe():
            try:
                data = request.get_json()
                user_id = data.get('user_id')
                
                preferences = NotificationPreferences(
                    user_id=user_id,
                    channels=[NotificationChannel(ch) for ch in data.get('channels', ['websocket'])],
                    min_priority=SignalPriority(data.get('min_priority', 2)),
                    symbols=data.get('symbols'),
                    strategies=data.get('strategies'),
                    custom_filters=data.get('custom_filters'),
                    quiet_hours=data.get('quiet_hours'),
                    max_notifications_per_hour=data.get('max_notifications_per_hour', 10),
                    email=data.get('email'),
                    phone=data.get('phone'),
                    slack_webhook=data.get('slack_webhook'),
                    discord_webhook=data.get('discord_webhook'),
                    custom_webhook=data.get('custom_webhook')
                )
                
                success = self.subscribe_user(user_id, preferences)
                
                return jsonify({
                    'success': success,
                    'message': 'Subscription successful' if success else 'Subscription failed'
                })
                
            except Exception as e:
                logger.error(f"Error in subscribe route: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/v1/signals/unsubscribe', methods=['POST'])
        def unsubscribe():
            try:
                data = request.get_json()
                user_id = data.get('user_id')
                
                success = self.unsubscribe_user(user_id)
                
                return jsonify({
                    'success': success,
                    'message': 'Unsubscription successful' if success else 'Unsubscription failed'
                })
                
            except Exception as e:
                logger.error(f"Error in unsubscribe route: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/v1/signals/recent', methods=['GET'])
        def get_recent():
            try:
                limit = int(request.args.get('limit', 20))
                signals = self.get_recent_signals(limit)
                
                return jsonify({
                    'success': True,
                    'signals': signals,
                    'count': len(signals)
                })
                
            except Exception as e:
                logger.error(f"Error in get_recent route: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500
    
    def _setup_socketio_events(self):
        """Setup SocketIO event handlers"""
        @self.socketio.on('connect')
        def handle_connect():
            logger.info(f"Client connected: {request.sid}")
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            logger.info(f"Client disconnected: {request.sid}")
        
        @self.socketio.on('join_user_room')
        def handle_join_room(data):
            try:
                user_id = data.get('user_id')
                if user_id:
                    join_room(user_id)
                    
                    # Track active connection
                    with self.lock:
                        if user_id not in self.active_connections:
                            self.active_connections[user_id] = set()
                        self.active_connections[user_id].add(request.sid)
                    
                    emit('joined', {'room': user_id})
                    logger.info(f"Client {request.sid} joined room {user_id}")
                    
            except Exception as e:
                logger.error(f"Error joining room: {e}")
        
        @self.socketio.on('leave_user_room')
        def handle_leave_room(data):
            try:
                user_id = data.get('user_id')
                if user_id:
                    leave_room(user_id)
                    
                    # Remove from active connections
                    with self.lock:
                        if user_id in self.active_connections:
                            self.active_connections[user_id].discard(request.sid)
                            if not self.active_connections[user_id]:
                                del self.active_connections[user_id]
                    
                    emit('left', {'room': user_id})
                    logger.info(f"Client {request.sid} left room {user_id}")
                    
            except Exception as e:
                logger.error(f"Error leaving room: {e}")
    
    def _init_database(self):
        """Initialize SQLite database for persistent storage"""
        try:
            with self._get_db_connection() as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS user_preferences (
                        user_id TEXT PRIMARY KEY,
                        preferences TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS notification_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        signal_id TEXT NOT NULL,
                        channel TEXT NOT NULL,
                        sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        success BOOLEAN NOT NULL
                    )
                ''')
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
    
    @contextmanager
    def _get_db_connection(self):
        """Get database connection with context manager"""
        conn = sqlite3.connect('signal_broadcaster.db')
        try:
            yield conn
        finally:
            conn.close()
    
    def _store_user_preferences(self, user_id: str, preferences: NotificationPreferences):
        """Store user preferences in database"""
        try:
            with self._get_db_connection() as conn:
                preferences_json = json.dumps(asdict(preferences))
                conn.execute('''
                    INSERT OR REPLACE INTO user_preferences (user_id, preferences, updated_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                ''', (user_id, preferences_json))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error storing user preferences: {e}")
    
    def _remove_user_preferences(self, user_id: str):
        """Remove user preferences from database"""
        try:
            with self._get_db_connection() as conn:
                conn.execute('DELETE FROM user_preferences WHERE user_id = ?', (user_id,))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error removing user preferences: {e}")
    
    def run_server(self, host: str = "0.0.0.0", port: int = 8005, debug: bool = False):
        """Run the Flask-SocketIO server"""
        self.start()
        self.socketio.run(self.app, host=host, port=port, debug=debug)

# Factory function
def create_signal_broadcaster(redis_host: str = "localhost", redis_port: int = 6379) -> SignalBroadcaster:
    """Create signal broadcaster"""
    return SignalBroadcaster(redis_host, redis_port)

if __name__ == "__main__":
    # Example usage
    broadcaster = create_signal_broadcaster()
    
    # Example signal broadcast
    signal_data = {
        "signal_id": "test_001",
        "symbol": "AAPL",
        "strategy_name": "Momentum Breakout",
        "signal_type": "BUY_CALL",
        "priority": 3,
        "confidence": 0.75,
        "expected_return": 0.08,
        "max_risk": 0.03,
        "entry_price": 150.0,
        "target_price": 162.0,
        "stop_loss": 145.5,
        "time_horizon": 48,
        "reasoning": "Strong upward breakout with volume confirmation",
        "technical_factors": {"rsi": 65, "volume_ratio": 1.8},
        "options_factors": {"strike": 150, "expiration": "2024-01-19"},
        "risk_factors": {"max_loss": 0.03},
        "score": {"overall_score": 85.5}
    }
    
    # Example user subscription
    preferences = NotificationPreferences(
        user_id="user_001",
        channels=[NotificationChannel.WEBSOCKET, NotificationChannel.EMAIL],
        min_priority=SignalPriority.MEDIUM,
        symbols=["AAPL", "MSFT", "GOOGL"],
        email="user@example.com",
        max_notifications_per_hour=5
    )
    
    broadcaster.subscribe_user("user_001", preferences)
    broadcaster.broadcast_signal(signal_data)
    
    # Run server
    broadcaster.run_server(port=8005, debug=True)

