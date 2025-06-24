"""
Risk Alerting and Notification System
Multi-channel risk alert management and notification delivery

This module provides comprehensive risk alerting capabilities including:
- Real-time risk monitoring and alert generation
- Multi-channel notification delivery (email, SMS, Slack, Discord, webhooks)
- Alert prioritization and escalation
- Alert acknowledgment and resolution tracking
- Notification preferences and filtering
- Alert analytics and reporting

Author: Manus AI
Version: 4.0.0
"""

import os
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json
import uuid
import smtplib
import requests
import threading
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from concurrent.futures import ThreadPoolExecutor
import queue

import redis
import psycopg2
from psycopg2.extras import RealDictCursor
from twilio.rest import Client as TwilioClient

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class AlertType(Enum):
    """Types of risk alerts"""
    POSITION_LIMIT = "position_limit"
    PORTFOLIO_RISK = "portfolio_risk"
    DRAWDOWN = "drawdown"
    VAR_BREACH = "var_breach"
    STRESS_TEST = "stress_test"
    LIQUIDITY = "liquidity"
    CONCENTRATION = "concentration"
    CORRELATION = "correlation"
    VOLATILITY = "volatility"
    SYSTEM = "system"

class NotificationChannel(Enum):
    """Notification delivery channels"""
    EMAIL = "email"
    SMS = "sms"
    SLACK = "slack"
    DISCORD = "discord"
    WEBHOOK = "webhook"
    PUSH = "push"
    IN_APP = "in_app"

class AlertStatus(Enum):
    """Alert status states"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"

@dataclass
class RiskAlert:
    """Risk alert definition"""
    alert_id: str
    portfolio_id: str
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    details: Dict[str, Any]
    
    # Metrics
    current_value: float
    threshold_value: float
    breach_percentage: float
    
    # Timing
    created_at: datetime
    first_breach_at: datetime
    last_updated_at: datetime
    
    # Status
    status: AlertStatus = AlertStatus.ACTIVE
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    
    # Escalation
    escalation_level: int = 0
    escalation_count: int = 0
    next_escalation_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['alert_type'] = self.alert_type.value
        data['severity'] = self.severity.value
        data['status'] = self.status.value
        data['created_at'] = self.created_at.isoformat()
        data['first_breach_at'] = self.first_breach_at.isoformat()
        data['last_updated_at'] = self.last_updated_at.isoformat()
        if self.acknowledged_at:
            data['acknowledged_at'] = self.acknowledged_at.isoformat()
        if self.resolved_at:
            data['resolved_at'] = self.resolved_at.isoformat()
        if self.next_escalation_at:
            data['next_escalation_at'] = self.next_escalation_at.isoformat()
        return data

@dataclass
class NotificationPreferences:
    """User notification preferences"""
    user_id: str
    portfolio_ids: List[str]
    
    # Channel preferences
    enabled_channels: List[NotificationChannel]
    email_address: Optional[str] = None
    phone_number: Optional[str] = None
    slack_webhook: Optional[str] = None
    discord_webhook: Optional[str] = None
    custom_webhook: Optional[str] = None
    
    # Filtering preferences
    min_severity: AlertSeverity = AlertSeverity.WARNING
    alert_types: List[AlertType] = None  # None means all types
    
    # Timing preferences
    quiet_hours_start: Optional[str] = None  # "22:00"
    quiet_hours_end: Optional[str] = None    # "08:00"
    timezone: str = "UTC"
    
    # Rate limiting
    max_alerts_per_hour: int = 10
    max_alerts_per_day: int = 50
    
    # Escalation preferences
    enable_escalation: bool = True
    escalation_delay_minutes: int = 15
    max_escalation_level: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['enabled_channels'] = [ch.value for ch in self.enabled_channels]
        data['min_severity'] = self.min_severity.value
        if self.alert_types:
            data['alert_types'] = [at.value for at in self.alert_types]
        return data

@dataclass
class NotificationDelivery:
    """Notification delivery record"""
    delivery_id: str
    alert_id: str
    user_id: str
    channel: NotificationChannel
    
    # Delivery details
    recipient: str  # email, phone, webhook URL, etc.
    subject: str
    content: str
    
    # Status
    status: str  # pending, sent, delivered, failed
    sent_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    failed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    
    # Metadata
    retry_count: int = 0
    max_retries: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['channel'] = self.channel.value
        if self.sent_at:
            data['sent_at'] = self.sent_at.isoformat()
        if self.delivered_at:
            data['delivered_at'] = self.delivered_at.isoformat()
        if self.failed_at:
            data['failed_at'] = self.failed_at.isoformat()
        return data

class AlertGenerator:
    """Generates risk alerts based on portfolio conditions"""
    
    def __init__(self, db_connection, redis_client):
        self.db_connection = db_connection
        self.redis_client = redis_client
    
    def check_position_limits(self, portfolio_id: str, position_data: Dict[str, Any]) -> List[RiskAlert]:
        """Check for position limit breaches"""
        alerts = []
        
        try:
            # Get position limits
            limits = self._get_position_limits(portfolio_id)
            if not limits:
                return alerts
            
            current_time = datetime.now()
            
            # Check individual position limits
            for position in position_data.get('positions', []):
                symbol = position['symbol']
                market_value = position['market_value']
                portfolio_value = position_data['portfolio_summary']['total_value']
                
                # Check position size limit
                if 'max_position_size' in limits:
                    max_size = limits['max_position_size']
                    if market_value > max_size:
                        breach_pct = (market_value - max_size) / max_size * 100
                        
                        alert = RiskAlert(
                            alert_id=str(uuid.uuid4()),
                            portfolio_id=portfolio_id,
                            alert_type=AlertType.POSITION_LIMIT,
                            severity=AlertSeverity.ERROR if breach_pct > 20 else AlertSeverity.WARNING,
                            title=f"Position Size Limit Breach - {symbol}",
                            message=f"Position {symbol} (${market_value:,.0f}) exceeds limit (${max_size:,.0f}) by {breach_pct:.1f}%",
                            details={
                                'symbol': symbol,
                                'position_value': market_value,
                                'limit_value': max_size,
                                'limit_type': 'position_size'
                            },
                            current_value=market_value,
                            threshold_value=max_size,
                            breach_percentage=breach_pct,
                            created_at=current_time,
                            first_breach_at=current_time,
                            last_updated_at=current_time
                        )
                        alerts.append(alert)
                
                # Check concentration limit
                if 'max_concentration' in limits:
                    concentration = market_value / portfolio_value
                    max_concentration = limits['max_concentration']
                    
                    if concentration > max_concentration:
                        breach_pct = (concentration - max_concentration) / max_concentration * 100
                        
                        alert = RiskAlert(
                            alert_id=str(uuid.uuid4()),
                            portfolio_id=portfolio_id,
                            alert_type=AlertType.CONCENTRATION,
                            severity=AlertSeverity.ERROR if breach_pct > 25 else AlertSeverity.WARNING,
                            title=f"Concentration Limit Breach - {symbol}",
                            message=f"Position {symbol} ({concentration:.1%}) exceeds concentration limit ({max_concentration:.1%}) by {breach_pct:.1f}%",
                            details={
                                'symbol': symbol,
                                'concentration': concentration,
                                'limit_concentration': max_concentration,
                                'limit_type': 'concentration'
                            },
                            current_value=concentration,
                            threshold_value=max_concentration,
                            breach_percentage=breach_pct,
                            created_at=current_time,
                            first_breach_at=current_time,
                            last_updated_at=current_time
                        )
                        alerts.append(alert)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error checking position limits: {e}")
            return []
    
    def check_portfolio_risk(self, portfolio_id: str, risk_metrics: Dict[str, Any]) -> List[RiskAlert]:
        """Check for portfolio risk breaches"""
        alerts = []
        
        try:
            # Get risk limits
            limits = self._get_risk_limits(portfolio_id)
            if not limits:
                return alerts
            
            current_time = datetime.now()
            
            # Check VaR limits
            if 'max_var_95' in limits and 'var_95' in risk_metrics:
                current_var = risk_metrics['var_95']
                max_var = limits['max_var_95']
                
                if current_var > max_var:
                    breach_pct = (current_var - max_var) / max_var * 100
                    
                    alert = RiskAlert(
                        alert_id=str(uuid.uuid4()),
                        portfolio_id=portfolio_id,
                        alert_type=AlertType.VAR_BREACH,
                        severity=AlertSeverity.CRITICAL if breach_pct > 50 else AlertSeverity.ERROR,
                        title="VaR 95% Limit Breach",
                        message=f"Portfolio VaR 95% (${current_var:,.0f}) exceeds limit (${max_var:,.0f}) by {breach_pct:.1f}%",
                        details={
                            'current_var_95': current_var,
                            'limit_var_95': max_var,
                            'portfolio_value': risk_metrics.get('portfolio_value', 0)
                        },
                        current_value=current_var,
                        threshold_value=max_var,
                        breach_percentage=breach_pct,
                        created_at=current_time,
                        first_breach_at=current_time,
                        last_updated_at=current_time
                    )
                    alerts.append(alert)
            
            # Check volatility limits
            if 'max_volatility' in limits and 'volatility' in risk_metrics:
                current_vol = risk_metrics['volatility']
                max_vol = limits['max_volatility']
                
                if current_vol > max_vol:
                    breach_pct = (current_vol - max_vol) / max_vol * 100
                    
                    alert = RiskAlert(
                        alert_id=str(uuid.uuid4()),
                        portfolio_id=portfolio_id,
                        alert_type=AlertType.VOLATILITY,
                        severity=AlertSeverity.WARNING if breach_pct < 25 else AlertSeverity.ERROR,
                        title="Volatility Limit Breach",
                        message=f"Portfolio volatility ({current_vol:.1%}) exceeds limit ({max_vol:.1%}) by {breach_pct:.1f}%",
                        details={
                            'current_volatility': current_vol,
                            'limit_volatility': max_vol
                        },
                        current_value=current_vol,
                        threshold_value=max_vol,
                        breach_percentage=breach_pct,
                        created_at=current_time,
                        first_breach_at=current_time,
                        last_updated_at=current_time
                    )
                    alerts.append(alert)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error checking portfolio risk: {e}")
            return []
    
    def check_drawdown_alerts(self, portfolio_id: str, drawdown_data: Dict[str, Any]) -> List[RiskAlert]:
        """Check for drawdown-related alerts"""
        alerts = []
        
        try:
            current_time = datetime.now()
            current_drawdown = drawdown_data.get('current_drawdown_pct', 0)
            max_drawdown = drawdown_data.get('max_drawdown_pct', 0)
            drawdown_duration = drawdown_data.get('drawdown_duration_days', 0)
            
            # Drawdown severity alerts
            if current_drawdown > 0.05:  # 5% drawdown
                severity = AlertSeverity.WARNING
                if current_drawdown > 0.10:  # 10% drawdown
                    severity = AlertSeverity.ERROR
                if current_drawdown > 0.15:  # 15% drawdown
                    severity = AlertSeverity.CRITICAL
                
                alert = RiskAlert(
                    alert_id=str(uuid.uuid4()),
                    portfolio_id=portfolio_id,
                    alert_type=AlertType.DRAWDOWN,
                    severity=severity,
                    title=f"Portfolio Drawdown Alert - {current_drawdown:.1%}",
                    message=f"Portfolio is in {current_drawdown:.1%} drawdown for {drawdown_duration} days",
                    details={
                        'current_drawdown_pct': current_drawdown,
                        'max_drawdown_pct': max_drawdown,
                        'drawdown_duration_days': drawdown_duration,
                        'portfolio_value': drawdown_data.get('current_value', 0),
                        'peak_value': drawdown_data.get('peak_value', 0)
                    },
                    current_value=current_drawdown,
                    threshold_value=0.05,  # 5% threshold
                    breach_percentage=(current_drawdown - 0.05) / 0.05 * 100,
                    created_at=current_time,
                    first_breach_at=current_time,
                    last_updated_at=current_time
                )
                alerts.append(alert)
            
            # Extended drawdown duration alert
            if drawdown_duration > 30 and current_drawdown > 0.03:  # 30+ days in 3%+ drawdown
                alert = RiskAlert(
                    alert_id=str(uuid.uuid4()),
                    portfolio_id=portfolio_id,
                    alert_type=AlertType.DRAWDOWN,
                    severity=AlertSeverity.WARNING,
                    title=f"Extended Drawdown Duration - {drawdown_duration} days",
                    message=f"Portfolio has been in drawdown for {drawdown_duration} days ({current_drawdown:.1%})",
                    details={
                        'drawdown_duration_days': drawdown_duration,
                        'current_drawdown_pct': current_drawdown
                    },
                    current_value=drawdown_duration,
                    threshold_value=30,
                    breach_percentage=(drawdown_duration - 30) / 30 * 100,
                    created_at=current_time,
                    first_breach_at=current_time,
                    last_updated_at=current_time
                )
                alerts.append(alert)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error checking drawdown alerts: {e}")
            return []
    
    def _get_position_limits(self, portfolio_id: str) -> Optional[Dict[str, Any]]:
        """Get position limits for portfolio"""
        try:
            with self.db_connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT limit_type, limit_value, threshold_value
                    FROM position_limits
                    WHERE portfolio_id = %s AND is_active = true
                """, (portfolio_id,))
                
                limits = {}
                for row in cursor.fetchall():
                    limits[row['limit_type']] = row['limit_value']
                
                return limits if limits else None
                
        except Exception as e:
            logger.error(f"Error getting position limits: {e}")
            return None
    
    def _get_risk_limits(self, portfolio_id: str) -> Optional[Dict[str, Any]]:
        """Get risk limits for portfolio"""
        try:
            with self.db_connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT limit_type, limit_value
                    FROM risk_limits
                    WHERE portfolio_id = %s AND is_active = true
                """, (portfolio_id,))
                
                limits = {}
                for row in cursor.fetchall():
                    limits[row['limit_type']] = row['limit_value']
                
                return limits if limits else None
                
        except Exception as e:
            logger.error(f"Error getting risk limits: {e}")
            return None

class NotificationDeliveryService:
    """Handles notification delivery across multiple channels"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.delivery_queue = queue.Queue()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._setup_clients()
        self._start_delivery_worker()
    
    def _setup_clients(self):
        """Setup notification service clients"""
        # Email configuration
        self.smtp_server = self.config.get('smtp_server', 'smtp.gmail.com')
        self.smtp_port = self.config.get('smtp_port', 587)
        self.smtp_username = self.config.get('smtp_username')
        self.smtp_password = self.config.get('smtp_password')
        
        # Twilio configuration
        self.twilio_account_sid = self.config.get('twilio_account_sid')
        self.twilio_auth_token = self.config.get('twilio_auth_token')
        self.twilio_from_number = self.config.get('twilio_from_number')
        
        if self.twilio_account_sid and self.twilio_auth_token:
            self.twilio_client = TwilioClient(self.twilio_account_sid, self.twilio_auth_token)
        else:
            self.twilio_client = None
    
    def _start_delivery_worker(self):
        """Start background delivery worker"""
        def delivery_worker():
            while True:
                try:
                    delivery = self.delivery_queue.get(timeout=1)
                    if delivery is None:  # Shutdown signal
                        break
                    
                    self._deliver_notification(delivery)
                    self.delivery_queue.task_done()
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Error in delivery worker: {e}")
        
        self.delivery_thread = threading.Thread(target=delivery_worker, daemon=True)
        self.delivery_thread.start()
    
    def queue_notification(self, delivery: NotificationDelivery):
        """Queue notification for delivery"""
        self.delivery_queue.put(delivery)
    
    def _deliver_notification(self, delivery: NotificationDelivery):
        """Deliver notification via specified channel"""
        try:
            if delivery.channel == NotificationChannel.EMAIL:
                self._send_email(delivery)
            elif delivery.channel == NotificationChannel.SMS:
                self._send_sms(delivery)
            elif delivery.channel == NotificationChannel.SLACK:
                self._send_slack(delivery)
            elif delivery.channel == NotificationChannel.DISCORD:
                self._send_discord(delivery)
            elif delivery.channel == NotificationChannel.WEBHOOK:
                self._send_webhook(delivery)
            else:
                logger.warning(f"Unsupported notification channel: {delivery.channel}")
                
        except Exception as e:
            logger.error(f"Error delivering notification {delivery.delivery_id}: {e}")
            delivery.status = "failed"
            delivery.failed_at = datetime.now()
            delivery.error_message = str(e)
    
    def _send_email(self, delivery: NotificationDelivery):
        """Send email notification"""
        if not self.smtp_username or not self.smtp_password:
            raise ValueError("Email configuration not provided")
        
        try:
            msg = MimeMultipart()
            msg['From'] = self.smtp_username
            msg['To'] = delivery.recipient
            msg['Subject'] = delivery.subject
            
            # Create HTML content
            html_content = self._create_email_html(delivery)
            msg.attach(MimeText(html_content, 'html'))
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_username, self.smtp_password)
                server.send_message(msg)
            
            delivery.status = "sent"
            delivery.sent_at = datetime.now()
            
        except Exception as e:
            raise Exception(f"Email delivery failed: {e}")
    
    def _send_sms(self, delivery: NotificationDelivery):
        """Send SMS notification"""
        if not self.twilio_client:
            raise ValueError("Twilio configuration not provided")
        
        try:
            message = self.twilio_client.messages.create(
                body=delivery.content,
                from_=self.twilio_from_number,
                to=delivery.recipient
            )
            
            delivery.status = "sent"
            delivery.sent_at = datetime.now()
            
        except Exception as e:
            raise Exception(f"SMS delivery failed: {e}")
    
    def _send_slack(self, delivery: NotificationDelivery):
        """Send Slack notification"""
        try:
            payload = {
                "text": delivery.subject,
                "attachments": [
                    {
                        "color": self._get_slack_color(delivery),
                        "fields": [
                            {
                                "title": "Alert Details",
                                "value": delivery.content,
                                "short": False
                            }
                        ],
                        "footer": "AI Options Trading System",
                        "ts": int(datetime.now().timestamp())
                    }
                ]
            }
            
            response = requests.post(delivery.recipient, json=payload, timeout=10)
            response.raise_for_status()
            
            delivery.status = "sent"
            delivery.sent_at = datetime.now()
            
        except Exception as e:
            raise Exception(f"Slack delivery failed: {e}")
    
    def _send_discord(self, delivery: NotificationDelivery):
        """Send Discord notification"""
        try:
            payload = {
                "content": delivery.subject,
                "embeds": [
                    {
                        "title": "Risk Alert",
                        "description": delivery.content,
                        "color": self._get_discord_color(delivery),
                        "footer": {
                            "text": "AI Options Trading System"
                        },
                        "timestamp": datetime.now().isoformat()
                    }
                ]
            }
            
            response = requests.post(delivery.recipient, json=payload, timeout=10)
            response.raise_for_status()
            
            delivery.status = "sent"
            delivery.sent_at = datetime.now()
            
        except Exception as e:
            raise Exception(f"Discord delivery failed: {e}")
    
    def _send_webhook(self, delivery: NotificationDelivery):
        """Send webhook notification"""
        try:
            payload = {
                "alert_id": delivery.alert_id,
                "subject": delivery.subject,
                "content": delivery.content,
                "timestamp": datetime.now().isoformat(),
                "user_id": delivery.user_id
            }
            
            response = requests.post(delivery.recipient, json=payload, timeout=10)
            response.raise_for_status()
            
            delivery.status = "sent"
            delivery.sent_at = datetime.now()
            
        except Exception as e:
            raise Exception(f"Webhook delivery failed: {e}")
    
    def _create_email_html(self, delivery: NotificationDelivery) -> str:
        """Create HTML email content"""
        return f"""
        <html>
        <body style="font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5;">
            <div style="max-width: 600px; margin: 0 auto; background-color: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <div style="background-color: #1f2937; color: white; padding: 20px; border-radius: 8px 8px 0 0;">
                    <h1 style="margin: 0; font-size: 24px;">🚨 Risk Alert</h1>
                    <p style="margin: 5px 0 0 0; opacity: 0.8;">AI Options Trading System</p>
                </div>
                <div style="padding: 20px;">
                    <h2 style="color: #dc2626; margin-top: 0;">{delivery.subject}</h2>
                    <div style="background-color: #fef2f2; border-left: 4px solid #dc2626; padding: 15px; margin: 15px 0;">
                        <p style="margin: 0; color: #7f1d1d;">{delivery.content}</p>
                    </div>
                    <p style="color: #6b7280; font-size: 14px; margin-top: 20px;">
                        Alert ID: {delivery.alert_id}<br>
                        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
                    </p>
                </div>
                <div style="background-color: #f9fafb; padding: 15px; border-radius: 0 0 8px 8px; text-align: center;">
                    <p style="margin: 0; color: #6b7280; font-size: 12px;">
                        This is an automated alert from the AI Options Trading System Risk Management Service.
                    </p>
                </div>
            </div>
        </body>
        </html>
        """
    
    def _get_slack_color(self, delivery: NotificationDelivery) -> str:
        """Get Slack color based on alert severity"""
        # Parse severity from alert details (simplified)
        if "CRITICAL" in delivery.content.upper():
            return "danger"
        elif "ERROR" in delivery.content.upper():
            return "warning"
        elif "WARNING" in delivery.content.upper():
            return "warning"
        else:
            return "good"
    
    def _get_discord_color(self, delivery: NotificationDelivery) -> int:
        """Get Discord color based on alert severity"""
        # Parse severity from alert details (simplified)
        if "CRITICAL" in delivery.content.upper():
            return 0xff0000  # Red
        elif "ERROR" in delivery.content.upper():
            return 0xff8c00  # Orange
        elif "WARNING" in delivery.content.upper():
            return 0xffd700  # Yellow
        else:
            return 0x00ff00  # Green

class RiskAlertManager:
    """Main risk alert management system"""
    
    def __init__(self, db_connection, redis_client, notification_config: Dict[str, Any]):
        self.db_connection = db_connection
        self.redis_client = redis_client
        self.alert_generator = AlertGenerator(db_connection, redis_client)
        self.notification_service = NotificationDeliveryService(notification_config)
        
        # Alert processing
        self.active_alerts = {}  # alert_id -> RiskAlert
        self.alert_history = []
        
        # Rate limiting
        self.user_alert_counts = {}  # user_id -> {hour: count, day: count}
        
        # Background monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
    
    def start_monitoring(self):
        """Start background alert monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Risk alert monitoring started")
    
    def stop_monitoring(self):
        """Stop background alert monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("Risk alert monitoring stopped")
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.monitoring_active:
            try:
                # Get all active portfolios
                portfolios = self._get_active_portfolios()
                
                for portfolio_id in portfolios:
                    self._check_portfolio_alerts(portfolio_id)
                
                # Process escalations
                self._process_escalations()
                
                # Clean up old alerts
                self._cleanup_old_alerts()
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _check_portfolio_alerts(self, portfolio_id: str):
        """Check all alert conditions for a portfolio"""
        try:
            # Get portfolio data
            portfolio_data = self._get_portfolio_data(portfolio_id)
            if not portfolio_data:
                return
            
            # Get risk metrics
            risk_metrics = self._get_risk_metrics(portfolio_id)
            if not risk_metrics:
                return
            
            # Get drawdown data
            drawdown_data = self._get_drawdown_data(portfolio_id)
            
            # Generate alerts
            new_alerts = []
            
            # Position limit alerts
            new_alerts.extend(
                self.alert_generator.check_position_limits(portfolio_id, portfolio_data)
            )
            
            # Portfolio risk alerts
            new_alerts.extend(
                self.alert_generator.check_portfolio_risk(portfolio_id, risk_metrics)
            )
            
            # Drawdown alerts
            if drawdown_data:
                new_alerts.extend(
                    self.alert_generator.check_drawdown_alerts(portfolio_id, drawdown_data)
                )
            
            # Process new alerts
            for alert in new_alerts:
                self._process_new_alert(alert)
                
        except Exception as e:
            logger.error(f"Error checking portfolio alerts for {portfolio_id}: {e}")
    
    def _process_new_alert(self, alert: RiskAlert):
        """Process a new alert"""
        try:
            # Check if similar alert already exists
            existing_alert = self._find_similar_alert(alert)
            
            if existing_alert:
                # Update existing alert
                existing_alert.last_updated_at = datetime.now()
                existing_alert.current_value = alert.current_value
                existing_alert.breach_percentage = alert.breach_percentage
                self._update_alert_in_db(existing_alert)
            else:
                # Create new alert
                self.active_alerts[alert.alert_id] = alert
                self._store_alert_in_db(alert)
                
                # Send notifications
                self._send_alert_notifications(alert)
            
        except Exception as e:
            logger.error(f"Error processing new alert: {e}")
    
    def _find_similar_alert(self, alert: RiskAlert) -> Optional[RiskAlert]:
        """Find similar existing alert"""
        for existing_alert in self.active_alerts.values():
            if (existing_alert.portfolio_id == alert.portfolio_id and
                existing_alert.alert_type == alert.alert_type and
                existing_alert.status == AlertStatus.ACTIVE):
                
                # Check if it's the same type of alert
                if alert.alert_type == AlertType.POSITION_LIMIT:
                    if existing_alert.details.get('symbol') == alert.details.get('symbol'):
                        return existing_alert
                elif alert.alert_type in [AlertType.VAR_BREACH, AlertType.VOLATILITY, AlertType.DRAWDOWN]:
                    return existing_alert
        
        return None
    
    def _send_alert_notifications(self, alert: RiskAlert):
        """Send notifications for an alert"""
        try:
            # Get notification preferences for this portfolio
            preferences = self._get_notification_preferences(alert.portfolio_id)
            
            for pref in preferences:
                # Check if user should receive this alert
                if not self._should_send_alert(alert, pref):
                    continue
                
                # Check rate limits
                if not self._check_rate_limits(pref.user_id):
                    continue
                
                # Send notifications for each enabled channel
                for channel in pref.enabled_channels:
                    delivery = self._create_notification_delivery(alert, pref, channel)
                    if delivery:
                        self.notification_service.queue_notification(delivery)
                        self._update_rate_limits(pref.user_id)
                
        except Exception as e:
            logger.error(f"Error sending alert notifications: {e}")
    
    def _should_send_alert(self, alert: RiskAlert, preferences: NotificationPreferences) -> bool:
        """Check if alert should be sent to user"""
        # Check severity
        severity_order = [AlertSeverity.INFO, AlertSeverity.WARNING, AlertSeverity.ERROR, AlertSeverity.CRITICAL]
        if severity_order.index(alert.severity) < severity_order.index(preferences.min_severity):
            return False
        
        # Check alert types
        if preferences.alert_types and alert.alert_type not in preferences.alert_types:
            return False
        
        # Check portfolio
        if alert.portfolio_id not in preferences.portfolio_ids:
            return False
        
        # Check quiet hours
        if self._is_quiet_hours(preferences):
            return alert.severity == AlertSeverity.CRITICAL
        
        return True
    
    def _is_quiet_hours(self, preferences: NotificationPreferences) -> bool:
        """Check if current time is in quiet hours"""
        if not preferences.quiet_hours_start or not preferences.quiet_hours_end:
            return False
        
        # Simplified quiet hours check (assumes same day)
        current_time = datetime.now().strftime("%H:%M")
        return preferences.quiet_hours_start <= current_time <= preferences.quiet_hours_end
    
    def _check_rate_limits(self, user_id: str) -> bool:
        """Check if user is within rate limits"""
        now = datetime.now()
        hour_key = now.strftime("%Y-%m-%d-%H")
        day_key = now.strftime("%Y-%m-%d")
        
        if user_id not in self.user_alert_counts:
            self.user_alert_counts[user_id] = {}
        
        user_counts = self.user_alert_counts[user_id]
        
        # Check hourly limit
        hourly_count = user_counts.get(f"hour_{hour_key}", 0)
        if hourly_count >= 10:  # Default hourly limit
            return False
        
        # Check daily limit
        daily_count = user_counts.get(f"day_{day_key}", 0)
        if daily_count >= 50:  # Default daily limit
            return False
        
        return True
    
    def _update_rate_limits(self, user_id: str):
        """Update rate limit counters"""
        now = datetime.now()
        hour_key = now.strftime("%Y-%m-%d-%H")
        day_key = now.strftime("%Y-%m-%d")
        
        if user_id not in self.user_alert_counts:
            self.user_alert_counts[user_id] = {}
        
        user_counts = self.user_alert_counts[user_id]
        user_counts[f"hour_{hour_key}"] = user_counts.get(f"hour_{hour_key}", 0) + 1
        user_counts[f"day_{day_key}"] = user_counts.get(f"day_{day_key}", 0) + 1
    
    def _create_notification_delivery(self, alert: RiskAlert, preferences: NotificationPreferences,
                                    channel: NotificationChannel) -> Optional[NotificationDelivery]:
        """Create notification delivery record"""
        try:
            # Get recipient based on channel
            recipient = None
            if channel == NotificationChannel.EMAIL:
                recipient = preferences.email_address
            elif channel == NotificationChannel.SMS:
                recipient = preferences.phone_number
            elif channel == NotificationChannel.SLACK:
                recipient = preferences.slack_webhook
            elif channel == NotificationChannel.DISCORD:
                recipient = preferences.discord_webhook
            elif channel == NotificationChannel.WEBHOOK:
                recipient = preferences.custom_webhook
            
            if not recipient:
                return None
            
            # Create subject and content
            subject = f"🚨 {alert.severity.value.upper()}: {alert.title}"
            content = f"{alert.message}\n\nPortfolio: {alert.portfolio_id}\nAlert ID: {alert.alert_id}"
            
            return NotificationDelivery(
                delivery_id=str(uuid.uuid4()),
                alert_id=alert.alert_id,
                user_id=preferences.user_id,
                channel=channel,
                recipient=recipient,
                subject=subject,
                content=content,
                status="pending"
            )
            
        except Exception as e:
            logger.error(f"Error creating notification delivery: {e}")
            return None
    
    def _process_escalations(self):
        """Process alert escalations"""
        try:
            current_time = datetime.now()
            
            for alert in self.active_alerts.values():
                if (alert.status == AlertStatus.ACTIVE and
                    alert.next_escalation_at and
                    current_time >= alert.next_escalation_at):
                    
                    # Escalate alert
                    alert.escalation_level += 1
                    alert.escalation_count += 1
                    alert.next_escalation_at = current_time + timedelta(minutes=15)
                    
                    # Send escalated notifications
                    self._send_escalated_notifications(alert)
                    
                    # Update in database
                    self._update_alert_in_db(alert)
                    
        except Exception as e:
            logger.error(f"Error processing escalations: {e}")
    
    def _send_escalated_notifications(self, alert: RiskAlert):
        """Send escalated alert notifications"""
        # Escalated alerts bypass some rate limits and quiet hours
        try:
            preferences = self._get_notification_preferences(alert.portfolio_id)
            
            for pref in preferences:
                if not pref.enable_escalation:
                    continue
                
                if alert.escalation_level > pref.max_escalation_level:
                    continue
                
                # Send to all enabled channels for escalated alerts
                for channel in pref.enabled_channels:
                    delivery = self._create_escalated_notification_delivery(alert, pref, channel)
                    if delivery:
                        self.notification_service.queue_notification(delivery)
                        
        except Exception as e:
            logger.error(f"Error sending escalated notifications: {e}")
    
    def _create_escalated_notification_delivery(self, alert: RiskAlert, preferences: NotificationPreferences,
                                              channel: NotificationChannel) -> Optional[NotificationDelivery]:
        """Create escalated notification delivery"""
        delivery = self._create_notification_delivery(alert, preferences, channel)
        if delivery:
            delivery.subject = f"🔥 ESCALATED {delivery.subject}"
            delivery.content = f"ESCALATION LEVEL {alert.escalation_level}\n\n{delivery.content}"
        return delivery
    
    def acknowledge_alert(self, alert_id: str, user_id: str) -> bool:
        """Acknowledge an alert"""
        try:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_by = user_id
                alert.acknowledged_at = datetime.now()
                
                self._update_alert_in_db(alert)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error acknowledging alert: {e}")
            return False
    
    def resolve_alert(self, alert_id: str, user_id: str) -> bool:
        """Resolve an alert"""
        try:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.status = AlertStatus.RESOLVED
                alert.resolved_at = datetime.now()
                
                self._update_alert_in_db(alert)
                
                # Move to history
                self.alert_history.append(alert)
                del self.active_alerts[alert_id]
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error resolving alert: {e}")
            return False
    
    def get_active_alerts(self, portfolio_id: Optional[str] = None) -> List[RiskAlert]:
        """Get active alerts"""
        alerts = list(self.active_alerts.values())
        
        if portfolio_id:
            alerts = [alert for alert in alerts if alert.portfolio_id == portfolio_id]
        
        return sorted(alerts, key=lambda x: x.created_at, reverse=True)
    
    def get_alert_statistics(self, portfolio_id: Optional[str] = None, days: int = 30) -> Dict[str, Any]:
        """Get alert statistics"""
        try:
            with self.db_connection.cursor(cursor_factory=RealDictCursor) as cursor:
                where_clause = "WHERE created_at >= %s"
                params = [datetime.now() - timedelta(days=days)]
                
                if portfolio_id:
                    where_clause += " AND portfolio_id = %s"
                    params.append(portfolio_id)
                
                # Total alerts
                cursor.execute(f"""
                    SELECT COUNT(*) as total_alerts,
                           COUNT(CASE WHEN severity = 'critical' THEN 1 END) as critical_alerts,
                           COUNT(CASE WHEN severity = 'error' THEN 1 END) as error_alerts,
                           COUNT(CASE WHEN severity = 'warning' THEN 1 END) as warning_alerts,
                           COUNT(CASE WHEN status = 'resolved' THEN 1 END) as resolved_alerts
                    FROM risk_alerts {where_clause}
                """, params)
                
                stats = dict(cursor.fetchone())
                
                # Alert types
                cursor.execute(f"""
                    SELECT alert_type, COUNT(*) as count
                    FROM risk_alerts {where_clause}
                    GROUP BY alert_type
                    ORDER BY count DESC
                """, params)
                
                stats['alert_types'] = {row['alert_type']: row['count'] for row in cursor.fetchall()}
                
                return stats
                
        except Exception as e:
            logger.error(f"Error getting alert statistics: {e}")
            return {}
    
    def _get_active_portfolios(self) -> List[str]:
        """Get list of active portfolio IDs"""
        try:
            with self.db_connection.cursor() as cursor:
                cursor.execute("SELECT portfolio_id FROM portfolios WHERE status = 'active'")
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting active portfolios: {e}")
            return []
    
    def _get_portfolio_data(self, portfolio_id: str) -> Optional[Dict[str, Any]]:
        """Get portfolio data for alert checking"""
        # This would integrate with the portfolio service
        # For now, return mock data
        return {
            'portfolio_summary': {'total_value': 125000},
            'positions': [
                {'symbol': 'AAPL', 'market_value': 25000},
                {'symbol': 'MSFT', 'market_value': 20000}
            ]
        }
    
    def _get_risk_metrics(self, portfolio_id: str) -> Optional[Dict[str, Any]]:
        """Get risk metrics for alert checking"""
        # This would integrate with the risk service
        # For now, return mock data
        return {
            'portfolio_value': 125000,
            'var_95': 2500,
            'var_99': 3750,
            'volatility': 0.18
        }
    
    def _get_drawdown_data(self, portfolio_id: str) -> Optional[Dict[str, Any]]:
        """Get drawdown data for alert checking"""
        # This would integrate with the drawdown protection service
        # For now, return mock data
        return {
            'current_value': 125000,
            'peak_value': 142500,
            'current_drawdown_pct': 0.123,
            'max_drawdown_pct': 0.15,
            'drawdown_duration_days': 15
        }
    
    def _get_notification_preferences(self, portfolio_id: str) -> List[NotificationPreferences]:
        """Get notification preferences for portfolio"""
        try:
            with self.db_connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT user_id, enabled_channels, email_address, phone_number,
                           slack_webhook, discord_webhook, custom_webhook,
                           min_severity, alert_types, quiet_hours_start, quiet_hours_end,
                           max_alerts_per_hour, max_alerts_per_day,
                           enable_escalation, escalation_delay_minutes, max_escalation_level
                    FROM notification_preferences
                    WHERE %s = ANY(portfolio_ids) AND is_active = true
                """, (portfolio_id,))
                
                preferences = []
                for row in cursor.fetchall():
                    pref = NotificationPreferences(
                        user_id=row['user_id'],
                        portfolio_ids=[portfolio_id],
                        enabled_channels=[NotificationChannel(ch) for ch in row['enabled_channels']],
                        email_address=row['email_address'],
                        phone_number=row['phone_number'],
                        slack_webhook=row['slack_webhook'],
                        discord_webhook=row['discord_webhook'],
                        custom_webhook=row['custom_webhook'],
                        min_severity=AlertSeverity(row['min_severity']),
                        alert_types=[AlertType(at) for at in row['alert_types']] if row['alert_types'] else None,
                        quiet_hours_start=row['quiet_hours_start'],
                        quiet_hours_end=row['quiet_hours_end'],
                        max_alerts_per_hour=row['max_alerts_per_hour'],
                        max_alerts_per_day=row['max_alerts_per_day'],
                        enable_escalation=row['enable_escalation'],
                        escalation_delay_minutes=row['escalation_delay_minutes'],
                        max_escalation_level=row['max_escalation_level']
                    )
                    preferences.append(pref)
                
                return preferences
                
        except Exception as e:
            logger.error(f"Error getting notification preferences: {e}")
            return []
    
    def _store_alert_in_db(self, alert: RiskAlert):
        """Store alert in database"""
        try:
            with self.db_connection.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO risk_alerts (
                        alert_id, portfolio_id, alert_type, severity, title, message, details,
                        current_value, threshold_value, breach_percentage,
                        created_at, first_breach_at, last_updated_at, status,
                        escalation_level, escalation_count, next_escalation_at
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                """, (
                    alert.alert_id, alert.portfolio_id, alert.alert_type.value, alert.severity.value,
                    alert.title, alert.message, json.dumps(alert.details),
                    alert.current_value, alert.threshold_value, alert.breach_percentage,
                    alert.created_at, alert.first_breach_at, alert.last_updated_at, alert.status.value,
                    alert.escalation_level, alert.escalation_count, alert.next_escalation_at
                ))
                self.db_connection.commit()
        except Exception as e:
            logger.error(f"Error storing alert in database: {e}")
            self.db_connection.rollback()
    
    def _update_alert_in_db(self, alert: RiskAlert):
        """Update alert in database"""
        try:
            with self.db_connection.cursor() as cursor:
                cursor.execute("""
                    UPDATE risk_alerts SET
                        current_value = %s, breach_percentage = %s, last_updated_at = %s,
                        status = %s, acknowledged_by = %s, acknowledged_at = %s, resolved_at = %s,
                        escalation_level = %s, escalation_count = %s, next_escalation_at = %s
                    WHERE alert_id = %s
                """, (
                    alert.current_value, alert.breach_percentage, alert.last_updated_at,
                    alert.status.value, alert.acknowledged_by, alert.acknowledged_at, alert.resolved_at,
                    alert.escalation_level, alert.escalation_count, alert.next_escalation_at,
                    alert.alert_id
                ))
                self.db_connection.commit()
        except Exception as e:
            logger.error(f"Error updating alert in database: {e}")
            self.db_connection.rollback()
    
    def _cleanup_old_alerts(self):
        """Clean up old resolved alerts"""
        try:
            cutoff_date = datetime.now() - timedelta(days=30)
            
            with self.db_connection.cursor() as cursor:
                cursor.execute("""
                    DELETE FROM risk_alerts
                    WHERE status = 'resolved' AND resolved_at < %s
                """, (cutoff_date,))
                self.db_connection.commit()
                
        except Exception as e:
            logger.error(f"Error cleaning up old alerts: {e}")
            self.db_connection.rollback()

