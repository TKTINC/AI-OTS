"""
Risk Dashboard and Visualization
Real-time risk displays and interactive dashboards

This module provides comprehensive risk dashboard and visualization capabilities including:
- Real-time risk metric displays
- Interactive risk charts and graphs
- Portfolio risk heatmaps
- Stress testing visualizations
- Compliance status dashboards
- Alert and notification displays
- Historical risk trend analysis
- Risk attribution breakdowns

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
import threading
from pathlib import Path

import pandas as pd
import numpy as np
import redis
import psycopg2
from psycopg2.extras import RealDictCursor
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.utils

logger = logging.getLogger(__name__)

class DashboardType(Enum):
    """Types of risk dashboards"""
    OVERVIEW = "overview"
    DETAILED = "detailed"
    COMPLIANCE = "compliance"
    STRESS_TEST = "stress_test"
    ALERTS = "alerts"
    HISTORICAL = "historical"
    ATTRIBUTION = "attribution"
    REAL_TIME = "real_time"

class ChartType(Enum):
    """Types of charts and visualizations"""
    LINE_CHART = "line_chart"
    BAR_CHART = "bar_chart"
    HEATMAP = "heatmap"
    GAUGE = "gauge"
    SCATTER = "scatter"
    PIE_CHART = "pie_chart"
    CANDLESTICK = "candlestick"
    HISTOGRAM = "histogram"
    BOX_PLOT = "box_plot"
    TREEMAP = "treemap"

class TimeFrame(Enum):
    """Time frames for dashboard data"""
    REAL_TIME = "real_time"
    INTRADAY = "intraday"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"

@dataclass
class DashboardWidget:
    """Dashboard widget configuration"""
    widget_id: str
    widget_type: str
    title: str
    chart_type: ChartType
    data_source: str
    refresh_interval: int  # seconds
    
    # Layout
    position: Dict[str, int]  # x, y, width, height
    
    # Configuration
    config: Dict[str, Any]
    filters: Dict[str, Any]
    
    # Data
    data: Optional[Dict[str, Any]] = None
    last_updated: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['chart_type'] = self.chart_type.value
        if self.last_updated:
            data['last_updated'] = self.last_updated.isoformat()
        return data

@dataclass
class Dashboard:
    """Dashboard configuration"""
    dashboard_id: str
    dashboard_type: DashboardType
    title: str
    description: str
    
    # Layout
    layout: Dict[str, Any]
    widgets: List[DashboardWidget]
    
    # Access control
    user_id: str
    is_public: bool = False
    
    # Metadata
    created_at: datetime
    updated_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['dashboard_type'] = self.dashboard_type.value
        data['widgets'] = [w.to_dict() for w in self.widgets]
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        return data

class RiskVisualizationEngine:
    """Generates risk visualizations and charts"""
    
    def __init__(self, db_connection, redis_client):
        self.db_connection = db_connection
        self.redis_client = redis_client
    
    def create_risk_overview_chart(self, portfolio_id: str, timeframe: TimeFrame = TimeFrame.DAILY) -> Dict[str, Any]:
        """Create risk overview chart"""
        try:
            # Get risk metrics data
            risk_data = self._get_risk_metrics_data(portfolio_id, timeframe)
            
            if not risk_data:
                return {'error': 'No risk data available'}
            
            # Create subplot figure
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('VaR Trends', 'Portfolio Value', 'Volatility', 'Drawdown'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # VaR trends
            fig.add_trace(
                go.Scatter(
                    x=risk_data['dates'],
                    y=risk_data['var_95'],
                    name='VaR 95%',
                    line=dict(color='red', width=2)
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=risk_data['dates'],
                    y=risk_data['var_99'],
                    name='VaR 99%',
                    line=dict(color='darkred', width=2)
                ),
                row=1, col=1
            )
            
            # Portfolio value
            fig.add_trace(
                go.Scatter(
                    x=risk_data['dates'],
                    y=risk_data['portfolio_value'],
                    name='Portfolio Value',
                    line=dict(color='blue', width=2),
                    fill='tonexty'
                ),
                row=1, col=2
            )
            
            # Volatility
            fig.add_trace(
                go.Scatter(
                    x=risk_data['dates'],
                    y=risk_data['volatility'],
                    name='Volatility',
                    line=dict(color='orange', width=2)
                ),
                row=2, col=1
            )
            
            # Drawdown
            fig.add_trace(
                go.Scatter(
                    x=risk_data['dates'],
                    y=risk_data['drawdown'],
                    name='Drawdown',
                    line=dict(color='purple', width=2),
                    fill='tozeroy'
                ),
                row=2, col=2
            )
            
            # Update layout
            fig.update_layout(
                title=f"Risk Overview - Portfolio {portfolio_id}",
                height=600,
                showlegend=True,
                template="plotly_white"
            )
            
            # Update axes
            fig.update_xaxes(title_text="Date")
            fig.update_yaxes(title_text="VaR ($)", row=1, col=1)
            fig.update_yaxes(title_text="Value ($)", row=1, col=2)
            fig.update_yaxes(title_text="Volatility (%)", row=2, col=1)
            fig.update_yaxes(title_text="Drawdown (%)", row=2, col=2)
            
            return {
                'chart_data': json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig)),
                'chart_type': 'risk_overview',
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error creating risk overview chart: {e}")
            return {'error': str(e)}
    
    def create_risk_gauge_chart(self, portfolio_id: str, metric: str) -> Dict[str, Any]:
        """Create risk gauge chart"""
        try:
            # Get current risk metrics
            current_metrics = self._get_current_risk_metrics(portfolio_id)
            
            if not current_metrics:
                return {'error': 'No current risk metrics available'}
            
            # Get metric configuration
            metric_config = self._get_metric_config(metric)
            
            current_value = current_metrics.get(metric, 0)
            limit_value = metric_config.get('limit', 100)
            warning_threshold = metric_config.get('warning_threshold', 0.8)
            
            # Determine color based on value
            if current_value <= limit_value * 0.6:
                color = "green"
            elif current_value <= limit_value * warning_threshold:
                color = "yellow"
            else:
                color = "red"
            
            # Create gauge chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = current_value,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': metric_config.get('title', metric.upper())},
                delta = {'reference': limit_value * 0.5},
                gauge = {
                    'axis': {'range': [None, limit_value]},
                    'bar': {'color': color},
                    'steps': [
                        {'range': [0, limit_value * 0.6], 'color': "lightgray"},
                        {'range': [limit_value * 0.6, limit_value * warning_threshold], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': limit_value
                    }
                }
            ))
            
            fig.update_layout(
                height=400,
                template="plotly_white"
            )
            
            return {
                'chart_data': json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig)),
                'chart_type': 'gauge',
                'metric': metric,
                'current_value': current_value,
                'limit_value': limit_value,
                'status': color,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error creating risk gauge chart: {e}")
            return {'error': str(e)}
    
    def create_position_heatmap(self, portfolio_id: str) -> Dict[str, Any]:
        """Create position risk heatmap"""
        try:
            # Get position data
            position_data = self._get_position_risk_data(portfolio_id)
            
            if not position_data:
                return {'error': 'No position data available'}
            
            # Prepare data for heatmap
            symbols = position_data['symbols']
            risk_metrics = ['VaR', 'Volatility', 'Concentration', 'Liquidity']
            
            # Create matrix
            z_data = []
            for metric in risk_metrics:
                row = []
                for symbol in symbols:
                    value = position_data['risk_matrix'].get(symbol, {}).get(metric.lower(), 0)
                    row.append(value)
                z_data.append(row)
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=z_data,
                x=symbols,
                y=risk_metrics,
                colorscale='RdYlGn_r',
                text=z_data,
                texttemplate="%{text:.2f}",
                textfont={"size": 10},
                hoverongaps=False
            ))
            
            fig.update_layout(
                title=f"Position Risk Heatmap - Portfolio {portfolio_id}",
                xaxis_title="Symbols",
                yaxis_title="Risk Metrics",
                height=400,
                template="plotly_white"
            )
            
            return {
                'chart_data': json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig)),
                'chart_type': 'heatmap',
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error creating position heatmap: {e}")
            return {'error': str(e)}
    
    def create_stress_test_chart(self, portfolio_id: str) -> Dict[str, Any]:
        """Create stress test results chart"""
        try:
            # Get stress test data
            stress_data = self._get_stress_test_data(portfolio_id)
            
            if not stress_data:
                return {'error': 'No stress test data available'}
            
            scenarios = stress_data['scenarios']
            losses = stress_data['losses']
            probabilities = stress_data['probabilities']
            
            # Create bar chart
            fig = go.Figure()
            
            # Add bars with color coding
            colors = ['green' if loss < 0.05 else 'yellow' if loss < 0.15 else 'red' for loss in losses]
            
            fig.add_trace(go.Bar(
                x=scenarios,
                y=losses,
                marker_color=colors,
                text=[f"{loss:.1%}" for loss in losses],
                textposition='auto',
                name='Portfolio Loss'
            ))
            
            # Add probability line
            fig.add_trace(go.Scatter(
                x=scenarios,
                y=[p * max(losses) for p in probabilities],  # Scale probabilities
                mode='lines+markers',
                name='Probability (scaled)',
                yaxis='y2',
                line=dict(color='blue', width=2)
            ))
            
            # Update layout
            fig.update_layout(
                title=f"Stress Test Results - Portfolio {portfolio_id}",
                xaxis_title="Scenarios",
                yaxis_title="Portfolio Loss (%)",
                yaxis2=dict(
                    title="Probability",
                    overlaying='y',
                    side='right'
                ),
                height=500,
                template="plotly_white",
                showlegend=True
            )
            
            return {
                'chart_data': json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig)),
                'chart_type': 'stress_test',
                'worst_case_loss': max(losses),
                'scenarios_failed': len([l for l in losses if l > 0.20]),
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error creating stress test chart: {e}")
            return {'error': str(e)}
    
    def create_compliance_status_chart(self, portfolio_id: str) -> Dict[str, Any]:
        """Create compliance status chart"""
        try:
            # Get compliance data
            compliance_data = self._get_compliance_data(portfolio_id)
            
            if not compliance_data:
                return {'error': 'No compliance data available'}
            
            # Prepare data
            frameworks = compliance_data['frameworks']
            compliance_scores = compliance_data['scores']
            violation_counts = compliance_data['violations']
            
            # Create subplot
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Compliance Scores', 'Violation Counts'),
                specs=[[{"type": "bar"}, {"type": "bar"}]]
            )
            
            # Compliance scores
            colors = ['green' if score >= 0.9 else 'yellow' if score >= 0.7 else 'red' for score in compliance_scores]
            
            fig.add_trace(
                go.Bar(
                    x=frameworks,
                    y=compliance_scores,
                    marker_color=colors,
                    text=[f"{score:.1%}" for score in compliance_scores],
                    textposition='auto',
                    name='Compliance Score'
                ),
                row=1, col=1
            )
            
            # Violation counts
            fig.add_trace(
                go.Bar(
                    x=frameworks,
                    y=violation_counts,
                    marker_color='red',
                    text=violation_counts,
                    textposition='auto',
                    name='Violations'
                ),
                row=1, col=2
            )
            
            # Update layout
            fig.update_layout(
                title=f"Compliance Status - Portfolio {portfolio_id}",
                height=400,
                template="plotly_white",
                showlegend=False
            )
            
            fig.update_yaxes(title_text="Score", row=1, col=1)
            fig.update_yaxes(title_text="Count", row=1, col=2)
            
            return {
                'chart_data': json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig)),
                'chart_type': 'compliance_status',
                'overall_score': sum(compliance_scores) / len(compliance_scores),
                'total_violations': sum(violation_counts),
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error creating compliance status chart: {e}")
            return {'error': str(e)}
    
    def create_alert_timeline_chart(self, portfolio_id: str, days: int = 7) -> Dict[str, Any]:
        """Create alert timeline chart"""
        try:
            # Get alert data
            alert_data = self._get_alert_timeline_data(portfolio_id, days)
            
            if not alert_data:
                return {'error': 'No alert data available'}
            
            # Create timeline chart
            fig = go.Figure()
            
            # Add traces for each severity level
            severities = ['INFO', 'WARNING', 'ERROR', 'CRITICAL']
            colors = {'INFO': 'blue', 'WARNING': 'orange', 'ERROR': 'red', 'CRITICAL': 'darkred'}
            
            for severity in severities:
                severity_data = [d for d in alert_data if d['severity'] == severity]
                if severity_data:
                    fig.add_trace(go.Scatter(
                        x=[d['timestamp'] for d in severity_data],
                        y=[severity] * len(severity_data),
                        mode='markers',
                        marker=dict(
                            color=colors[severity],
                            size=10,
                            symbol='circle'
                        ),
                        name=severity,
                        text=[d['title'] for d in severity_data],
                        hovertemplate='<b>%{text}</b><br>Time: %{x}<br>Severity: %{y}<extra></extra>'
                    ))
            
            # Update layout
            fig.update_layout(
                title=f"Alert Timeline - Portfolio {portfolio_id} (Last {days} days)",
                xaxis_title="Time",
                yaxis_title="Severity",
                height=400,
                template="plotly_white",
                showlegend=True
            )
            
            return {
                'chart_data': json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig)),
                'chart_type': 'alert_timeline',
                'total_alerts': len(alert_data),
                'critical_alerts': len([d for d in alert_data if d['severity'] == 'CRITICAL']),
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error creating alert timeline chart: {e}")
            return {'error': str(e)}
    
    def _get_risk_metrics_data(self, portfolio_id: str, timeframe: TimeFrame) -> Optional[Dict[str, Any]]:
        """Get risk metrics data for visualization"""
        try:
            # Determine date range based on timeframe
            if timeframe == TimeFrame.DAILY:
                start_date = datetime.now() - timedelta(days=30)
            elif timeframe == TimeFrame.WEEKLY:
                start_date = datetime.now() - timedelta(weeks=12)
            elif timeframe == TimeFrame.MONTHLY:
                start_date = datetime.now() - timedelta(days=365)
            else:
                start_date = datetime.now() - timedelta(days=7)
            
            # Mock data - would integrate with risk service
            dates = pd.date_range(start=start_date, end=datetime.now(), freq='D')
            
            # Generate realistic mock data
            np.random.seed(42)
            base_value = 125000
            volatility = 0.02
            
            portfolio_values = []
            var_95_values = []
            var_99_values = []
            volatility_values = []
            drawdown_values = []
            
            peak_value = base_value
            
            for i, date in enumerate(dates):
                # Portfolio value with random walk
                daily_return = np.random.normal(0.0005, volatility)
                if i == 0:
                    portfolio_value = base_value
                else:
                    portfolio_value = portfolio_values[-1] * (1 + daily_return)
                
                portfolio_values.append(portfolio_value)
                
                # Update peak
                if portfolio_value > peak_value:
                    peak_value = portfolio_value
                
                # Calculate metrics
                var_95 = portfolio_value * 0.02  # 2% VaR
                var_99 = portfolio_value * 0.03  # 3% VaR
                vol = abs(daily_return) * 100  # Daily volatility as percentage
                drawdown = (peak_value - portfolio_value) / peak_value * 100
                
                var_95_values.append(var_95)
                var_99_values.append(var_99)
                volatility_values.append(vol)
                drawdown_values.append(drawdown)
            
            return {
                'dates': [d.strftime('%Y-%m-%d') for d in dates],
                'portfolio_value': portfolio_values,
                'var_95': var_95_values,
                'var_99': var_99_values,
                'volatility': volatility_values,
                'drawdown': drawdown_values
            }
            
        except Exception as e:
            logger.error(f"Error getting risk metrics data: {e}")
            return None
    
    def _get_current_risk_metrics(self, portfolio_id: str) -> Optional[Dict[str, Any]]:
        """Get current risk metrics"""
        # Mock data - would integrate with risk service
        return {
            'var_95': 2500,
            'var_99': 3750,
            'volatility': 18.5,
            'drawdown': 8.2,
            'leverage': 2.3,
            'concentration': 15.8
        }
    
    def _get_metric_config(self, metric: str) -> Dict[str, Any]:
        """Get metric configuration"""
        configs = {
            'var_95': {
                'title': 'VaR 95%',
                'limit': 5000,
                'warning_threshold': 0.8,
                'unit': '$'
            },
            'var_99': {
                'title': 'VaR 99%',
                'limit': 7500,
                'warning_threshold': 0.8,
                'unit': '$'
            },
            'volatility': {
                'title': 'Volatility',
                'limit': 25,
                'warning_threshold': 0.8,
                'unit': '%'
            },
            'drawdown': {
                'title': 'Drawdown',
                'limit': 20,
                'warning_threshold': 0.8,
                'unit': '%'
            },
            'leverage': {
                'title': 'Leverage',
                'limit': 3.0,
                'warning_threshold': 0.8,
                'unit': 'x'
            },
            'concentration': {
                'title': 'Concentration',
                'limit': 20,
                'warning_threshold': 0.8,
                'unit': '%'
            }
        }
        return configs.get(metric, {'title': metric, 'limit': 100, 'warning_threshold': 0.8})
    
    def _get_position_risk_data(self, portfolio_id: str) -> Optional[Dict[str, Any]]:
        """Get position risk data"""
        # Mock data - would integrate with portfolio service
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
        
        risk_matrix = {}
        for symbol in symbols:
            risk_matrix[symbol] = {
                'var': np.random.uniform(0.5, 2.0),
                'volatility': np.random.uniform(15, 35),
                'concentration': np.random.uniform(5, 25),
                'liquidity': np.random.uniform(0.1, 1.0)
            }
        
        return {
            'symbols': symbols,
            'risk_matrix': risk_matrix
        }
    
    def _get_stress_test_data(self, portfolio_id: str) -> Optional[Dict[str, Any]]:
        """Get stress test data"""
        # Mock data - would integrate with stress testing service
        scenarios = ['Market Crash', 'Vol Spike', 'Rate Hike', 'Sector Rotation', 'Liquidity Crisis']
        losses = [0.18, 0.12, 0.08, 0.06, 0.25]
        probabilities = [0.02, 0.05, 0.15, 0.20, 0.01]
        
        return {
            'scenarios': scenarios,
            'losses': losses,
            'probabilities': probabilities
        }
    
    def _get_compliance_data(self, portfolio_id: str) -> Optional[Dict[str, Any]]:
        """Get compliance data"""
        # Mock data - would integrate with compliance service
        frameworks = ['SEC', 'FINRA', 'Basel III', 'Dodd-Frank']
        scores = [0.95, 0.88, 0.92, 0.85]
        violations = [0, 2, 1, 3]
        
        return {
            'frameworks': frameworks,
            'scores': scores,
            'violations': violations
        }
    
    def _get_alert_timeline_data(self, portfolio_id: str, days: int) -> List[Dict[str, Any]]:
        """Get alert timeline data"""
        # Mock data - would integrate with alerting service
        alerts = []
        
        for i in range(20):
            alert_time = datetime.now() - timedelta(
                days=np.random.randint(0, days),
                hours=np.random.randint(0, 24),
                minutes=np.random.randint(0, 60)
            )
            
            severity = np.random.choice(['INFO', 'WARNING', 'ERROR', 'CRITICAL'], p=[0.4, 0.3, 0.2, 0.1])
            
            alerts.append({
                'timestamp': alert_time.isoformat(),
                'severity': severity,
                'title': f"Risk Alert {i+1}",
                'description': f"Sample alert description {i+1}"
            })
        
        return sorted(alerts, key=lambda x: x['timestamp'])

class DashboardManager:
    """Manages risk dashboards and widgets"""
    
    def __init__(self, db_connection, redis_client):
        self.db_connection = db_connection
        self.redis_client = redis_client
        self.visualization_engine = RiskVisualizationEngine(db_connection, redis_client)
        
        # Real-time updates
        self.update_threads = {}
        self.active_dashboards = set()
    
    def create_dashboard(self, dashboard_type: DashboardType, title: str, description: str,
                        user_id: str, portfolio_id: Optional[str] = None) -> Dashboard:
        """Create new dashboard"""
        try:
            dashboard_id = str(uuid.uuid4())
            
            # Get default layout and widgets for dashboard type
            layout, widgets = self._get_default_dashboard_config(dashboard_type, portfolio_id)
            
            dashboard = Dashboard(
                dashboard_id=dashboard_id,
                dashboard_type=dashboard_type,
                title=title,
                description=description,
                layout=layout,
                widgets=widgets,
                user_id=user_id,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            # Store dashboard
            self._store_dashboard(dashboard)
            
            return dashboard
            
        except Exception as e:
            logger.error(f"Error creating dashboard: {e}")
            raise
    
    def get_dashboard(self, dashboard_id: str) -> Optional[Dashboard]:
        """Get dashboard by ID"""
        try:
            with self.db_connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT dashboard_id, dashboard_type, title, description, layout,
                           user_id, is_public, created_at, updated_at
                    FROM dashboards
                    WHERE dashboard_id = %s
                """, (dashboard_id,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                # Get widgets
                widgets = self._get_dashboard_widgets(dashboard_id)
                
                return Dashboard(
                    dashboard_id=row['dashboard_id'],
                    dashboard_type=DashboardType(row['dashboard_type']),
                    title=row['title'],
                    description=row['description'],
                    layout=row['layout'],
                    widgets=widgets,
                    user_id=row['user_id'],
                    is_public=row['is_public'],
                    created_at=row['created_at'],
                    updated_at=row['updated_at']
                )
                
        except Exception as e:
            logger.error(f"Error getting dashboard: {e}")
            return None
    
    def update_dashboard_data(self, dashboard_id: str) -> Dict[str, Any]:
        """Update dashboard data"""
        try:
            dashboard = self.get_dashboard(dashboard_id)
            if not dashboard:
                return {'error': 'Dashboard not found'}
            
            updated_widgets = []
            
            for widget in dashboard.widgets:
                # Update widget data based on type
                widget_data = self._update_widget_data(widget)
                widget.data = widget_data
                widget.last_updated = datetime.now()
                updated_widgets.append(widget)
            
            # Update dashboard
            dashboard.widgets = updated_widgets
            dashboard.updated_at = datetime.now()
            
            # Store updated dashboard
            self._store_dashboard(dashboard)
            
            # Cache updated data
            self._cache_dashboard_data(dashboard)
            
            return {
                'dashboard_id': dashboard_id,
                'widgets_updated': len(updated_widgets),
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error updating dashboard data: {e}")
            return {'error': str(e)}
    
    def get_dashboard_data(self, dashboard_id: str) -> Dict[str, Any]:
        """Get dashboard data for display"""
        try:
            # Check cache first
            cached_data = self._get_cached_dashboard_data(dashboard_id)
            if cached_data:
                return cached_data
            
            # Get dashboard
            dashboard = self.get_dashboard(dashboard_id)
            if not dashboard:
                return {'error': 'Dashboard not found'}
            
            # Update data if needed
            self.update_dashboard_data(dashboard_id)
            
            # Return dashboard data
            return {
                'dashboard': dashboard.to_dict(),
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting dashboard data: {e}")
            return {'error': str(e)}
    
    def start_real_time_updates(self, dashboard_id: str):
        """Start real-time updates for dashboard"""
        if dashboard_id in self.active_dashboards:
            return
        
        self.active_dashboards.add(dashboard_id)
        
        def update_loop():
            while dashboard_id in self.active_dashboards:
                try:
                    self.update_dashboard_data(dashboard_id)
                    time.sleep(30)  # Update every 30 seconds
                except Exception as e:
                    logger.error(f"Error in real-time update loop: {e}")
                    time.sleep(60)  # Wait longer on error
        
        thread = threading.Thread(target=update_loop, daemon=True)
        thread.start()
        self.update_threads[dashboard_id] = thread
        
        logger.info(f"Started real-time updates for dashboard {dashboard_id}")
    
    def stop_real_time_updates(self, dashboard_id: str):
        """Stop real-time updates for dashboard"""
        if dashboard_id in self.active_dashboards:
            self.active_dashboards.remove(dashboard_id)
        
        if dashboard_id in self.update_threads:
            del self.update_threads[dashboard_id]
        
        logger.info(f"Stopped real-time updates for dashboard {dashboard_id}")
    
    def _get_default_dashboard_config(self, dashboard_type: DashboardType, 
                                    portfolio_id: Optional[str]) -> Tuple[Dict[str, Any], List[DashboardWidget]]:
        """Get default dashboard configuration"""
        layout = {
            'grid_size': {'width': 12, 'height': 8},
            'widget_margin': 10,
            'responsive': True
        }
        
        widgets = []
        
        if dashboard_type == DashboardType.OVERVIEW:
            widgets = [
                DashboardWidget(
                    widget_id=str(uuid.uuid4()),
                    widget_type="risk_overview",
                    title="Risk Overview",
                    chart_type=ChartType.LINE_CHART,
                    data_source="risk_metrics",
                    refresh_interval=30,
                    position={'x': 0, 'y': 0, 'width': 8, 'height': 4},
                    config={'timeframe': 'daily'},
                    filters={'portfolio_id': portfolio_id}
                ),
                DashboardWidget(
                    widget_id=str(uuid.uuid4()),
                    widget_type="var_gauge",
                    title="VaR 95%",
                    chart_type=ChartType.GAUGE,
                    data_source="current_metrics",
                    refresh_interval=10,
                    position={'x': 8, 'y': 0, 'width': 4, 'height': 2},
                    config={'metric': 'var_95'},
                    filters={'portfolio_id': portfolio_id}
                ),
                DashboardWidget(
                    widget_id=str(uuid.uuid4()),
                    widget_type="volatility_gauge",
                    title="Volatility",
                    chart_type=ChartType.GAUGE,
                    data_source="current_metrics",
                    refresh_interval=10,
                    position={'x': 8, 'y': 2, 'width': 4, 'height': 2},
                    config={'metric': 'volatility'},
                    filters={'portfolio_id': portfolio_id}
                ),
                DashboardWidget(
                    widget_id=str(uuid.uuid4()),
                    widget_type="position_heatmap",
                    title="Position Risk Heatmap",
                    chart_type=ChartType.HEATMAP,
                    data_source="position_risk",
                    refresh_interval=60,
                    position={'x': 0, 'y': 4, 'width': 12, 'height': 4},
                    config={},
                    filters={'portfolio_id': portfolio_id}
                )
            ]
        
        elif dashboard_type == DashboardType.STRESS_TEST:
            widgets = [
                DashboardWidget(
                    widget_id=str(uuid.uuid4()),
                    widget_type="stress_test_results",
                    title="Stress Test Results",
                    chart_type=ChartType.BAR_CHART,
                    data_source="stress_tests",
                    refresh_interval=300,
                    position={'x': 0, 'y': 0, 'width': 12, 'height': 6},
                    config={},
                    filters={'portfolio_id': portfolio_id}
                )
            ]
        
        elif dashboard_type == DashboardType.COMPLIANCE:
            widgets = [
                DashboardWidget(
                    widget_id=str(uuid.uuid4()),
                    widget_type="compliance_status",
                    title="Compliance Status",
                    chart_type=ChartType.BAR_CHART,
                    data_source="compliance",
                    refresh_interval=300,
                    position={'x': 0, 'y': 0, 'width': 8, 'height': 4},
                    config={},
                    filters={'portfolio_id': portfolio_id}
                ),
                DashboardWidget(
                    widget_id=str(uuid.uuid4()),
                    widget_type="alert_timeline",
                    title="Recent Alerts",
                    chart_type=ChartType.SCATTER,
                    data_source="alerts",
                    refresh_interval=60,
                    position={'x': 0, 'y': 4, 'width': 12, 'height': 4},
                    config={'days': 7},
                    filters={'portfolio_id': portfolio_id}
                )
            ]
        
        return layout, widgets
    
    def _update_widget_data(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Update widget data based on type"""
        try:
            portfolio_id = widget.filters.get('portfolio_id')
            
            if widget.widget_type == "risk_overview":
                return self.visualization_engine.create_risk_overview_chart(portfolio_id)
            
            elif widget.widget_type in ["var_gauge", "volatility_gauge"]:
                metric = widget.config.get('metric')
                return self.visualization_engine.create_risk_gauge_chart(portfolio_id, metric)
            
            elif widget.widget_type == "position_heatmap":
                return self.visualization_engine.create_position_heatmap(portfolio_id)
            
            elif widget.widget_type == "stress_test_results":
                return self.visualization_engine.create_stress_test_chart(portfolio_id)
            
            elif widget.widget_type == "compliance_status":
                return self.visualization_engine.create_compliance_status_chart(portfolio_id)
            
            elif widget.widget_type == "alert_timeline":
                days = widget.config.get('days', 7)
                return self.visualization_engine.create_alert_timeline_chart(portfolio_id, days)
            
            else:
                return {'error': f'Unknown widget type: {widget.widget_type}'}
                
        except Exception as e:
            logger.error(f"Error updating widget data: {e}")
            return {'error': str(e)}
    
    def _store_dashboard(self, dashboard: Dashboard):
        """Store dashboard in database"""
        try:
            with self.db_connection.cursor() as cursor:
                # Store dashboard
                cursor.execute("""
                    INSERT INTO dashboards (
                        dashboard_id, dashboard_type, title, description, layout,
                        user_id, is_public, created_at, updated_at
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    ON CONFLICT (dashboard_id) DO UPDATE SET
                        title = EXCLUDED.title,
                        description = EXCLUDED.description,
                        layout = EXCLUDED.layout,
                        updated_at = EXCLUDED.updated_at
                """, (
                    dashboard.dashboard_id, dashboard.dashboard_type.value,
                    dashboard.title, dashboard.description, json.dumps(dashboard.layout),
                    dashboard.user_id, dashboard.is_public,
                    dashboard.created_at, dashboard.updated_at
                ))
                
                # Store widgets
                for widget in dashboard.widgets:
                    cursor.execute("""
                        INSERT INTO dashboard_widgets (
                            widget_id, dashboard_id, widget_type, title, chart_type,
                            data_source, refresh_interval, position, config, filters,
                            data, last_updated
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                        )
                        ON CONFLICT (widget_id) DO UPDATE SET
                            title = EXCLUDED.title,
                            position = EXCLUDED.position,
                            config = EXCLUDED.config,
                            data = EXCLUDED.data,
                            last_updated = EXCLUDED.last_updated
                    """, (
                        widget.widget_id, dashboard.dashboard_id, widget.widget_type,
                        widget.title, widget.chart_type.value, widget.data_source,
                        widget.refresh_interval, json.dumps(widget.position),
                        json.dumps(widget.config), json.dumps(widget.filters),
                        json.dumps(widget.data), widget.last_updated
                    ))
                
                self.db_connection.commit()
                
        except Exception as e:
            logger.error(f"Error storing dashboard: {e}")
            self.db_connection.rollback()
            raise
    
    def _get_dashboard_widgets(self, dashboard_id: str) -> List[DashboardWidget]:
        """Get dashboard widgets"""
        try:
            with self.db_connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT widget_id, widget_type, title, chart_type, data_source,
                           refresh_interval, position, config, filters, data, last_updated
                    FROM dashboard_widgets
                    WHERE dashboard_id = %s
                    ORDER BY widget_id
                """, (dashboard_id,))
                
                widgets = []
                for row in cursor.fetchall():
                    widget = DashboardWidget(
                        widget_id=row['widget_id'],
                        widget_type=row['widget_type'],
                        title=row['title'],
                        chart_type=ChartType(row['chart_type']),
                        data_source=row['data_source'],
                        refresh_interval=row['refresh_interval'],
                        position=row['position'],
                        config=row['config'],
                        filters=row['filters'],
                        data=row['data'],
                        last_updated=row['last_updated']
                    )
                    widgets.append(widget)
                
                return widgets
                
        except Exception as e:
            logger.error(f"Error getting dashboard widgets: {e}")
            return []
    
    def _cache_dashboard_data(self, dashboard: Dashboard):
        """Cache dashboard data in Redis"""
        try:
            cache_key = f"dashboard_data:{dashboard.dashboard_id}"
            cache_data = dashboard.to_dict()
            
            self.redis_client.setex(
                cache_key,
                300,  # 5 minute expiry
                json.dumps(cache_data, default=str)
            )
            
        except Exception as e:
            logger.error(f"Error caching dashboard data: {e}")
    
    def _get_cached_dashboard_data(self, dashboard_id: str) -> Optional[Dict[str, Any]]:
        """Get cached dashboard data"""
        try:
            cache_key = f"dashboard_data:{dashboard_id}"
            cached_data = self.redis_client.get(cache_key)
            
            if cached_data:
                return json.loads(cached_data)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting cached dashboard data: {e}")
            return None

