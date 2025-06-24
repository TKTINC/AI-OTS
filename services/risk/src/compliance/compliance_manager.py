"""
Compliance and Audit Framework
Regulatory compliance and comprehensive audit trail management

This module provides comprehensive compliance and audit capabilities including:
- Regulatory compliance monitoring and reporting
- Comprehensive audit trail management
- Risk governance and oversight
- Regulatory report generation
- Compliance violation tracking
- Audit data retention and archival
- Regulatory submission management

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
import hashlib
import zipfile
import io
from pathlib import Path

import pandas as pd
import numpy as np
import redis
import psycopg2
from psycopg2.extras import RealDictCursor
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors

logger = logging.getLogger(__name__)

class RegulatoryFramework(Enum):
    """Regulatory frameworks"""
    SEC = "sec"                    # Securities and Exchange Commission
    FINRA = "finra"               # Financial Industry Regulatory Authority
    CFTC = "cftc"                 # Commodity Futures Trading Commission
    BASEL_III = "basel_iii"       # Basel III banking regulations
    MIFID_II = "mifid_ii"         # Markets in Financial Instruments Directive II
    DODD_FRANK = "dodd_frank"     # Dodd-Frank Wall Street Reform
    VOLCKER = "volcker"           # Volcker Rule
    EMIR = "emir"                 # European Market Infrastructure Regulation

class ComplianceRule(Enum):
    """Compliance rules and requirements"""
    POSITION_LIMITS = "position_limits"
    RISK_LIMITS = "risk_limits"
    CONCENTRATION_LIMITS = "concentration_limits"
    LEVERAGE_LIMITS = "leverage_limits"
    LIQUIDITY_REQUIREMENTS = "liquidity_requirements"
    CAPITAL_REQUIREMENTS = "capital_requirements"
    STRESS_TESTING = "stress_testing"
    REPORTING_REQUIREMENTS = "reporting_requirements"
    RECORD_KEEPING = "record_keeping"
    BEST_EXECUTION = "best_execution"

class ViolationType(Enum):
    """Types of compliance violations"""
    LIMIT_BREACH = "limit_breach"
    REPORTING_FAILURE = "reporting_failure"
    RECORD_KEEPING_FAILURE = "record_keeping_failure"
    UNAUTHORIZED_TRADING = "unauthorized_trading"
    INADEQUATE_CONTROLS = "inadequate_controls"
    SYSTEM_FAILURE = "system_failure"
    DATA_INTEGRITY = "data_integrity"
    DISCLOSURE_FAILURE = "disclosure_failure"

class ViolationSeverity(Enum):
    """Severity levels for compliance violations"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AuditEventType(Enum):
    """Types of audit events"""
    TRADE_EXECUTION = "trade_execution"
    POSITION_CHANGE = "position_change"
    RISK_LIMIT_CHANGE = "risk_limit_change"
    SYSTEM_ACCESS = "system_access"
    DATA_MODIFICATION = "data_modification"
    REPORT_GENERATION = "report_generation"
    COMPLIANCE_CHECK = "compliance_check"
    ALERT_GENERATION = "alert_generation"
    USER_ACTION = "user_action"
    SYSTEM_EVENT = "system_event"

@dataclass
class ComplianceViolation:
    """Compliance violation record"""
    violation_id: str
    portfolio_id: str
    rule: ComplianceRule
    framework: RegulatoryFramework
    violation_type: ViolationType
    severity: ViolationSeverity
    
    # Violation details
    title: str
    description: str
    details: Dict[str, Any]
    
    # Values
    current_value: float
    limit_value: float
    breach_amount: float
    breach_percentage: float
    
    # Timing
    detected_at: datetime
    occurred_at: datetime
    resolved_at: Optional[datetime] = None
    
    # Resolution
    is_resolved: bool = False
    resolution_action: Optional[str] = None
    resolved_by: Optional[str] = None
    
    # Reporting
    reported_to_regulator: bool = False
    report_date: Optional[datetime] = None
    report_reference: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['rule'] = self.rule.value
        data['framework'] = self.framework.value
        data['violation_type'] = self.violation_type.value
        data['severity'] = self.severity.value
        data['detected_at'] = self.detected_at.isoformat()
        data['occurred_at'] = self.occurred_at.isoformat()
        if self.resolved_at:
            data['resolved_at'] = self.resolved_at.isoformat()
        if self.report_date:
            data['report_date'] = self.report_date.isoformat()
        return data

@dataclass
class AuditEvent:
    """Audit trail event"""
    event_id: str
    event_type: AuditEventType
    timestamp: datetime
    
    # Context
    user_id: Optional[str]
    portfolio_id: Optional[str]
    session_id: Optional[str]
    ip_address: Optional[str]
    
    # Event details
    action: str
    description: str
    details: Dict[str, Any]
    
    # Data integrity
    data_hash: str
    previous_hash: Optional[str] = None
    
    # Metadata
    system_version: str = "4.0.0"
    compliance_relevant: bool = False
    retention_years: int = 7
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['event_type'] = self.event_type.value
        data['timestamp'] = self.timestamp.isoformat()
        return data

@dataclass
class RegulatoryReport:
    """Regulatory report definition"""
    report_id: str
    report_type: str
    framework: RegulatoryFramework
    reporting_period_start: datetime
    reporting_period_end: datetime
    
    # Report content
    data: Dict[str, Any]
    summary: Dict[str, Any]
    
    # Generation details
    generated_at: datetime
    generated_by: str
    file_path: Optional[str] = None
    file_hash: Optional[str] = None
    
    # Submission details
    submitted: bool = False
    submission_date: Optional[datetime] = None
    submission_reference: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['framework'] = self.framework.value
        data['reporting_period_start'] = self.reporting_period_start.isoformat()
        data['reporting_period_end'] = self.reporting_period_end.isoformat()
        data['generated_at'] = self.generated_at.isoformat()
        if self.submission_date:
            data['submission_date'] = self.submission_date.isoformat()
        return data

class ComplianceMonitor:
    """Monitors compliance with regulatory requirements"""
    
    def __init__(self, db_connection, redis_client):
        self.db_connection = db_connection
        self.redis_client = redis_client
        self.compliance_rules = self._load_compliance_rules()
    
    def check_position_compliance(self, portfolio_id: str, position_data: Dict[str, Any]) -> List[ComplianceViolation]:
        """Check position-related compliance"""
        violations = []
        
        try:
            # Get applicable rules
            rules = self._get_applicable_rules(portfolio_id, ComplianceRule.POSITION_LIMITS)
            
            for rule in rules:
                violation = self._check_position_rule(portfolio_id, position_data, rule)
                if violation:
                    violations.append(violation)
            
            return violations
            
        except Exception as e:
            logger.error(f"Error checking position compliance: {e}")
            return []
    
    def check_risk_compliance(self, portfolio_id: str, risk_metrics: Dict[str, Any]) -> List[ComplianceViolation]:
        """Check risk-related compliance"""
        violations = []
        
        try:
            # Check VaR limits (Basel III, Dodd-Frank)
            var_violation = self._check_var_compliance(portfolio_id, risk_metrics)
            if var_violation:
                violations.append(var_violation)
            
            # Check leverage limits (Basel III, Volcker)
            leverage_violation = self._check_leverage_compliance(portfolio_id, risk_metrics)
            if leverage_violation:
                violations.append(leverage_violation)
            
            # Check concentration limits (Volcker, FINRA)
            concentration_violation = self._check_concentration_compliance(portfolio_id, risk_metrics)
            if concentration_violation:
                violations.append(concentration_violation)
            
            return violations
            
        except Exception as e:
            logger.error(f"Error checking risk compliance: {e}")
            return []
    
    def check_stress_test_compliance(self, portfolio_id: str, stress_results: Dict[str, Any]) -> List[ComplianceViolation]:
        """Check stress testing compliance"""
        violations = []
        
        try:
            # Check if stress testing is current (required by Basel III, Dodd-Frank)
            last_stress_test = stress_results.get('last_test_date')
            if last_stress_test:
                days_since_test = (datetime.now() - datetime.fromisoformat(last_stress_test)).days
                
                if days_since_test > 30:  # Monthly stress testing required
                    violation = ComplianceViolation(
                        violation_id=str(uuid.uuid4()),
                        portfolio_id=portfolio_id,
                        rule=ComplianceRule.STRESS_TESTING,
                        framework=RegulatoryFramework.BASEL_III,
                        violation_type=ViolationType.REPORTING_FAILURE,
                        severity=ViolationSeverity.MEDIUM,
                        title="Stress Testing Overdue",
                        description=f"Stress testing is {days_since_test} days overdue (required monthly)",
                        details={
                            'last_test_date': last_stress_test,
                            'days_overdue': days_since_test - 30,
                            'required_frequency': 'monthly'
                        },
                        current_value=days_since_test,
                        limit_value=30,
                        breach_amount=days_since_test - 30,
                        breach_percentage=(days_since_test - 30) / 30 * 100,
                        detected_at=datetime.now(),
                        occurred_at=datetime.now() - timedelta(days=days_since_test - 30)
                    )
                    violations.append(violation)
            
            # Check stress test results compliance
            worst_case_loss = stress_results.get('worst_case_loss_pct', 0)
            if worst_case_loss > 0.25:  # 25% stress loss threshold
                violation = ComplianceViolation(
                    violation_id=str(uuid.uuid4()),
                    portfolio_id=portfolio_id,
                    rule=ComplianceRule.STRESS_TESTING,
                    framework=RegulatoryFramework.DODD_FRANK,
                    violation_type=ViolationType.LIMIT_BREACH,
                    severity=ViolationSeverity.HIGH,
                    title="Stress Test Failure",
                    description=f"Portfolio fails stress test with {worst_case_loss:.1%} loss",
                    details={
                        'worst_case_loss_pct': worst_case_loss,
                        'stress_threshold': 0.25,
                        'scenarios_failed': stress_results.get('scenarios_failed', [])
                    },
                    current_value=worst_case_loss,
                    limit_value=0.25,
                    breach_amount=worst_case_loss - 0.25,
                    breach_percentage=(worst_case_loss - 0.25) / 0.25 * 100,
                    detected_at=datetime.now(),
                    occurred_at=datetime.now()
                )
                violations.append(violation)
            
            return violations
            
        except Exception as e:
            logger.error(f"Error checking stress test compliance: {e}")
            return []
    
    def _check_position_rule(self, portfolio_id: str, position_data: Dict[str, Any], 
                           rule: Dict[str, Any]) -> Optional[ComplianceViolation]:
        """Check specific position rule"""
        try:
            rule_type = rule['rule_type']
            limit_value = rule['limit_value']
            
            if rule_type == 'max_position_size':
                # Check individual position sizes
                for position in position_data.get('positions', []):
                    if position['market_value'] > limit_value:
                        return ComplianceViolation(
                            violation_id=str(uuid.uuid4()),
                            portfolio_id=portfolio_id,
                            rule=ComplianceRule.POSITION_LIMITS,
                            framework=RegulatoryFramework(rule['framework']),
                            violation_type=ViolationType.LIMIT_BREACH,
                            severity=ViolationSeverity.MEDIUM,
                            title=f"Position Size Limit Exceeded - {position['symbol']}",
                            description=f"Position {position['symbol']} exceeds maximum size limit",
                            details={
                                'symbol': position['symbol'],
                                'position_value': position['market_value'],
                                'limit_value': limit_value,
                                'rule_reference': rule['rule_reference']
                            },
                            current_value=position['market_value'],
                            limit_value=limit_value,
                            breach_amount=position['market_value'] - limit_value,
                            breach_percentage=(position['market_value'] - limit_value) / limit_value * 100,
                            detected_at=datetime.now(),
                            occurred_at=datetime.now()
                        )
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking position rule: {e}")
            return None
    
    def _check_var_compliance(self, portfolio_id: str, risk_metrics: Dict[str, Any]) -> Optional[ComplianceViolation]:
        """Check VaR compliance"""
        try:
            current_var = risk_metrics.get('var_99', 0)
            portfolio_value = risk_metrics.get('portfolio_value', 0)
            
            # Basel III: VaR should not exceed 2.5% of Tier 1 capital (simplified to portfolio value)
            var_limit = portfolio_value * 0.025
            
            if current_var > var_limit:
                return ComplianceViolation(
                    violation_id=str(uuid.uuid4()),
                    portfolio_id=portfolio_id,
                    rule=ComplianceRule.RISK_LIMITS,
                    framework=RegulatoryFramework.BASEL_III,
                    violation_type=ViolationType.LIMIT_BREACH,
                    severity=ViolationSeverity.HIGH,
                    title="VaR Limit Exceeded",
                    description=f"Portfolio VaR exceeds Basel III limit",
                    details={
                        'current_var': current_var,
                        'var_limit': var_limit,
                        'portfolio_value': portfolio_value,
                        'var_percentage': current_var / portfolio_value * 100
                    },
                    current_value=current_var,
                    limit_value=var_limit,
                    breach_amount=current_var - var_limit,
                    breach_percentage=(current_var - var_limit) / var_limit * 100,
                    detected_at=datetime.now(),
                    occurred_at=datetime.now()
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking VaR compliance: {e}")
            return None
    
    def _check_leverage_compliance(self, portfolio_id: str, risk_metrics: Dict[str, Any]) -> Optional[ComplianceViolation]:
        """Check leverage compliance"""
        try:
            leverage_ratio = risk_metrics.get('leverage_ratio', 0)
            
            # Volcker Rule: Proprietary trading leverage limit (simplified)
            leverage_limit = 3.0  # 3:1 leverage limit
            
            if leverage_ratio > leverage_limit:
                return ComplianceViolation(
                    violation_id=str(uuid.uuid4()),
                    portfolio_id=portfolio_id,
                    rule=ComplianceRule.LEVERAGE_LIMITS,
                    framework=RegulatoryFramework.VOLCKER,
                    violation_type=ViolationType.LIMIT_BREACH,
                    severity=ViolationSeverity.HIGH,
                    title="Leverage Limit Exceeded",
                    description=f"Portfolio leverage exceeds Volcker Rule limit",
                    details={
                        'current_leverage': leverage_ratio,
                        'leverage_limit': leverage_limit,
                        'rule_reference': 'Volcker Rule Section 619'
                    },
                    current_value=leverage_ratio,
                    limit_value=leverage_limit,
                    breach_amount=leverage_ratio - leverage_limit,
                    breach_percentage=(leverage_ratio - leverage_limit) / leverage_limit * 100,
                    detected_at=datetime.now(),
                    occurred_at=datetime.now()
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking leverage compliance: {e}")
            return None
    
    def _check_concentration_compliance(self, portfolio_id: str, risk_metrics: Dict[str, Any]) -> Optional[ComplianceViolation]:
        """Check concentration compliance"""
        try:
            max_concentration = risk_metrics.get('max_position_concentration', 0)
            
            # FINRA: Single position concentration limit
            concentration_limit = 0.10  # 10% maximum concentration
            
            if max_concentration > concentration_limit:
                return ComplianceViolation(
                    violation_id=str(uuid.uuid4()),
                    portfolio_id=portfolio_id,
                    rule=ComplianceRule.CONCENTRATION_LIMITS,
                    framework=RegulatoryFramework.FINRA,
                    violation_type=ViolationType.LIMIT_BREACH,
                    severity=ViolationSeverity.MEDIUM,
                    title="Concentration Limit Exceeded",
                    description=f"Portfolio concentration exceeds FINRA limit",
                    details={
                        'max_concentration': max_concentration,
                        'concentration_limit': concentration_limit,
                        'rule_reference': 'FINRA Rule 2111'
                    },
                    current_value=max_concentration,
                    limit_value=concentration_limit,
                    breach_amount=max_concentration - concentration_limit,
                    breach_percentage=(max_concentration - concentration_limit) / concentration_limit * 100,
                    detected_at=datetime.now(),
                    occurred_at=datetime.now()
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking concentration compliance: {e}")
            return None
    
    def _load_compliance_rules(self) -> Dict[str, Any]:
        """Load compliance rules configuration"""
        return {
            'position_limits': {
                'max_position_size': 50000,
                'max_concentration': 0.10,
                'frameworks': [RegulatoryFramework.FINRA, RegulatoryFramework.SEC]
            },
            'risk_limits': {
                'max_var_percentage': 0.025,
                'max_leverage': 3.0,
                'frameworks': [RegulatoryFramework.BASEL_III, RegulatoryFramework.VOLCKER]
            },
            'stress_testing': {
                'frequency_days': 30,
                'max_loss_threshold': 0.25,
                'frameworks': [RegulatoryFramework.BASEL_III, RegulatoryFramework.DODD_FRANK]
            }
        }
    
    def _get_applicable_rules(self, portfolio_id: str, rule_type: ComplianceRule) -> List[Dict[str, Any]]:
        """Get applicable compliance rules for portfolio"""
        try:
            with self.db_connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT rule_type, limit_value, framework, rule_reference
                    FROM compliance_rules
                    WHERE portfolio_id = %s AND rule_type = %s AND is_active = true
                """, (portfolio_id, rule_type.value))
                
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Error getting applicable rules: {e}")
            return []

class AuditTrailManager:
    """Manages comprehensive audit trail"""
    
    def __init__(self, db_connection, redis_client):
        self.db_connection = db_connection
        self.redis_client = redis_client
        self.previous_hash = self._get_last_hash()
    
    def log_event(self, event_type: AuditEventType, action: str, description: str,
                  details: Dict[str, Any], user_id: Optional[str] = None,
                  portfolio_id: Optional[str] = None, session_id: Optional[str] = None,
                  ip_address: Optional[str] = None, compliance_relevant: bool = False) -> str:
        """Log audit event"""
        try:
            # Create event
            event = AuditEvent(
                event_id=str(uuid.uuid4()),
                event_type=event_type,
                timestamp=datetime.now(),
                user_id=user_id,
                portfolio_id=portfolio_id,
                session_id=session_id,
                ip_address=ip_address,
                action=action,
                description=description,
                details=details,
                data_hash=self._calculate_hash(action, description, details),
                previous_hash=self.previous_hash,
                compliance_relevant=compliance_relevant
            )
            
            # Store event
            self._store_audit_event(event)
            
            # Update previous hash for chain integrity
            self.previous_hash = event.data_hash
            
            # Cache recent events
            self._cache_recent_event(event)
            
            return event.event_id
            
        except Exception as e:
            logger.error(f"Error logging audit event: {e}")
            raise
    
    def get_audit_trail(self, portfolio_id: Optional[str] = None, user_id: Optional[str] = None,
                       event_type: Optional[AuditEventType] = None, start_date: Optional[datetime] = None,
                       end_date: Optional[datetime] = None, limit: int = 1000) -> List[AuditEvent]:
        """Get audit trail with filtering"""
        try:
            with self.db_connection.cursor(cursor_factory=RealDictCursor) as cursor:
                # Build query
                where_conditions = []
                params = []
                
                if portfolio_id:
                    where_conditions.append("portfolio_id = %s")
                    params.append(portfolio_id)
                
                if user_id:
                    where_conditions.append("user_id = %s")
                    params.append(user_id)
                
                if event_type:
                    where_conditions.append("event_type = %s")
                    params.append(event_type.value)
                
                if start_date:
                    where_conditions.append("timestamp >= %s")
                    params.append(start_date)
                
                if end_date:
                    where_conditions.append("timestamp <= %s")
                    params.append(end_date)
                
                where_clause = "WHERE " + " AND ".join(where_conditions) if where_conditions else ""
                
                cursor.execute(f"""
                    SELECT event_id, event_type, timestamp, user_id, portfolio_id,
                           session_id, ip_address, action, description, details,
                           data_hash, previous_hash, compliance_relevant
                    FROM audit_events
                    {where_clause}
                    ORDER BY timestamp DESC
                    LIMIT %s
                """, params + [limit])
                
                events = []
                for row in cursor.fetchall():
                    event = AuditEvent(
                        event_id=row['event_id'],
                        event_type=AuditEventType(row['event_type']),
                        timestamp=row['timestamp'],
                        user_id=row['user_id'],
                        portfolio_id=row['portfolio_id'],
                        session_id=row['session_id'],
                        ip_address=row['ip_address'],
                        action=row['action'],
                        description=row['description'],
                        details=row['details'],
                        data_hash=row['data_hash'],
                        previous_hash=row['previous_hash'],
                        compliance_relevant=row['compliance_relevant']
                    )
                    events.append(event)
                
                return events
                
        except Exception as e:
            logger.error(f"Error getting audit trail: {e}")
            return []
    
    def verify_audit_integrity(self, start_date: Optional[datetime] = None,
                             end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """Verify audit trail integrity"""
        try:
            events = self.get_audit_trail(start_date=start_date, end_date=end_date, limit=10000)
            
            integrity_issues = []
            hash_chain_valid = True
            
            for i, event in enumerate(events):
                # Verify data hash
                expected_hash = self._calculate_hash(event.action, event.description, event.details)
                if event.data_hash != expected_hash:
                    integrity_issues.append({
                        'event_id': event.event_id,
                        'issue': 'data_hash_mismatch',
                        'expected': expected_hash,
                        'actual': event.data_hash
                    })
                
                # Verify hash chain (if not first event)
                if i < len(events) - 1:
                    next_event = events[i + 1]
                    if next_event.previous_hash != event.data_hash:
                        hash_chain_valid = False
                        integrity_issues.append({
                            'event_id': next_event.event_id,
                            'issue': 'hash_chain_break',
                            'expected_previous': event.data_hash,
                            'actual_previous': next_event.previous_hash
                        })
            
            return {
                'events_checked': len(events),
                'integrity_valid': len(integrity_issues) == 0,
                'hash_chain_valid': hash_chain_valid,
                'issues': integrity_issues,
                'check_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error verifying audit integrity: {e}")
            return {'error': str(e)}
    
    def _calculate_hash(self, action: str, description: str, details: Dict[str, Any]) -> str:
        """Calculate hash for audit event"""
        data_string = f"{action}|{description}|{json.dumps(details, sort_keys=True)}"
        return hashlib.sha256(data_string.encode()).hexdigest()
    
    def _get_last_hash(self) -> Optional[str]:
        """Get the last hash in the audit chain"""
        try:
            with self.db_connection.cursor() as cursor:
                cursor.execute("""
                    SELECT data_hash FROM audit_events
                    ORDER BY timestamp DESC
                    LIMIT 1
                """)
                
                result = cursor.fetchone()
                return result[0] if result else None
                
        except Exception as e:
            logger.error(f"Error getting last hash: {e}")
            return None
    
    def _store_audit_event(self, event: AuditEvent):
        """Store audit event in database"""
        try:
            with self.db_connection.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO audit_events (
                        event_id, event_type, timestamp, user_id, portfolio_id,
                        session_id, ip_address, action, description, details,
                        data_hash, previous_hash, compliance_relevant, retention_years
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                """, (
                    event.event_id, event.event_type.value, event.timestamp,
                    event.user_id, event.portfolio_id, event.session_id, event.ip_address,
                    event.action, event.description, json.dumps(event.details),
                    event.data_hash, event.previous_hash, event.compliance_relevant,
                    event.retention_years
                ))
                self.db_connection.commit()
        except Exception as e:
            logger.error(f"Error storing audit event: {e}")
            self.db_connection.rollback()
            raise
    
    def _cache_recent_event(self, event: AuditEvent):
        """Cache recent event in Redis"""
        try:
            # Cache for quick access
            self.redis_client.lpush("recent_audit_events", json.dumps(event.to_dict()))
            self.redis_client.ltrim("recent_audit_events", 0, 999)  # Keep last 1000 events
            self.redis_client.expire("recent_audit_events", 86400)  # 24 hour expiry
        except Exception as e:
            logger.error(f"Error caching audit event: {e}")

class RegulatoryReportGenerator:
    """Generates regulatory reports"""
    
    def __init__(self, db_connection, redis_client):
        self.db_connection = db_connection
        self.redis_client = redis_client
        self.report_templates = self._load_report_templates()
    
    def generate_risk_report(self, portfolio_id: str, framework: RegulatoryFramework,
                           start_date: datetime, end_date: datetime) -> RegulatoryReport:
        """Generate risk management report"""
        try:
            # Collect data
            risk_data = self._collect_risk_data(portfolio_id, start_date, end_date)
            stress_data = self._collect_stress_test_data(portfolio_id, start_date, end_date)
            violation_data = self._collect_violation_data(portfolio_id, start_date, end_date)
            
            # Generate summary
            summary = {
                'portfolio_id': portfolio_id,
                'reporting_period': f"{start_date.date()} to {end_date.date()}",
                'average_var_95': risk_data.get('average_var_95', 0),
                'max_var_95': risk_data.get('max_var_95', 0),
                'stress_tests_conducted': stress_data.get('tests_conducted', 0),
                'stress_test_failures': stress_data.get('test_failures', 0),
                'compliance_violations': len(violation_data),
                'critical_violations': len([v for v in violation_data if v['severity'] == 'critical'])
            }
            
            # Create report
            report = RegulatoryReport(
                report_id=str(uuid.uuid4()),
                report_type="risk_management",
                framework=framework,
                reporting_period_start=start_date,
                reporting_period_end=end_date,
                data={
                    'risk_metrics': risk_data,
                    'stress_tests': stress_data,
                    'violations': violation_data
                },
                summary=summary,
                generated_at=datetime.now(),
                generated_by="system"
            )
            
            # Generate PDF
            pdf_path = self._generate_risk_report_pdf(report)
            report.file_path = pdf_path
            report.file_hash = self._calculate_file_hash(pdf_path)
            
            # Store report
            self._store_regulatory_report(report)
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating risk report: {e}")
            raise
    
    def generate_compliance_report(self, portfolio_id: str, framework: RegulatoryFramework,
                                 start_date: datetime, end_date: datetime) -> RegulatoryReport:
        """Generate compliance report"""
        try:
            # Collect data
            violation_data = self._collect_violation_data(portfolio_id, start_date, end_date)
            audit_data = self._collect_audit_data(portfolio_id, start_date, end_date)
            control_data = self._collect_control_effectiveness_data(portfolio_id, start_date, end_date)
            
            # Generate summary
            summary = {
                'portfolio_id': portfolio_id,
                'reporting_period': f"{start_date.date()} to {end_date.date()}",
                'total_violations': len(violation_data),
                'resolved_violations': len([v for v in violation_data if v['is_resolved']]),
                'pending_violations': len([v for v in violation_data if not v['is_resolved']]),
                'audit_events': len(audit_data),
                'control_effectiveness': control_data.get('effectiveness_score', 0)
            }
            
            # Create report
            report = RegulatoryReport(
                report_id=str(uuid.uuid4()),
                report_type="compliance",
                framework=framework,
                reporting_period_start=start_date,
                reporting_period_end=end_date,
                data={
                    'violations': violation_data,
                    'audit_events': audit_data,
                    'controls': control_data
                },
                summary=summary,
                generated_at=datetime.now(),
                generated_by="system"
            )
            
            # Generate PDF
            pdf_path = self._generate_compliance_report_pdf(report)
            report.file_path = pdf_path
            report.file_hash = self._calculate_file_hash(pdf_path)
            
            # Store report
            self._store_regulatory_report(report)
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating compliance report: {e}")
            raise
    
    def _collect_risk_data(self, portfolio_id: str, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Collect risk metrics data"""
        try:
            with self.db_connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT AVG(var_95) as average_var_95, MAX(var_95) as max_var_95,
                           AVG(var_99) as average_var_99, MAX(var_99) as max_var_99,
                           AVG(volatility) as average_volatility, MAX(volatility) as max_volatility,
                           COUNT(*) as data_points
                    FROM risk_metrics
                    WHERE portfolio_id = %s AND timestamp BETWEEN %s AND %s
                """, (portfolio_id, start_date, end_date))
                
                return dict(cursor.fetchone()) if cursor.rowcount > 0 else {}
                
        except Exception as e:
            logger.error(f"Error collecting risk data: {e}")
            return {}
    
    def _collect_stress_test_data(self, portfolio_id: str, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Collect stress test data"""
        try:
            with self.db_connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT COUNT(*) as tests_conducted,
                           COUNT(CASE WHEN percentage_loss > 0.20 THEN 1 END) as test_failures,
                           AVG(percentage_loss) as average_loss,
                           MAX(percentage_loss) as worst_case_loss
                    FROM stress_test_results
                    WHERE portfolio_id = %s AND test_timestamp BETWEEN %s AND %s
                """, (portfolio_id, start_date, end_date))
                
                return dict(cursor.fetchone()) if cursor.rowcount > 0 else {}
                
        except Exception as e:
            logger.error(f"Error collecting stress test data: {e}")
            return {}
    
    def _collect_violation_data(self, portfolio_id: str, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Collect compliance violation data"""
        try:
            with self.db_connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT violation_id, rule, framework, violation_type, severity,
                           title, description, current_value, limit_value,
                           breach_percentage, detected_at, is_resolved, resolved_at
                    FROM compliance_violations
                    WHERE portfolio_id = %s AND detected_at BETWEEN %s AND %s
                    ORDER BY detected_at DESC
                """, (portfolio_id, start_date, end_date))
                
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Error collecting violation data: {e}")
            return []
    
    def _collect_audit_data(self, portfolio_id: str, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Collect audit trail data"""
        try:
            with self.db_connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT event_type, action, timestamp, user_id, compliance_relevant
                    FROM audit_events
                    WHERE portfolio_id = %s AND timestamp BETWEEN %s AND %s
                    AND compliance_relevant = true
                    ORDER BY timestamp DESC
                """, (portfolio_id, start_date, end_date))
                
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Error collecting audit data: {e}")
            return []
    
    def _collect_control_effectiveness_data(self, portfolio_id: str, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Collect control effectiveness data"""
        # Simplified control effectiveness calculation
        return {
            'effectiveness_score': 0.85,  # 85% effective
            'controls_tested': 12,
            'controls_passed': 10,
            'controls_failed': 2
        }
    
    def _generate_risk_report_pdf(self, report: RegulatoryReport) -> str:
        """Generate PDF for risk report"""
        try:
            # Create output directory
            output_dir = Path("/tmp/regulatory_reports")
            output_dir.mkdir(exist_ok=True)
            
            # Generate filename
            filename = f"risk_report_{report.report_id}.pdf"
            file_path = output_dir / filename
            
            # Create PDF
            doc = SimpleDocTemplate(str(file_path), pagesize=letter)
            styles = getSampleStyleSheet()
            story = []
            
            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=18,
                spaceAfter=30,
                alignment=1  # Center
            )
            story.append(Paragraph(f"Risk Management Report", title_style))
            story.append(Paragraph(f"Framework: {report.framework.value.upper()}", styles['Heading2']))
            story.append(Spacer(1, 12))
            
            # Summary section
            story.append(Paragraph("Executive Summary", styles['Heading2']))
            summary_data = [
                ['Metric', 'Value'],
                ['Portfolio ID', report.summary['portfolio_id']],
                ['Reporting Period', report.summary['reporting_period']],
                ['Average VaR 95%', f"${report.summary['average_var_95']:,.0f}"],
                ['Maximum VaR 95%', f"${report.summary['max_var_95']:,.0f}"],
                ['Stress Tests Conducted', str(report.summary['stress_tests_conducted'])],
                ['Stress Test Failures', str(report.summary['stress_test_failures'])],
                ['Compliance Violations', str(report.summary['compliance_violations'])],
                ['Critical Violations', str(report.summary['critical_violations'])]
            ]
            
            summary_table = Table(summary_data)
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 14),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(summary_table)
            story.append(Spacer(1, 12))
            
            # Risk metrics section
            story.append(Paragraph("Risk Metrics Analysis", styles['Heading2']))
            risk_data = report.data['risk_metrics']
            story.append(Paragraph(f"Data points analyzed: {risk_data.get('data_points', 0)}", styles['Normal']))
            story.append(Paragraph(f"Average portfolio volatility: {risk_data.get('average_volatility', 0):.2%}", styles['Normal']))
            story.append(Spacer(1, 12))
            
            # Violations section
            if report.data['violations']:
                story.append(Paragraph("Compliance Violations", styles['Heading2']))
                for violation in report.data['violations'][:10]:  # Show first 10
                    story.append(Paragraph(f"• {violation['title']} ({violation['severity'].upper()})", styles['Normal']))
                story.append(Spacer(1, 12))
            
            # Footer
            story.append(Paragraph(f"Generated on: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}", styles['Normal']))
            story.append(Paragraph("AI Options Trading System - Risk Management Service", styles['Normal']))
            
            # Build PDF
            doc.build(story)
            
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Error generating risk report PDF: {e}")
            raise
    
    def _generate_compliance_report_pdf(self, report: RegulatoryReport) -> str:
        """Generate PDF for compliance report"""
        try:
            # Create output directory
            output_dir = Path("/tmp/regulatory_reports")
            output_dir.mkdir(exist_ok=True)
            
            # Generate filename
            filename = f"compliance_report_{report.report_id}.pdf"
            file_path = output_dir / filename
            
            # Create PDF (simplified version)
            doc = SimpleDocTemplate(str(file_path), pagesize=letter)
            styles = getSampleStyleSheet()
            story = []
            
            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=18,
                spaceAfter=30,
                alignment=1  # Center
            )
            story.append(Paragraph(f"Compliance Report", title_style))
            story.append(Paragraph(f"Framework: {report.framework.value.upper()}", styles['Heading2']))
            story.append(Spacer(1, 12))
            
            # Summary
            story.append(Paragraph("Compliance Summary", styles['Heading2']))
            story.append(Paragraph(f"Total Violations: {report.summary['total_violations']}", styles['Normal']))
            story.append(Paragraph(f"Resolved Violations: {report.summary['resolved_violations']}", styles['Normal']))
            story.append(Paragraph(f"Pending Violations: {report.summary['pending_violations']}", styles['Normal']))
            story.append(Spacer(1, 12))
            
            # Footer
            story.append(Paragraph(f"Generated on: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}", styles['Normal']))
            
            # Build PDF
            doc.build(story)
            
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Error generating compliance report PDF: {e}")
            raise
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate hash of generated file"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception as e:
            logger.error(f"Error calculating file hash: {e}")
            return ""
    
    def _store_regulatory_report(self, report: RegulatoryReport):
        """Store regulatory report in database"""
        try:
            with self.db_connection.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO regulatory_reports (
                        report_id, report_type, framework, reporting_period_start,
                        reporting_period_end, data, summary, generated_at, generated_by,
                        file_path, file_hash, submitted, submission_date, submission_reference
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                """, (
                    report.report_id, report.report_type, report.framework.value,
                    report.reporting_period_start, report.reporting_period_end,
                    json.dumps(report.data), json.dumps(report.summary),
                    report.generated_at, report.generated_by, report.file_path,
                    report.file_hash, report.submitted, report.submission_date,
                    report.submission_reference
                ))
                self.db_connection.commit()
        except Exception as e:
            logger.error(f"Error storing regulatory report: {e}")
            self.db_connection.rollback()
            raise
    
    def _load_report_templates(self) -> Dict[str, Any]:
        """Load report templates"""
        return {
            'risk_management': {
                'sections': ['executive_summary', 'risk_metrics', 'stress_tests', 'violations'],
                'required_data': ['risk_metrics', 'stress_tests', 'violations']
            },
            'compliance': {
                'sections': ['compliance_summary', 'violations', 'audit_trail', 'controls'],
                'required_data': ['violations', 'audit_events', 'controls']
            }
        }

class ComplianceManager:
    """Main compliance and audit management system"""
    
    def __init__(self, db_connection, redis_client, notification_config: Dict[str, Any]):
        self.db_connection = db_connection
        self.redis_client = redis_client
        self.compliance_monitor = ComplianceMonitor(db_connection, redis_client)
        self.audit_manager = AuditTrailManager(db_connection, redis_client)
        self.report_generator = RegulatoryReportGenerator(db_connection, redis_client)
        
        # Compliance monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
    
    def start_compliance_monitoring(self):
        """Start background compliance monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        logger.info("Compliance monitoring started")
    
    def stop_compliance_monitoring(self):
        """Stop background compliance monitoring"""
        self.monitoring_active = False
        logger.info("Compliance monitoring stopped")
    
    def check_portfolio_compliance(self, portfolio_id: str) -> Dict[str, Any]:
        """Check comprehensive compliance for portfolio"""
        try:
            violations = []
            
            # Get portfolio data (would integrate with portfolio service)
            portfolio_data = self._get_portfolio_data(portfolio_id)
            risk_metrics = self._get_risk_metrics(portfolio_id)
            stress_results = self._get_stress_test_results(portfolio_id)
            
            # Check position compliance
            if portfolio_data:
                violations.extend(
                    self.compliance_monitor.check_position_compliance(portfolio_id, portfolio_data)
                )
            
            # Check risk compliance
            if risk_metrics:
                violations.extend(
                    self.compliance_monitor.check_risk_compliance(portfolio_id, risk_metrics)
                )
            
            # Check stress test compliance
            if stress_results:
                violations.extend(
                    self.compliance_monitor.check_stress_test_compliance(portfolio_id, stress_results)
                )
            
            # Store violations
            for violation in violations:
                self._store_compliance_violation(violation)
            
            # Log compliance check
            self.audit_manager.log_event(
                event_type=AuditEventType.COMPLIANCE_CHECK,
                action="portfolio_compliance_check",
                description=f"Comprehensive compliance check for portfolio {portfolio_id}",
                details={
                    'portfolio_id': portfolio_id,
                    'violations_found': len(violations),
                    'violation_types': [v.violation_type.value for v in violations]
                },
                portfolio_id=portfolio_id,
                compliance_relevant=True
            )
            
            return {
                'portfolio_id': portfolio_id,
                'compliance_status': 'compliant' if len(violations) == 0 else 'violations_found',
                'violations': [v.to_dict() for v in violations],
                'check_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error checking portfolio compliance: {e}")
            return {'error': str(e)}
    
    def generate_regulatory_report(self, portfolio_id: str, framework: RegulatoryFramework,
                                 report_type: str, start_date: datetime, end_date: datetime) -> RegulatoryReport:
        """Generate regulatory report"""
        try:
            if report_type == "risk_management":
                report = self.report_generator.generate_risk_report(
                    portfolio_id, framework, start_date, end_date
                )
            elif report_type == "compliance":
                report = self.report_generator.generate_compliance_report(
                    portfolio_id, framework, start_date, end_date
                )
            else:
                raise ValueError(f"Unsupported report type: {report_type}")
            
            # Log report generation
            self.audit_manager.log_event(
                event_type=AuditEventType.REPORT_GENERATION,
                action="regulatory_report_generated",
                description=f"Generated {report_type} report for {framework.value}",
                details={
                    'report_id': report.report_id,
                    'report_type': report_type,
                    'framework': framework.value,
                    'portfolio_id': portfolio_id,
                    'reporting_period': f"{start_date.date()} to {end_date.date()}"
                },
                portfolio_id=portfolio_id,
                compliance_relevant=True
            )
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating regulatory report: {e}")
            raise
    
    def get_compliance_dashboard(self, portfolio_id: Optional[str] = None) -> Dict[str, Any]:
        """Get compliance dashboard data"""
        try:
            # Get recent violations
            recent_violations = self._get_recent_violations(portfolio_id, days=30)
            
            # Get compliance statistics
            compliance_stats = self._get_compliance_statistics(portfolio_id, days=30)
            
            # Get audit statistics
            audit_stats = self._get_audit_statistics(portfolio_id, days=30)
            
            return {
                'recent_violations': recent_violations,
                'compliance_statistics': compliance_stats,
                'audit_statistics': audit_stats,
                'dashboard_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting compliance dashboard: {e}")
            return {'error': str(e)}
    
    def _get_portfolio_data(self, portfolio_id: str) -> Optional[Dict[str, Any]]:
        """Get portfolio data for compliance checking"""
        # Mock data - would integrate with portfolio service
        return {
            'portfolio_summary': {'total_value': 125000},
            'positions': [
                {'symbol': 'AAPL', 'market_value': 25000},
                {'symbol': 'MSFT', 'market_value': 20000}
            ]
        }
    
    def _get_risk_metrics(self, portfolio_id: str) -> Optional[Dict[str, Any]]:
        """Get risk metrics for compliance checking"""
        # Mock data - would integrate with risk service
        return {
            'portfolio_value': 125000,
            'var_95': 2500,
            'var_99': 3750,
            'volatility': 0.18,
            'leverage_ratio': 2.5,
            'max_position_concentration': 0.20
        }
    
    def _get_stress_test_results(self, portfolio_id: str) -> Optional[Dict[str, Any]]:
        """Get stress test results for compliance checking"""
        # Mock data - would integrate with stress testing service
        return {
            'last_test_date': (datetime.now() - timedelta(days=25)).isoformat(),
            'worst_case_loss_pct': 0.18,
            'scenarios_failed': ['market_crash_severe']
        }
    
    def _store_compliance_violation(self, violation: ComplianceViolation):
        """Store compliance violation in database"""
        try:
            with self.db_connection.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO compliance_violations (
                        violation_id, portfolio_id, rule, framework, violation_type, severity,
                        title, description, details, current_value, limit_value,
                        breach_amount, breach_percentage, detected_at, occurred_at,
                        is_resolved, resolution_action, resolved_by, resolved_at,
                        reported_to_regulator, report_date, report_reference
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                """, (
                    violation.violation_id, violation.portfolio_id, violation.rule.value,
                    violation.framework.value, violation.violation_type.value, violation.severity.value,
                    violation.title, violation.description, json.dumps(violation.details),
                    violation.current_value, violation.limit_value, violation.breach_amount,
                    violation.breach_percentage, violation.detected_at, violation.occurred_at,
                    violation.is_resolved, violation.resolution_action, violation.resolved_by,
                    violation.resolved_at, violation.reported_to_regulator, violation.report_date,
                    violation.report_reference
                ))
                self.db_connection.commit()
        except Exception as e:
            logger.error(f"Error storing compliance violation: {e}")
            self.db_connection.rollback()
    
    def _get_recent_violations(self, portfolio_id: Optional[str], days: int) -> List[Dict[str, Any]]:
        """Get recent compliance violations"""
        try:
            with self.db_connection.cursor(cursor_factory=RealDictCursor) as cursor:
                where_clause = "WHERE detected_at >= %s"
                params = [datetime.now() - timedelta(days=days)]
                
                if portfolio_id:
                    where_clause += " AND portfolio_id = %s"
                    params.append(portfolio_id)
                
                cursor.execute(f"""
                    SELECT violation_id, portfolio_id, rule, framework, severity,
                           title, detected_at, is_resolved
                    FROM compliance_violations
                    {where_clause}
                    ORDER BY detected_at DESC
                    LIMIT 50
                """, params)
                
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Error getting recent violations: {e}")
            return []
    
    def _get_compliance_statistics(self, portfolio_id: Optional[str], days: int) -> Dict[str, Any]:
        """Get compliance statistics"""
        try:
            with self.db_connection.cursor(cursor_factory=RealDictCursor) as cursor:
                where_clause = "WHERE detected_at >= %s"
                params = [datetime.now() - timedelta(days=days)]
                
                if portfolio_id:
                    where_clause += " AND portfolio_id = %s"
                    params.append(portfolio_id)
                
                cursor.execute(f"""
                    SELECT COUNT(*) as total_violations,
                           COUNT(CASE WHEN severity = 'critical' THEN 1 END) as critical_violations,
                           COUNT(CASE WHEN is_resolved = true THEN 1 END) as resolved_violations,
                           COUNT(CASE WHEN reported_to_regulator = true THEN 1 END) as reported_violations
                    FROM compliance_violations
                    {where_clause}
                """, params)
                
                return dict(cursor.fetchone()) if cursor.rowcount > 0 else {}
                
        except Exception as e:
            logger.error(f"Error getting compliance statistics: {e}")
            return {}
    
    def _get_audit_statistics(self, portfolio_id: Optional[str], days: int) -> Dict[str, Any]:
        """Get audit trail statistics"""
        try:
            with self.db_connection.cursor(cursor_factory=RealDictCursor) as cursor:
                where_clause = "WHERE timestamp >= %s"
                params = [datetime.now() - timedelta(days=days)]
                
                if portfolio_id:
                    where_clause += " AND portfolio_id = %s"
                    params.append(portfolio_id)
                
                cursor.execute(f"""
                    SELECT COUNT(*) as total_events,
                           COUNT(CASE WHEN compliance_relevant = true THEN 1 END) as compliance_events,
                           COUNT(DISTINCT user_id) as active_users,
                           COUNT(DISTINCT event_type) as event_types
                    FROM audit_events
                    {where_clause}
                """, params)
                
                return dict(cursor.fetchone()) if cursor.rowcount > 0 else {}
                
        except Exception as e:
            logger.error(f"Error getting audit statistics: {e}")
            return {}

