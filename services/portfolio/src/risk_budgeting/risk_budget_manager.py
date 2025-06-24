"""
Risk Budgeting System
Advanced risk budgeting with dynamic allocation and portfolio-level controls
"""

import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import math

logger = logging.getLogger(__name__)

class RiskBudgetType(Enum):
    """Risk budget allocation types"""
    EQUAL_RISK = "equal_risk"
    VOLATILITY_WEIGHTED = "volatility_weighted"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    STRATEGY_WEIGHTED = "strategy_weighted"
    SECTOR_WEIGHTED = "sector_weighted"
    CUSTOM = "custom"

class RiskMetric(Enum):
    """Risk metrics for budgeting"""
    VOLATILITY = "volatility"
    VAR_95 = "var_95"
    VAR_99 = "var_99"
    EXPECTED_SHORTFALL = "expected_shortfall"
    MAX_DRAWDOWN = "max_drawdown"
    BETA_ADJUSTED_EXPOSURE = "beta_adjusted_exposure"

class AllocationStatus(Enum):
    """Risk allocation status"""
    WITHIN_BUDGET = "within_budget"
    APPROACHING_LIMIT = "approaching_limit"
    OVER_BUDGET = "over_budget"
    CRITICAL = "critical"

@dataclass
class RiskBudget:
    """Risk budget definition"""
    budget_id: str
    name: str
    risk_metric: RiskMetric
    total_budget: float  # Total risk budget (e.g., 2% VaR)
    
    # Allocation rules
    allocation_type: RiskBudgetType
    allocations: Dict[str, float]  # entity -> allocation %
    
    # Limits and controls
    max_single_allocation: float = 0.30  # 30% max to single entity
    min_allocation: float = 0.01  # 1% minimum allocation
    rebalance_threshold: float = 0.05  # 5% threshold for rebalancing
    
    # Dynamic adjustment
    dynamic_adjustment: bool = True
    confidence_scaling: bool = True
    market_regime_adjustment: bool = True
    
    # Monitoring
    warning_threshold: float = 0.80  # 80% of budget triggers warning
    critical_threshold: float = 0.95  # 95% of budget triggers critical alert
    
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
        if self.updated_at is None:
            self.updated_at = datetime.now(timezone.utc)

@dataclass
class RiskAllocation:
    """Current risk allocation"""
    entity_id: str  # Strategy, sector, symbol, etc.
    entity_type: str  # "strategy", "sector", "symbol"
    entity_name: str
    
    # Budget allocation
    allocated_budget: float  # Allocated risk budget
    allocated_percent: float  # % of total budget
    
    # Current usage
    current_risk: float  # Current risk usage
    current_percent: float  # % of total budget used
    utilization: float  # current_risk / allocated_budget
    
    # Status
    status: AllocationStatus
    days_over_budget: int = 0
    max_risk_today: float = 0.0
    
    # Positions contributing to risk
    contributing_positions: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.contributing_positions is None:
            self.contributing_positions = []

@dataclass
class RiskBudgetReport:
    """Risk budget monitoring report"""
    budget_id: str
    report_date: datetime
    
    # Overall budget status
    total_budget: float
    total_risk_used: float
    budget_utilization: float
    status: AllocationStatus
    
    # Allocations
    allocations: List[RiskAllocation]
    
    # Violations and warnings
    violations: List[str]
    warnings: List[str]
    
    # Recommendations
    recommendations: List[str]
    
    # Historical tracking
    utilization_trend: List[float]  # Last 30 days
    max_utilization_30d: float
    avg_utilization_30d: float

class RiskBudgetManager:
    """
    Advanced risk budgeting system
    Manages dynamic risk allocation across strategies, sectors, and positions
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        
        # Risk parameters
        self.default_var_confidence = self.config.get('default_var_confidence', 0.95)
        self.lookback_period = self.config.get('lookback_period', 252)
        self.rebalance_frequency = self.config.get('rebalance_frequency', 'daily')
        
        # Dynamic adjustment parameters
        self.confidence_multiplier = self.config.get('confidence_multiplier', 1.5)
        self.regime_adjustments = self.config.get('regime_adjustments', {
            'bull': 1.2,
            'bear': 0.8,
            'volatile': 0.7,
            'stable': 1.1
        })
        
        # Storage
        self.risk_budgets: Dict[str, RiskBudget] = {}
        self.allocation_history: List[RiskBudgetReport] = []
        
        logger.info("Risk Budget Manager initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for risk budgeting"""
        return {
            'default_var_confidence': 0.95,
            'lookback_period': 252,
            'rebalance_frequency': 'daily',
            'confidence_multiplier': 1.5,
            'regime_adjustments': {
                'bull': 1.2,
                'bear': 0.8,
                'volatile': 0.7,
                'stable': 1.1
            },
            'max_budget_utilization': 0.95,
            'warning_threshold': 0.80,
            'critical_threshold': 0.95
        }
    
    def create_risk_budget(self, budget_id: str, name: str, total_budget: float,
                          risk_metric: RiskMetric = RiskMetric.VAR_95,
                          allocation_type: RiskBudgetType = RiskBudgetType.EQUAL_RISK,
                          entities: List[str] = None, **kwargs) -> RiskBudget:
        """
        Create a new risk budget
        
        Args:
            budget_id: Unique budget identifier
            name: Human-readable budget name
            total_budget: Total risk budget amount
            risk_metric: Risk metric to use
            allocation_type: How to allocate risk
            entities: List of entities to allocate to
            **kwargs: Additional budget parameters
            
        Returns:
            Created risk budget
        """
        try:
            # Calculate initial allocations
            if entities:
                allocations = self._calculate_initial_allocations(entities, allocation_type)
            else:
                allocations = {}
            
            # Create budget
            budget = RiskBudget(
                budget_id=budget_id,
                name=name,
                risk_metric=risk_metric,
                total_budget=total_budget,
                allocation_type=allocation_type,
                allocations=allocations,
                max_single_allocation=kwargs.get('max_single_allocation', 0.30),
                min_allocation=kwargs.get('min_allocation', 0.01),
                rebalance_threshold=kwargs.get('rebalance_threshold', 0.05),
                dynamic_adjustment=kwargs.get('dynamic_adjustment', True),
                confidence_scaling=kwargs.get('confidence_scaling', True),
                market_regime_adjustment=kwargs.get('market_regime_adjustment', True),
                warning_threshold=kwargs.get('warning_threshold', 0.80),
                critical_threshold=kwargs.get('critical_threshold', 0.95)
            )
            
            # Store budget
            self.risk_budgets[budget_id] = budget
            
            logger.info(f"Created risk budget {budget_id}: {name}")
            return budget
            
        except Exception as e:
            logger.error(f"Error creating risk budget: {e}")
            raise
    
    def _calculate_initial_allocations(self, entities: List[str], 
                                     allocation_type: RiskBudgetType) -> Dict[str, float]:
        """Calculate initial risk allocations"""
        try:
            n_entities = len(entities)
            allocations = {}
            
            if allocation_type == RiskBudgetType.EQUAL_RISK:
                # Equal allocation to all entities
                allocation = 1.0 / n_entities
                for entity in entities:
                    allocations[entity] = allocation
            
            elif allocation_type == RiskBudgetType.VOLATILITY_WEIGHTED:
                # Inverse volatility weighting (lower vol gets more allocation)
                volatilities = self._get_entity_volatilities(entities)
                inv_vol = {entity: 1.0 / max(vol, 0.01) for entity, vol in volatilities.items()}
                total_inv_vol = sum(inv_vol.values())
                
                for entity in entities:
                    allocations[entity] = inv_vol[entity] / total_inv_vol
            
            elif allocation_type == RiskBudgetType.CONFIDENCE_WEIGHTED:
                # Confidence-based weighting
                confidences = self._get_entity_confidences(entities)
                total_confidence = sum(confidences.values())
                
                if total_confidence > 0:
                    for entity in entities:
                        allocations[entity] = confidences[entity] / total_confidence
                else:
                    # Fallback to equal allocation
                    allocation = 1.0 / n_entities
                    for entity in entities:
                        allocations[entity] = allocation
            
            else:
                # Default to equal allocation
                allocation = 1.0 / n_entities
                for entity in entities:
                    allocations[entity] = allocation
            
            return allocations
            
        except Exception as e:
            logger.error(f"Error calculating initial allocations: {e}")
            # Fallback to equal allocation
            allocation = 1.0 / len(entities)
            return {entity: allocation for entity in entities}
    
    def update_risk_allocations(self, budget_id: str, portfolio_data: Dict[str, Any],
                               market_regime: str = "normal") -> RiskBudgetReport:
        """
        Update risk allocations based on current portfolio
        
        Args:
            budget_id: Risk budget to update
            portfolio_data: Current portfolio positions and metrics
            market_regime: Current market regime
            
        Returns:
            Risk budget report
        """
        try:
            if budget_id not in self.risk_budgets:
                raise ValueError(f"Risk budget {budget_id} not found")
            
            budget = self.risk_budgets[budget_id]
            
            # Calculate current risk usage by entity
            current_allocations = self._calculate_current_risk_usage(budget, portfolio_data)
            
            # Apply dynamic adjustments
            if budget.dynamic_adjustment:
                adjusted_allocations = self._apply_dynamic_adjustments(
                    budget, current_allocations, market_regime, portfolio_data
                )
            else:
                adjusted_allocations = current_allocations
            
            # Check for violations and warnings
            violations, warnings = self._check_budget_violations(budget, adjusted_allocations)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(budget, adjusted_allocations, violations)
            
            # Calculate overall status
            total_risk_used = sum(alloc.current_risk for alloc in adjusted_allocations)
            budget_utilization = total_risk_used / budget.total_budget if budget.total_budget > 0 else 0
            
            if budget_utilization >= budget.critical_threshold:
                status = AllocationStatus.CRITICAL
            elif budget_utilization >= budget.warning_threshold:
                status = AllocationStatus.APPROACHING_LIMIT
            elif any(alloc.status == AllocationStatus.OVER_BUDGET for alloc in adjusted_allocations):
                status = AllocationStatus.OVER_BUDGET
            else:
                status = AllocationStatus.WITHIN_BUDGET
            
            # Create report
            report = RiskBudgetReport(
                budget_id=budget_id,
                report_date=datetime.now(timezone.utc),
                total_budget=budget.total_budget,
                total_risk_used=total_risk_used,
                budget_utilization=budget_utilization,
                status=status,
                allocations=adjusted_allocations,
                violations=violations,
                warnings=warnings,
                recommendations=recommendations,
                utilization_trend=self._get_utilization_trend(budget_id),
                max_utilization_30d=self._get_max_utilization_30d(budget_id),
                avg_utilization_30d=self._get_avg_utilization_30d(budget_id)
            )
            
            # Store report
            self.allocation_history.append(report)
            
            # Update budget timestamp
            budget.updated_at = datetime.now(timezone.utc)
            
            return report
            
        except Exception as e:
            logger.error(f"Error updating risk allocations: {e}")
            raise
    
    def _calculate_current_risk_usage(self, budget: RiskBudget, 
                                    portfolio_data: Dict[str, Any]) -> List[RiskAllocation]:
        """Calculate current risk usage by entity"""
        try:
            allocations = []
            positions = portfolio_data.get('positions', [])
            
            # Group positions by entity type
            entity_groups = self._group_positions_by_entity(positions, budget)
            
            for entity_id, entity_data in entity_groups.items():
                # Calculate risk for this entity
                entity_risk = self._calculate_entity_risk(entity_data, budget.risk_metric)
                
                # Get allocated budget
                allocated_percent = budget.allocations.get(entity_id, 0.0)
                allocated_budget = allocated_percent * budget.total_budget
                
                # Calculate utilization
                utilization = entity_risk / allocated_budget if allocated_budget > 0 else 0
                
                # Determine status
                if utilization >= 1.0:
                    status = AllocationStatus.OVER_BUDGET
                elif utilization >= 0.9:
                    status = AllocationStatus.APPROACHING_LIMIT
                else:
                    status = AllocationStatus.WITHIN_BUDGET
                
                allocation = RiskAllocation(
                    entity_id=entity_id,
                    entity_type=entity_data.get('type', 'unknown'),
                    entity_name=entity_data.get('name', entity_id),
                    allocated_budget=allocated_budget,
                    allocated_percent=allocated_percent,
                    current_risk=entity_risk,
                    current_percent=entity_risk / budget.total_budget if budget.total_budget > 0 else 0,
                    utilization=utilization,
                    status=status,
                    contributing_positions=entity_data.get('positions', [])
                )
                
                allocations.append(allocation)
            
            return allocations
            
        except Exception as e:
            logger.error(f"Error calculating current risk usage: {e}")
            return []
    
    def _group_positions_by_entity(self, positions: List[Dict], budget: RiskBudget) -> Dict[str, Dict]:
        """Group positions by entity (strategy, sector, symbol)"""
        try:
            entity_groups = {}
            
            for position in positions:
                # Determine entity based on budget allocation type
                if 'strategy' in budget.allocations:
                    entity_id = position.get('strategy_id', 'unknown')
                    entity_type = 'strategy'
                elif 'sector' in budget.allocations:
                    entity_id = position.get('sector', 'unknown')
                    entity_type = 'sector'
                else:
                    entity_id = position.get('symbol', 'unknown')
                    entity_type = 'symbol'
                
                if entity_id not in entity_groups:
                    entity_groups[entity_id] = {
                        'type': entity_type,
                        'name': entity_id,
                        'positions': [],
                        'total_value': 0.0,
                        'total_risk': 0.0
                    }
                
                entity_groups[entity_id]['positions'].append(position)
                entity_groups[entity_id]['total_value'] += position.get('market_value', 0.0)
            
            return entity_groups
            
        except Exception as e:
            logger.error(f"Error grouping positions: {e}")
            return {}
    
    def _calculate_entity_risk(self, entity_data: Dict, risk_metric: RiskMetric) -> float:
        """Calculate risk for an entity group"""
        try:
            positions = entity_data.get('positions', [])
            
            if risk_metric == RiskMetric.VOLATILITY:
                # Portfolio volatility calculation
                return self._calculate_portfolio_volatility(positions)
            
            elif risk_metric in [RiskMetric.VAR_95, RiskMetric.VAR_99]:
                # Value at Risk calculation
                confidence = 0.95 if risk_metric == RiskMetric.VAR_95 else 0.99
                return self._calculate_portfolio_var(positions, confidence)
            
            elif risk_metric == RiskMetric.EXPECTED_SHORTFALL:
                # Expected Shortfall (Conditional VaR)
                return self._calculate_expected_shortfall(positions)
            
            elif risk_metric == RiskMetric.BETA_ADJUSTED_EXPOSURE:
                # Beta-adjusted exposure
                return self._calculate_beta_adjusted_exposure(positions)
            
            else:
                # Default to simple volatility
                return self._calculate_portfolio_volatility(positions)
                
        except Exception as e:
            logger.error(f"Error calculating entity risk: {e}")
            return 0.0
    
    def _calculate_portfolio_volatility(self, positions: List[Dict]) -> float:
        """Calculate portfolio volatility for positions"""
        try:
            if not positions:
                return 0.0
            
            # Simple volatility calculation (weighted average)
            total_value = sum(pos.get('market_value', 0.0) for pos in positions)
            if total_value == 0:
                return 0.0
            
            weighted_vol = 0.0
            for position in positions:
                weight = position.get('market_value', 0.0) / total_value
                volatility = position.get('volatility', 0.2)  # Default 20% vol
                weighted_vol += weight * volatility
            
            return weighted_vol * total_value
            
        except Exception as e:
            logger.error(f"Error calculating portfolio volatility: {e}")
            return 0.0
    
    def _calculate_portfolio_var(self, positions: List[Dict], confidence: float) -> float:
        """Calculate portfolio Value at Risk"""
        try:
            if not positions:
                return 0.0
            
            # Simplified VaR calculation
            portfolio_vol = self._calculate_portfolio_volatility(positions)
            total_value = sum(pos.get('market_value', 0.0) for pos in positions)
            
            # Assuming normal distribution
            from scipy.stats import norm
            z_score = norm.ppf(1 - confidence)
            var = -z_score * portfolio_vol
            
            return var
            
        except Exception as e:
            logger.error(f"Error calculating portfolio VaR: {e}")
            return 0.0
    
    def _calculate_expected_shortfall(self, positions: List[Dict]) -> float:
        """Calculate Expected Shortfall (Conditional VaR)"""
        try:
            # ES is approximately 1.28 * VaR for normal distribution at 95% confidence
            var_95 = self._calculate_portfolio_var(positions, 0.95)
            return var_95 * 1.28
            
        except Exception as e:
            logger.error(f"Error calculating expected shortfall: {e}")
            return 0.0
    
    def _calculate_beta_adjusted_exposure(self, positions: List[Dict]) -> float:
        """Calculate beta-adjusted exposure"""
        try:
            if not positions:
                return 0.0
            
            beta_adjusted_value = 0.0
            for position in positions:
                market_value = position.get('market_value', 0.0)
                beta = position.get('beta', 1.0)
                beta_adjusted_value += market_value * abs(beta)
            
            return beta_adjusted_value
            
        except Exception as e:
            logger.error(f"Error calculating beta-adjusted exposure: {e}")
            return 0.0
    
    def _apply_dynamic_adjustments(self, budget: RiskBudget, allocations: List[RiskAllocation],
                                 market_regime: str, portfolio_data: Dict[str, Any]) -> List[RiskAllocation]:
        """Apply dynamic adjustments to risk allocations"""
        try:
            adjusted_allocations = []
            
            # Get regime adjustment factor
            regime_factor = self.regime_adjustments.get(market_regime, 1.0)
            
            for allocation in allocations:
                adjusted_allocation = allocation
                
                # Market regime adjustment
                if budget.market_regime_adjustment:
                    adjusted_allocation.allocated_budget *= regime_factor
                    adjusted_allocation.allocated_percent *= regime_factor
                
                # Confidence scaling
                if budget.confidence_scaling:
                    avg_confidence = self._get_entity_average_confidence(allocation.entity_id, portfolio_data)
                    confidence_factor = 0.5 + (avg_confidence * self.confidence_multiplier)
                    confidence_factor = max(0.3, min(confidence_factor, 2.0))  # Cap between 0.3x and 2x
                    
                    adjusted_allocation.allocated_budget *= confidence_factor
                    adjusted_allocation.allocated_percent *= confidence_factor
                
                # Recalculate utilization
                adjusted_allocation.utilization = (
                    adjusted_allocation.current_risk / adjusted_allocation.allocated_budget 
                    if adjusted_allocation.allocated_budget > 0 else 0
                )
                
                # Update status
                if adjusted_allocation.utilization >= 1.0:
                    adjusted_allocation.status = AllocationStatus.OVER_BUDGET
                elif adjusted_allocation.utilization >= 0.9:
                    adjusted_allocation.status = AllocationStatus.APPROACHING_LIMIT
                else:
                    adjusted_allocation.status = AllocationStatus.WITHIN_BUDGET
                
                adjusted_allocations.append(adjusted_allocation)
            
            return adjusted_allocations
            
        except Exception as e:
            logger.error(f"Error applying dynamic adjustments: {e}")
            return allocations
    
    def _check_budget_violations(self, budget: RiskBudget, 
                               allocations: List[RiskAllocation]) -> Tuple[List[str], List[str]]:
        """Check for budget violations and warnings"""
        violations = []
        warnings = []
        
        try:
            total_risk = sum(alloc.current_risk for alloc in allocations)
            total_utilization = total_risk / budget.total_budget if budget.total_budget > 0 else 0
            
            # Overall budget violations
            if total_utilization > 1.0:
                violations.append(f"Total risk budget exceeded: {total_utilization:.1%} of {budget.total_budget}")
            elif total_utilization > budget.warning_threshold:
                warnings.append(f"Approaching total risk budget limit: {total_utilization:.1%}")
            
            # Individual allocation violations
            for allocation in allocations:
                if allocation.status == AllocationStatus.OVER_BUDGET:
                    violations.append(
                        f"{allocation.entity_name} over budget: {allocation.utilization:.1%} utilization"
                    )
                elif allocation.status == AllocationStatus.APPROACHING_LIMIT:
                    warnings.append(
                        f"{allocation.entity_name} approaching limit: {allocation.utilization:.1%} utilization"
                    )
            
            return violations, warnings
            
        except Exception as e:
            logger.error(f"Error checking budget violations: {e}")
            return [], []
    
    def _generate_recommendations(self, budget: RiskBudget, allocations: List[RiskAllocation],
                                violations: List[str]) -> List[str]:
        """Generate risk management recommendations"""
        recommendations = []
        
        try:
            if violations:
                recommendations.append("Immediate action required due to budget violations")
            
            # Check for rebalancing opportunities
            over_budget = [alloc for alloc in allocations if alloc.status == AllocationStatus.OVER_BUDGET]
            under_utilized = [alloc for alloc in allocations if alloc.utilization < 0.5]
            
            if over_budget and under_utilized:
                recommendations.append("Consider rebalancing: reduce over-budget allocations and increase under-utilized ones")
            
            # Check concentration
            max_allocation = max(alloc.current_percent for alloc in allocations) if allocations else 0
            if max_allocation > budget.max_single_allocation:
                recommendations.append(f"High concentration detected: {max_allocation:.1%} in single entity")
            
            # Check for unused budget
            total_utilization = sum(alloc.current_risk for alloc in allocations) / budget.total_budget
            if total_utilization < 0.7:
                recommendations.append("Consider increasing position sizes - significant unused risk budget")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return ["Error generating recommendations"]
    
    def rebalance_risk_budget(self, budget_id: str, target_utilization: float = 0.85) -> Dict[str, float]:
        """
        Rebalance risk budget allocations
        
        Args:
            budget_id: Risk budget to rebalance
            target_utilization: Target overall utilization
            
        Returns:
            New allocation percentages
        """
        try:
            if budget_id not in self.risk_budgets:
                raise ValueError(f"Risk budget {budget_id} not found")
            
            budget = self.risk_budgets[budget_id]
            
            # Get latest report
            latest_report = self._get_latest_report(budget_id)
            if not latest_report:
                raise ValueError("No risk budget reports found")
            
            # Calculate new allocations
            new_allocations = {}
            total_current_risk = sum(alloc.current_risk for alloc in latest_report.allocations)
            
            if total_current_risk > 0:
                # Scale allocations to target utilization
                scale_factor = (target_utilization * budget.total_budget) / total_current_risk
                
                for allocation in latest_report.allocations:
                    new_risk_budget = allocation.current_risk * scale_factor
                    new_percent = new_risk_budget / budget.total_budget
                    
                    # Apply constraints
                    new_percent = max(budget.min_allocation, 
                                    min(new_percent, budget.max_single_allocation))
                    
                    new_allocations[allocation.entity_id] = new_percent
                
                # Normalize to sum to 1.0
                total_percent = sum(new_allocations.values())
                if total_percent > 0:
                    for entity_id in new_allocations:
                        new_allocations[entity_id] /= total_percent
            
            # Update budget
            budget.allocations = new_allocations
            budget.updated_at = datetime.now(timezone.utc)
            
            logger.info(f"Rebalanced risk budget {budget_id}")
            return new_allocations
            
        except Exception as e:
            logger.error(f"Error rebalancing risk budget: {e}")
            return {}
    
    def get_risk_budget_summary(self, budget_id: str = None) -> Dict[str, Any]:
        """Get risk budget summary"""
        try:
            if budget_id:
                if budget_id not in self.risk_budgets:
                    return {'error': f'Budget {budget_id} not found'}
                
                budget = self.risk_budgets[budget_id]
                latest_report = self._get_latest_report(budget_id)
                
                return {
                    'budget_id': budget_id,
                    'name': budget.name,
                    'total_budget': budget.total_budget,
                    'risk_metric': budget.risk_metric.value,
                    'allocation_type': budget.allocation_type.value,
                    'current_status': latest_report.status.value if latest_report else 'unknown',
                    'utilization': latest_report.budget_utilization if latest_report else 0.0,
                    'violations': len(latest_report.violations) if latest_report else 0,
                    'warnings': len(latest_report.warnings) if latest_report else 0,
                    'last_updated': budget.updated_at.isoformat()
                }
            else:
                # Summary of all budgets
                summaries = []
                for bid in self.risk_budgets:
                    summaries.append(self.get_risk_budget_summary(bid))
                
                return {
                    'total_budgets': len(self.risk_budgets),
                    'budgets': summaries
                }
                
        except Exception as e:
            logger.error(f"Error getting risk budget summary: {e}")
            return {'error': str(e)}
    
    # Helper methods
    def _get_entity_volatilities(self, entities: List[str]) -> Dict[str, float]:
        """Get volatilities for entities (mock implementation)"""
        # In production, this would fetch from analytics service
        return {entity: 0.2 + hash(entity) % 10 * 0.01 for entity in entities}
    
    def _get_entity_confidences(self, entities: List[str]) -> Dict[str, float]:
        """Get confidence scores for entities (mock implementation)"""
        # In production, this would fetch from signal service
        return {entity: 0.6 + hash(entity) % 30 * 0.01 for entity in entities}
    
    def _get_entity_average_confidence(self, entity_id: str, portfolio_data: Dict[str, Any]) -> float:
        """Get average confidence for entity"""
        # Mock implementation
        return 0.75
    
    def _get_latest_report(self, budget_id: str) -> Optional[RiskBudgetReport]:
        """Get latest report for budget"""
        reports = [r for r in self.allocation_history if r.budget_id == budget_id]
        return max(reports, key=lambda r: r.report_date) if reports else None
    
    def _get_utilization_trend(self, budget_id: str) -> List[float]:
        """Get 30-day utilization trend"""
        # Mock implementation
        return [0.6 + i * 0.01 for i in range(30)]
    
    def _get_max_utilization_30d(self, budget_id: str) -> float:
        """Get maximum utilization in last 30 days"""
        trend = self._get_utilization_trend(budget_id)
        return max(trend) if trend else 0.0
    
    def _get_avg_utilization_30d(self, budget_id: str) -> float:
        """Get average utilization in last 30 days"""
        trend = self._get_utilization_trend(budget_id)
        return sum(trend) / len(trend) if trend else 0.0

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create risk budget manager
    manager = RiskBudgetManager()
    
    # Create strategy-based risk budget
    strategies = ["momentum_breakout", "volatility_squeeze", "gamma_scalping", "earnings_play"]
    
    budget = manager.create_risk_budget(
        budget_id="strategy_var_budget",
        name="Strategy VaR Budget",
        total_budget=0.02,  # 2% VaR budget
        risk_metric=RiskMetric.VAR_95,
        allocation_type=RiskBudgetType.CONFIDENCE_WEIGHTED,
        entities=strategies,
        max_single_allocation=0.40,
        warning_threshold=0.75
    )
    
    print(f"Created risk budget: {budget.name}")
    print(f"Total budget: {budget.total_budget:.2%}")
    print(f"Allocations:")
    for entity, allocation in budget.allocations.items():
        print(f"  {entity}: {allocation:.2%}")
    
    # Mock portfolio data
    portfolio_data = {
        'positions': [
            {'symbol': 'AAPL', 'strategy_id': 'momentum_breakout', 'market_value': 10000, 'volatility': 0.25, 'beta': 1.2},
            {'symbol': 'MSFT', 'strategy_id': 'momentum_breakout', 'market_value': 8000, 'volatility': 0.22, 'beta': 1.1},
            {'symbol': 'GOOGL', 'strategy_id': 'volatility_squeeze', 'market_value': 12000, 'volatility': 0.28, 'beta': 1.3},
            {'symbol': 'TSLA', 'strategy_id': 'gamma_scalping', 'market_value': 5000, 'volatility': 0.45, 'beta': 1.8}
        ]
    }
    
    # Update risk allocations
    report = manager.update_risk_allocations("strategy_var_budget", portfolio_data, "normal")
    
    print(f"\nRisk Budget Report:")
    print(f"Total utilization: {report.budget_utilization:.1%}")
    print(f"Status: {report.status.value}")
    print(f"Violations: {len(report.violations)}")
    print(f"Warnings: {len(report.warnings)}")
    
    print(f"\nAllocations:")
    for allocation in report.allocations:
        print(f"  {allocation.entity_name}:")
        print(f"    Allocated: {allocation.allocated_percent:.1%}")
        print(f"    Used: {allocation.current_percent:.1%}")
        print(f"    Utilization: {allocation.utilization:.1%}")
        print(f"    Status: {allocation.status.value}")
    
    if report.recommendations:
        print(f"\nRecommendations:")
        for rec in report.recommendations:
            print(f"  - {rec}")
    
    print("\nRisk budgeting example completed")

