"""
Performance Attribution Analysis System
Advanced attribution analysis with strategy, sector, and factor-level breakdowns
"""

import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
from collections import defaultdict
import math

logger = logging.getLogger(__name__)

class AttributionLevel(Enum):
    """Attribution analysis levels"""
    STRATEGY = "strategy"
    SECTOR = "sector"
    SYMBOL = "symbol"
    FACTOR = "factor"
    STYLE = "style"
    GEOGRAPHIC = "geographic"

class AttributionMethod(Enum):
    """Attribution calculation methods"""
    BRINSON = "brinson"  # Brinson-Hood-Beebower
    ARITHMETIC = "arithmetic"
    GEOMETRIC = "geometric"
    INTERACTION = "interaction"
    FAMA_FRENCH = "fama_french"

class PerformanceComponent(Enum):
    """Performance attribution components"""
    ALLOCATION_EFFECT = "allocation_effect"
    SELECTION_EFFECT = "selection_effect"
    INTERACTION_EFFECT = "interaction_effect"
    CURRENCY_EFFECT = "currency_effect"
    TIMING_EFFECT = "timing_effect"

@dataclass
class AttributionSegment:
    """Attribution analysis segment"""
    segment_id: str
    segment_name: str
    segment_type: AttributionLevel
    
    # Portfolio data
    portfolio_weight: float
    portfolio_return: float
    portfolio_value: float
    
    # Benchmark data
    benchmark_weight: float
    benchmark_return: float
    
    # Attribution effects
    allocation_effect: float
    selection_effect: float
    interaction_effect: float
    total_effect: float
    
    # Risk metrics
    tracking_error: float
    information_ratio: float
    volatility: float
    sharpe_ratio: float
    
    # Additional metrics
    active_weight: float  # portfolio_weight - benchmark_weight
    active_return: float  # portfolio_return - benchmark_return
    contribution_to_return: float
    contribution_to_risk: float
    
    # Positions in segment
    position_count: int = 0
    positions: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class FactorAttribution:
    """Factor-based attribution analysis"""
    factor_name: str
    factor_type: str  # "style", "sector", "country", "custom"
    
    # Factor exposure
    portfolio_exposure: float
    benchmark_exposure: float
    active_exposure: float
    
    # Factor returns
    factor_return: float
    factor_volatility: float
    
    # Attribution
    factor_contribution: float
    specific_contribution: float
    
    # Risk decomposition
    factor_risk_contribution: float
    specific_risk_contribution: float

@dataclass
class AttributionReport:
    """Comprehensive attribution analysis report"""
    portfolio_id: str
    report_date: datetime
    analysis_period: str
    attribution_method: AttributionMethod
    
    # Overall performance
    portfolio_return: float
    benchmark_return: float
    active_return: float
    tracking_error: float
    information_ratio: float
    
    # Attribution breakdown
    segments: List[AttributionSegment]
    factor_attribution: List[FactorAttribution]
    
    # Summary effects
    total_allocation_effect: float
    total_selection_effect: float
    total_interaction_effect: float
    
    # Risk attribution
    total_active_risk: float
    systematic_risk: float
    specific_risk: float
    
    # Top contributors/detractors
    top_contributors: List[Dict[str, Any]]
    top_detractors: List[Dict[str, Any]]
    
    # Sector/strategy breakdown
    strategy_attribution: Dict[str, Dict[str, float]]
    sector_attribution: Dict[str, Dict[str, float]]

class PerformanceAttributor:
    """
    Advanced performance attribution analysis system
    Provides multi-level attribution with strategy, sector, and factor breakdowns
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        
        # Attribution parameters
        self.risk_free_rate = self.config.get('risk_free_rate', 0.02)
        self.attribution_frequency = self.config.get('attribution_frequency', 'daily')
        self.benchmark_symbol = self.config.get('benchmark_symbol', 'SPY')
        
        # Factor model parameters
        self.factor_models = self.config.get('factor_models', ['fama_french_3', 'momentum', 'quality'])
        self.style_factors = self.config.get('style_factors', ['value', 'growth', 'momentum', 'quality', 'volatility'])
        
        # Data storage
        self.attribution_history: Dict[str, List[AttributionReport]] = defaultdict(list)
        self.benchmark_data: Dict[str, Any] = {}
        self.factor_data: Dict[str, Any] = {}
        
        logger.info("Performance Attributor initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for attribution analysis"""
        return {
            'risk_free_rate': 0.02,
            'attribution_frequency': 'daily',
            'benchmark_symbol': 'SPY',
            'factor_models': ['fama_french_3', 'momentum', 'quality'],
            'style_factors': ['value', 'growth', 'momentum', 'quality', 'volatility'],
            'min_attribution_period': 30,  # Minimum 30 days for attribution
            'max_segments': 20,  # Maximum segments to analyze
            'significance_threshold': 0.001,  # 0.1% minimum contribution
            'cache_duration': 3600  # 1 hour cache
        }
    
    def calculate_attribution(self, portfolio_id: str, portfolio_data: Dict[str, Any],
                            benchmark_data: Dict[str, Any] = None,
                            period: str = "1M",
                            method: AttributionMethod = AttributionMethod.BRINSON) -> AttributionReport:
        """
        Calculate comprehensive performance attribution
        
        Args:
            portfolio_id: Portfolio identifier
            portfolio_data: Portfolio positions and performance data
            benchmark_data: Benchmark composition and returns
            period: Analysis period
            method: Attribution calculation method
            
        Returns:
            Attribution analysis report
        """
        try:
            # Get benchmark data if not provided
            if benchmark_data is None:
                benchmark_data = self._get_benchmark_data(period)
            
            # Calculate portfolio and benchmark returns
            portfolio_return = self._calculate_portfolio_return(portfolio_data, period)
            benchmark_return = self._calculate_benchmark_return(benchmark_data, period)
            active_return = portfolio_return - benchmark_return
            
            # Calculate tracking error and information ratio
            tracking_error = self._calculate_tracking_error(portfolio_data, benchmark_data, period)
            information_ratio = active_return / tracking_error if tracking_error > 0 else 0
            
            # Perform attribution analysis by level
            strategy_segments = self._calculate_strategy_attribution(portfolio_data, benchmark_data, method)
            sector_segments = self._calculate_sector_attribution(portfolio_data, benchmark_data, method)
            symbol_segments = self._calculate_symbol_attribution(portfolio_data, benchmark_data, method)
            
            # Combine all segments
            all_segments = strategy_segments + sector_segments + symbol_segments
            
            # Calculate factor attribution
            factor_attribution = self._calculate_factor_attribution(portfolio_data, benchmark_data, period)
            
            # Calculate summary effects
            total_allocation_effect = sum(seg.allocation_effect for seg in all_segments if seg.segment_type == AttributionLevel.SECTOR)
            total_selection_effect = sum(seg.selection_effect for seg in all_segments if seg.segment_type == AttributionLevel.SYMBOL)
            total_interaction_effect = sum(seg.interaction_effect for seg in all_segments)
            
            # Risk attribution
            total_active_risk = self._calculate_active_risk(portfolio_data, benchmark_data)
            systematic_risk, specific_risk = self._decompose_risk(portfolio_data, benchmark_data)
            
            # Top contributors and detractors
            top_contributors = self._get_top_contributors(all_segments, 5)
            top_detractors = self._get_top_detractors(all_segments, 5)
            
            # Create detailed breakdowns
            strategy_attribution = self._create_strategy_breakdown(strategy_segments)
            sector_attribution = self._create_sector_breakdown(sector_segments)
            
            # Create attribution report
            report = AttributionReport(
                portfolio_id=portfolio_id,
                report_date=datetime.now(timezone.utc),
                analysis_period=period,
                attribution_method=method,
                portfolio_return=portfolio_return,
                benchmark_return=benchmark_return,
                active_return=active_return,
                tracking_error=tracking_error,
                information_ratio=information_ratio,
                segments=all_segments,
                factor_attribution=factor_attribution,
                total_allocation_effect=total_allocation_effect,
                total_selection_effect=total_selection_effect,
                total_interaction_effect=total_interaction_effect,
                total_active_risk=total_active_risk,
                systematic_risk=systematic_risk,
                specific_risk=specific_risk,
                top_contributors=top_contributors,
                top_detractors=top_detractors,
                strategy_attribution=strategy_attribution,
                sector_attribution=sector_attribution
            )
            
            # Store report
            self.attribution_history[portfolio_id].append(report)
            
            return report
            
        except Exception as e:
            logger.error(f"Error calculating attribution: {e}")
            raise
    
    def _calculate_strategy_attribution(self, portfolio_data: Dict[str, Any], 
                                      benchmark_data: Dict[str, Any],
                                      method: AttributionMethod) -> List[AttributionSegment]:
        """Calculate attribution by trading strategy"""
        try:
            segments = []
            positions = portfolio_data.get('positions', [])
            
            # Group positions by strategy
            strategy_groups = defaultdict(list)
            for position in positions:
                strategy_id = position.get('strategy_id', 'unknown')
                strategy_groups[strategy_id].append(position)
            
            total_portfolio_value = sum(pos.get('market_value', 0) for pos in positions)
            
            for strategy_id, strategy_positions in strategy_groups.items():
                # Calculate strategy metrics
                strategy_value = sum(pos.get('market_value', 0) for pos in strategy_positions)
                strategy_weight = strategy_value / total_portfolio_value if total_portfolio_value > 0 else 0
                
                # Calculate strategy return
                strategy_return = self._calculate_segment_return(strategy_positions)
                
                # Get benchmark weight and return for this strategy (simplified)
                benchmark_weight = self._get_benchmark_strategy_weight(strategy_id, benchmark_data)
                benchmark_return = self._get_benchmark_strategy_return(strategy_id, benchmark_data)
                
                # Calculate attribution effects
                allocation_effect, selection_effect, interaction_effect = self._calculate_attribution_effects(
                    strategy_weight, strategy_return, benchmark_weight, benchmark_return, method
                )
                
                # Calculate additional metrics
                active_weight = strategy_weight - benchmark_weight
                active_return = strategy_return - benchmark_return
                contribution_to_return = strategy_weight * strategy_return
                
                # Risk metrics
                tracking_error = self._calculate_segment_tracking_error(strategy_positions, benchmark_data)
                information_ratio = active_return / tracking_error if tracking_error > 0 else 0
                volatility = self._calculate_segment_volatility(strategy_positions)
                sharpe_ratio = (strategy_return - self.risk_free_rate) / volatility if volatility > 0 else 0
                
                segment = AttributionSegment(
                    segment_id=strategy_id,
                    segment_name=strategy_id.replace('_', ' ').title(),
                    segment_type=AttributionLevel.STRATEGY,
                    portfolio_weight=strategy_weight,
                    portfolio_return=strategy_return,
                    portfolio_value=strategy_value,
                    benchmark_weight=benchmark_weight,
                    benchmark_return=benchmark_return,
                    allocation_effect=allocation_effect,
                    selection_effect=selection_effect,
                    interaction_effect=interaction_effect,
                    total_effect=allocation_effect + selection_effect + interaction_effect,
                    tracking_error=tracking_error,
                    information_ratio=information_ratio,
                    volatility=volatility,
                    sharpe_ratio=sharpe_ratio,
                    active_weight=active_weight,
                    active_return=active_return,
                    contribution_to_return=contribution_to_return,
                    contribution_to_risk=0.0,  # Would need covariance matrix
                    position_count=len(strategy_positions),
                    positions=strategy_positions
                )
                
                segments.append(segment)
            
            return segments
            
        except Exception as e:
            logger.error(f"Error calculating strategy attribution: {e}")
            return []
    
    def _calculate_sector_attribution(self, portfolio_data: Dict[str, Any], 
                                    benchmark_data: Dict[str, Any],
                                    method: AttributionMethod) -> List[AttributionSegment]:
        """Calculate attribution by sector"""
        try:
            segments = []
            positions = portfolio_data.get('positions', [])
            
            # Group positions by sector
            sector_groups = defaultdict(list)
            for position in positions:
                sector = position.get('sector', 'unknown')
                sector_groups[sector].append(position)
            
            total_portfolio_value = sum(pos.get('market_value', 0) for pos in positions)
            
            for sector, sector_positions in sector_groups.items():
                # Calculate sector metrics
                sector_value = sum(pos.get('market_value', 0) for pos in sector_positions)
                sector_weight = sector_value / total_portfolio_value if total_portfolio_value > 0 else 0
                
                # Calculate sector return
                sector_return = self._calculate_segment_return(sector_positions)
                
                # Get benchmark weight and return for this sector
                benchmark_weight = self._get_benchmark_sector_weight(sector, benchmark_data)
                benchmark_return = self._get_benchmark_sector_return(sector, benchmark_data)
                
                # Calculate attribution effects
                allocation_effect, selection_effect, interaction_effect = self._calculate_attribution_effects(
                    sector_weight, sector_return, benchmark_weight, benchmark_return, method
                )
                
                # Calculate additional metrics
                active_weight = sector_weight - benchmark_weight
                active_return = sector_return - benchmark_return
                contribution_to_return = sector_weight * sector_return
                
                # Risk metrics
                tracking_error = self._calculate_segment_tracking_error(sector_positions, benchmark_data)
                information_ratio = active_return / tracking_error if tracking_error > 0 else 0
                volatility = self._calculate_segment_volatility(sector_positions)
                sharpe_ratio = (sector_return - self.risk_free_rate) / volatility if volatility > 0 else 0
                
                segment = AttributionSegment(
                    segment_id=sector,
                    segment_name=sector.replace('_', ' ').title(),
                    segment_type=AttributionLevel.SECTOR,
                    portfolio_weight=sector_weight,
                    portfolio_return=sector_return,
                    portfolio_value=sector_value,
                    benchmark_weight=benchmark_weight,
                    benchmark_return=benchmark_return,
                    allocation_effect=allocation_effect,
                    selection_effect=selection_effect,
                    interaction_effect=interaction_effect,
                    total_effect=allocation_effect + selection_effect + interaction_effect,
                    tracking_error=tracking_error,
                    information_ratio=information_ratio,
                    volatility=volatility,
                    sharpe_ratio=sharpe_ratio,
                    active_weight=active_weight,
                    active_return=active_return,
                    contribution_to_return=contribution_to_return,
                    contribution_to_risk=0.0,
                    position_count=len(sector_positions),
                    positions=sector_positions
                )
                
                segments.append(segment)
            
            return segments
            
        except Exception as e:
            logger.error(f"Error calculating sector attribution: {e}")
            return []
    
    def _calculate_symbol_attribution(self, portfolio_data: Dict[str, Any], 
                                    benchmark_data: Dict[str, Any],
                                    method: AttributionMethod) -> List[AttributionSegment]:
        """Calculate attribution by individual symbols"""
        try:
            segments = []
            positions = portfolio_data.get('positions', [])
            total_portfolio_value = sum(pos.get('market_value', 0) for pos in positions)
            
            for position in positions:
                symbol = position.get('symbol', 'unknown')
                
                # Calculate position metrics
                position_value = position.get('market_value', 0)
                position_weight = position_value / total_portfolio_value if total_portfolio_value > 0 else 0
                
                # Calculate position return
                position_return = self._calculate_position_return(position)
                
                # Get benchmark weight and return for this symbol
                benchmark_weight = self._get_benchmark_symbol_weight(symbol, benchmark_data)
                benchmark_return = self._get_benchmark_symbol_return(symbol, benchmark_data)
                
                # Calculate attribution effects
                allocation_effect, selection_effect, interaction_effect = self._calculate_attribution_effects(
                    position_weight, position_return, benchmark_weight, benchmark_return, method
                )
                
                # Calculate additional metrics
                active_weight = position_weight - benchmark_weight
                active_return = position_return - benchmark_return
                contribution_to_return = position_weight * position_return
                
                # Risk metrics (simplified for individual positions)
                volatility = position.get('volatility', 0.2)
                sharpe_ratio = (position_return - self.risk_free_rate) / volatility if volatility > 0 else 0
                
                segment = AttributionSegment(
                    segment_id=symbol,
                    segment_name=symbol,
                    segment_type=AttributionLevel.SYMBOL,
                    portfolio_weight=position_weight,
                    portfolio_return=position_return,
                    portfolio_value=position_value,
                    benchmark_weight=benchmark_weight,
                    benchmark_return=benchmark_return,
                    allocation_effect=allocation_effect,
                    selection_effect=selection_effect,
                    interaction_effect=interaction_effect,
                    total_effect=allocation_effect + selection_effect + interaction_effect,
                    tracking_error=0.0,  # Individual position tracking error
                    information_ratio=0.0,
                    volatility=volatility,
                    sharpe_ratio=sharpe_ratio,
                    active_weight=active_weight,
                    active_return=active_return,
                    contribution_to_return=contribution_to_return,
                    contribution_to_risk=0.0,
                    position_count=1,
                    positions=[position]
                )
                
                segments.append(segment)
            
            return segments
            
        except Exception as e:
            logger.error(f"Error calculating symbol attribution: {e}")
            return []
    
    def _calculate_factor_attribution(self, portfolio_data: Dict[str, Any], 
                                    benchmark_data: Dict[str, Any],
                                    period: str) -> List[FactorAttribution]:
        """Calculate factor-based attribution"""
        try:
            factor_attributions = []
            
            # Style factors
            for factor in self.style_factors:
                # Calculate factor exposures
                portfolio_exposure = self._calculate_factor_exposure(portfolio_data, factor)
                benchmark_exposure = self._calculate_benchmark_factor_exposure(benchmark_data, factor)
                active_exposure = portfolio_exposure - benchmark_exposure
                
                # Get factor return and volatility
                factor_return = self._get_factor_return(factor, period)
                factor_volatility = self._get_factor_volatility(factor, period)
                
                # Calculate factor contribution
                factor_contribution = active_exposure * factor_return
                specific_contribution = 0.0  # Would need residual calculation
                
                # Risk decomposition
                factor_risk_contribution = (active_exposure ** 2) * (factor_volatility ** 2)
                specific_risk_contribution = 0.0  # Would need specific risk calculation
                
                factor_attr = FactorAttribution(
                    factor_name=factor,
                    factor_type="style",
                    portfolio_exposure=portfolio_exposure,
                    benchmark_exposure=benchmark_exposure,
                    active_exposure=active_exposure,
                    factor_return=factor_return,
                    factor_volatility=factor_volatility,
                    factor_contribution=factor_contribution,
                    specific_contribution=specific_contribution,
                    factor_risk_contribution=factor_risk_contribution,
                    specific_risk_contribution=specific_risk_contribution
                )
                
                factor_attributions.append(factor_attr)
            
            return factor_attributions
            
        except Exception as e:
            logger.error(f"Error calculating factor attribution: {e}")
            return []
    
    def _calculate_attribution_effects(self, portfolio_weight: float, portfolio_return: float,
                                     benchmark_weight: float, benchmark_return: float,
                                     method: AttributionMethod) -> Tuple[float, float, float]:
        """Calculate allocation, selection, and interaction effects"""
        try:
            if method == AttributionMethod.BRINSON:
                # Brinson-Hood-Beebower attribution
                allocation_effect = (portfolio_weight - benchmark_weight) * benchmark_return
                selection_effect = benchmark_weight * (portfolio_return - benchmark_return)
                interaction_effect = (portfolio_weight - benchmark_weight) * (portfolio_return - benchmark_return)
                
            elif method == AttributionMethod.ARITHMETIC:
                # Arithmetic attribution
                allocation_effect = (portfolio_weight - benchmark_weight) * benchmark_return
                selection_effect = benchmark_weight * (portfolio_return - benchmark_return)
                interaction_effect = 0.0  # No interaction in arithmetic method
                
            elif method == AttributionMethod.GEOMETRIC:
                # Geometric attribution (simplified)
                total_effect = portfolio_weight * portfolio_return - benchmark_weight * benchmark_return
                allocation_effect = (portfolio_weight - benchmark_weight) * benchmark_return
                selection_effect = total_effect - allocation_effect
                interaction_effect = 0.0
                
            else:
                # Default to Brinson method
                allocation_effect = (portfolio_weight - benchmark_weight) * benchmark_return
                selection_effect = benchmark_weight * (portfolio_return - benchmark_return)
                interaction_effect = (portfolio_weight - benchmark_weight) * (portfolio_return - benchmark_return)
            
            return allocation_effect, selection_effect, interaction_effect
            
        except Exception as e:
            logger.error(f"Error calculating attribution effects: {e}")
            return 0.0, 0.0, 0.0
    
    def get_attribution_summary(self, portfolio_id: str, period: str = "1M") -> Dict[str, Any]:
        """Get attribution analysis summary"""
        try:
            # Get latest attribution report
            if portfolio_id not in self.attribution_history or not self.attribution_history[portfolio_id]:
                return {'error': 'No attribution data available'}
            
            latest_report = self.attribution_history[portfolio_id][-1]
            
            # Create summary
            summary = {
                'portfolio_id': portfolio_id,
                'report_date': latest_report.report_date.isoformat(),
                'analysis_period': latest_report.analysis_period,
                'portfolio_return': latest_report.portfolio_return,
                'benchmark_return': latest_report.benchmark_return,
                'active_return': latest_report.active_return,
                'tracking_error': latest_report.tracking_error,
                'information_ratio': latest_report.information_ratio,
                'total_allocation_effect': latest_report.total_allocation_effect,
                'total_selection_effect': latest_report.total_selection_effect,
                'total_interaction_effect': latest_report.total_interaction_effect,
                'top_contributors': latest_report.top_contributors,
                'top_detractors': latest_report.top_detractors,
                'strategy_breakdown': latest_report.strategy_attribution,
                'sector_breakdown': latest_report.sector_attribution
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting attribution summary: {e}")
            return {'error': str(e)}
    
    def get_detailed_attribution(self, portfolio_id: str, level: AttributionLevel) -> List[Dict[str, Any]]:
        """Get detailed attribution for specific level"""
        try:
            if portfolio_id not in self.attribution_history or not self.attribution_history[portfolio_id]:
                return []
            
            latest_report = self.attribution_history[portfolio_id][-1]
            
            # Filter segments by level
            segments = [seg for seg in latest_report.segments if seg.segment_type == level]
            
            # Convert to dictionaries
            detailed_attribution = []
            for segment in segments:
                attr_dict = {
                    'segment_id': segment.segment_id,
                    'segment_name': segment.segment_name,
                    'portfolio_weight': segment.portfolio_weight,
                    'portfolio_return': segment.portfolio_return,
                    'benchmark_weight': segment.benchmark_weight,
                    'benchmark_return': segment.benchmark_return,
                    'active_weight': segment.active_weight,
                    'active_return': segment.active_return,
                    'allocation_effect': segment.allocation_effect,
                    'selection_effect': segment.selection_effect,
                    'interaction_effect': segment.interaction_effect,
                    'total_effect': segment.total_effect,
                    'contribution_to_return': segment.contribution_to_return,
                    'tracking_error': segment.tracking_error,
                    'information_ratio': segment.information_ratio,
                    'sharpe_ratio': segment.sharpe_ratio,
                    'position_count': segment.position_count
                }
                detailed_attribution.append(attr_dict)
            
            # Sort by total effect (descending)
            detailed_attribution.sort(key=lambda x: x['total_effect'], reverse=True)
            
            return detailed_attribution
            
        except Exception as e:
            logger.error(f"Error getting detailed attribution: {e}")
            return []
    
    # Helper methods (simplified implementations)
    def _get_benchmark_data(self, period: str) -> Dict[str, Any]:
        """Get benchmark composition and returns"""
        # Mock implementation - in production would fetch real benchmark data
        return {
            'symbol': self.benchmark_symbol,
            'return': 0.05,  # 5% benchmark return
            'volatility': 0.15,
            'sectors': {
                'technology': 0.25,
                'healthcare': 0.15,
                'finance': 0.20,
                'consumer': 0.15,
                'industrial': 0.10,
                'energy': 0.05,
                'utilities': 0.05,
                'materials': 0.05
            }
        }
    
    def _calculate_portfolio_return(self, portfolio_data: Dict[str, Any], period: str) -> float:
        """Calculate portfolio return for period"""
        # Simplified calculation - in production would use time-weighted returns
        positions = portfolio_data.get('positions', [])
        total_value = sum(pos.get('market_value', 0) for pos in positions)
        total_cost = sum(pos.get('cost_basis', 0) for pos in positions)
        
        if total_cost > 0:
            return (total_value - total_cost) / total_cost
        return 0.0
    
    def _calculate_benchmark_return(self, benchmark_data: Dict[str, Any], period: str) -> float:
        """Calculate benchmark return for period"""
        return benchmark_data.get('return', 0.05)
    
    def _calculate_tracking_error(self, portfolio_data: Dict[str, Any], 
                                benchmark_data: Dict[str, Any], period: str) -> float:
        """Calculate tracking error"""
        # Simplified calculation - in production would use time series of returns
        return 0.02  # 2% tracking error
    
    def _calculate_segment_return(self, positions: List[Dict[str, Any]]) -> float:
        """Calculate return for a segment of positions"""
        total_value = sum(pos.get('market_value', 0) for pos in positions)
        total_cost = sum(pos.get('cost_basis', 0) for pos in positions)
        
        if total_cost > 0:
            return (total_value - total_cost) / total_cost
        return 0.0
    
    def _calculate_position_return(self, position: Dict[str, Any]) -> float:
        """Calculate return for individual position"""
        market_value = position.get('market_value', 0)
        cost_basis = position.get('cost_basis', 0)
        
        if cost_basis > 0:
            return (market_value - cost_basis) / cost_basis
        return 0.0
    
    def _get_benchmark_strategy_weight(self, strategy_id: str, benchmark_data: Dict[str, Any]) -> float:
        """Get benchmark weight for strategy (mock)"""
        # In production, would map strategies to benchmark components
        strategy_weights = {
            'momentum_breakout': 0.15,
            'volatility_squeeze': 0.10,
            'gamma_scalping': 0.05,
            'earnings_play': 0.08
        }
        return strategy_weights.get(strategy_id, 0.0)
    
    def _get_benchmark_strategy_return(self, strategy_id: str, benchmark_data: Dict[str, Any]) -> float:
        """Get benchmark return for strategy (mock)"""
        # In production, would calculate strategy-specific benchmark returns
        return benchmark_data.get('return', 0.05) * np.random.uniform(0.8, 1.2)
    
    def _get_benchmark_sector_weight(self, sector: str, benchmark_data: Dict[str, Any]) -> float:
        """Get benchmark weight for sector"""
        sectors = benchmark_data.get('sectors', {})
        return sectors.get(sector, 0.0)
    
    def _get_benchmark_sector_return(self, sector: str, benchmark_data: Dict[str, Any]) -> float:
        """Get benchmark return for sector (mock)"""
        # In production, would fetch sector-specific returns
        sector_returns = {
            'technology': 0.08,
            'healthcare': 0.06,
            'finance': 0.04,
            'consumer': 0.05,
            'industrial': 0.03,
            'energy': 0.02,
            'utilities': 0.01,
            'materials': 0.03
        }
        return sector_returns.get(sector, 0.05)
    
    def _get_benchmark_symbol_weight(self, symbol: str, benchmark_data: Dict[str, Any]) -> float:
        """Get benchmark weight for symbol (mock)"""
        # In production, would fetch actual benchmark weights
        symbol_weights = {
            'AAPL': 0.07, 'MSFT': 0.06, 'GOOGL': 0.04, 'AMZN': 0.03,
            'TSLA': 0.02, 'NVDA': 0.02, 'META': 0.02
        }
        return symbol_weights.get(symbol, 0.0)
    
    def _get_benchmark_symbol_return(self, symbol: str, benchmark_data: Dict[str, Any]) -> float:
        """Get benchmark return for symbol (mock)"""
        # In production, would fetch actual symbol returns
        return benchmark_data.get('return', 0.05) * np.random.uniform(0.5, 1.5)
    
    def _calculate_segment_tracking_error(self, positions: List[Dict[str, Any]], 
                                        benchmark_data: Dict[str, Any]) -> float:
        """Calculate tracking error for segment"""
        # Simplified calculation
        return 0.03  # 3% tracking error
    
    def _calculate_segment_volatility(self, positions: List[Dict[str, Any]]) -> float:
        """Calculate volatility for segment"""
        if not positions:
            return 0.0
        
        # Weighted average volatility
        total_value = sum(pos.get('market_value', 0) for pos in positions)
        if total_value == 0:
            return 0.0
        
        weighted_vol = 0.0
        for pos in positions:
            weight = pos.get('market_value', 0) / total_value
            volatility = pos.get('volatility', 0.2)
            weighted_vol += weight * volatility
        
        return weighted_vol
    
    def _calculate_active_risk(self, portfolio_data: Dict[str, Any], 
                             benchmark_data: Dict[str, Any]) -> float:
        """Calculate total active risk"""
        # Simplified calculation
        return 0.04  # 4% active risk
    
    def _decompose_risk(self, portfolio_data: Dict[str, Any], 
                       benchmark_data: Dict[str, Any]) -> Tuple[float, float]:
        """Decompose risk into systematic and specific components"""
        total_risk = self._calculate_active_risk(portfolio_data, benchmark_data)
        systematic_risk = total_risk * 0.7  # 70% systematic
        specific_risk = total_risk * 0.3    # 30% specific
        return systematic_risk, specific_risk
    
    def _get_top_contributors(self, segments: List[AttributionSegment], count: int) -> List[Dict[str, Any]]:
        """Get top contributing segments"""
        sorted_segments = sorted(segments, key=lambda s: s.total_effect, reverse=True)
        
        contributors = []
        for segment in sorted_segments[:count]:
            if segment.total_effect > 0:
                contributors.append({
                    'name': segment.segment_name,
                    'type': segment.segment_type.value,
                    'contribution': segment.total_effect,
                    'weight': segment.portfolio_weight,
                    'return': segment.portfolio_return
                })
        
        return contributors
    
    def _get_top_detractors(self, segments: List[AttributionSegment], count: int) -> List[Dict[str, Any]]:
        """Get top detracting segments"""
        sorted_segments = sorted(segments, key=lambda s: s.total_effect)
        
        detractors = []
        for segment in sorted_segments[:count]:
            if segment.total_effect < 0:
                detractors.append({
                    'name': segment.segment_name,
                    'type': segment.segment_type.value,
                    'contribution': segment.total_effect,
                    'weight': segment.portfolio_weight,
                    'return': segment.portfolio_return
                })
        
        return detractors
    
    def _create_strategy_breakdown(self, strategy_segments: List[AttributionSegment]) -> Dict[str, Dict[str, float]]:
        """Create strategy attribution breakdown"""
        breakdown = {}
        
        for segment in strategy_segments:
            breakdown[segment.segment_id] = {
                'weight': segment.portfolio_weight,
                'return': segment.portfolio_return,
                'allocation_effect': segment.allocation_effect,
                'selection_effect': segment.selection_effect,
                'total_effect': segment.total_effect,
                'information_ratio': segment.information_ratio
            }
        
        return breakdown
    
    def _create_sector_breakdown(self, sector_segments: List[AttributionSegment]) -> Dict[str, Dict[str, float]]:
        """Create sector attribution breakdown"""
        breakdown = {}
        
        for segment in sector_segments:
            breakdown[segment.segment_id] = {
                'weight': segment.portfolio_weight,
                'return': segment.portfolio_return,
                'allocation_effect': segment.allocation_effect,
                'selection_effect': segment.selection_effect,
                'total_effect': segment.total_effect,
                'information_ratio': segment.information_ratio
            }
        
        return breakdown
    
    def _calculate_factor_exposure(self, portfolio_data: Dict[str, Any], factor: str) -> float:
        """Calculate portfolio exposure to factor"""
        # Mock implementation - in production would use factor model
        factor_exposures = {
            'value': 0.2, 'growth': 0.3, 'momentum': 0.4, 
            'quality': 0.1, 'volatility': -0.1
        }
        return factor_exposures.get(factor, 0.0)
    
    def _calculate_benchmark_factor_exposure(self, benchmark_data: Dict[str, Any], factor: str) -> float:
        """Calculate benchmark exposure to factor"""
        # Mock implementation
        return 0.0  # Assume benchmark is factor-neutral
    
    def _get_factor_return(self, factor: str, period: str) -> float:
        """Get factor return for period"""
        # Mock implementation
        factor_returns = {
            'value': 0.02, 'growth': 0.08, 'momentum': 0.06,
            'quality': 0.04, 'volatility': -0.03
        }
        return factor_returns.get(factor, 0.0)
    
    def _get_factor_volatility(self, factor: str, period: str) -> float:
        """Get factor volatility for period"""
        # Mock implementation
        factor_volatilities = {
            'value': 0.15, 'growth': 0.20, 'momentum': 0.18,
            'quality': 0.12, 'volatility': 0.25
        }
        return factor_volatilities.get(factor, 0.15)

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create performance attributor
    attributor = PerformanceAttributor()
    
    # Mock portfolio data
    portfolio_data = {
        'positions': [
            {
                'symbol': 'AAPL', 'market_value': 15000, 'cost_basis': 14000,
                'strategy_id': 'momentum_breakout', 'sector': 'technology', 'volatility': 0.25
            },
            {
                'symbol': 'MSFT', 'market_value': 12000, 'cost_basis': 11500,
                'strategy_id': 'momentum_breakout', 'sector': 'technology', 'volatility': 0.22
            },
            {
                'symbol': 'GOOGL', 'market_value': 8000, 'cost_basis': 8500,
                'strategy_id': 'volatility_squeeze', 'sector': 'technology', 'volatility': 0.28
            },
            {
                'symbol': 'JNJ', 'market_value': 10000, 'cost_basis': 9800,
                'strategy_id': 'gamma_scalping', 'sector': 'healthcare', 'volatility': 0.18
            }
        ]
    }
    
    # Calculate attribution
    report = attributor.calculate_attribution("test_portfolio", portfolio_data, period="1M")
    
    print("Performance Attribution Analysis:")
    print("=" * 50)
    print(f"Portfolio Return: {report.portfolio_return:.2%}")
    print(f"Benchmark Return: {report.benchmark_return:.2%}")
    print(f"Active Return: {report.active_return:.2%}")
    print(f"Tracking Error: {report.tracking_error:.2%}")
    print(f"Information Ratio: {report.information_ratio:.3f}")
    
    print(f"\nAttribution Effects:")
    print(f"  Allocation Effect: {report.total_allocation_effect:.4f}")
    print(f"  Selection Effect: {report.total_selection_effect:.4f}")
    print(f"  Interaction Effect: {report.total_interaction_effect:.4f}")
    
    print(f"\nTop Contributors:")
    for contributor in report.top_contributors:
        print(f"  {contributor['name']}: {contributor['contribution']:.4f}")
    
    print(f"\nTop Detractors:")
    for detractor in report.top_detractors:
        print(f"  {detractor['name']}: {detractor['contribution']:.4f}")
    
    print(f"\nStrategy Attribution:")
    for strategy, metrics in report.strategy_attribution.items():
        print(f"  {strategy}:")
        print(f"    Weight: {metrics['weight']:.2%}")
        print(f"    Return: {metrics['return']:.2%}")
        print(f"    Total Effect: {metrics['total_effect']:.4f}")
    
    print(f"\nSector Attribution:")
    for sector, metrics in report.sector_attribution.items():
        print(f"  {sector}:")
        print(f"    Weight: {metrics['weight']:.2%}")
        print(f"    Return: {metrics['return']:.2%}")
        print(f"    Total Effect: {metrics['total_effect']:.4f}")
    
    print("\nPerformance attribution example completed")

