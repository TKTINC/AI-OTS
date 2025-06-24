"""
Stress Testing Framework
Advanced scenario analysis and portfolio stress testing

This module provides comprehensive stress testing capabilities including:
- Market scenario simulation
- Portfolio stress testing
- Monte Carlo analysis
- Historical scenario replay
- Custom stress scenarios
- Risk factor sensitivity analysis

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
import math
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import redis
import psycopg2
from psycopg2.extras import RealDictCursor
from scipy import stats
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

class ScenarioType(Enum):
    """Types of stress test scenarios"""
    MARKET_CRASH = "market_crash"           # Market crash scenarios
    VOLATILITY_SPIKE = "volatility_spike"   # Volatility spike scenarios
    INTEREST_RATE = "interest_rate"         # Interest rate scenarios
    SECTOR_ROTATION = "sector_rotation"     # Sector rotation scenarios
    CORRELATION_BREAKDOWN = "correlation_breakdown"  # Correlation breakdown
    LIQUIDITY_CRISIS = "liquidity_crisis"   # Liquidity crisis scenarios
    HISTORICAL_REPLAY = "historical_replay" # Historical event replay
    MONTE_CARLO = "monte_carlo"             # Monte Carlo simulation
    CUSTOM = "custom"                       # Custom scenarios

class StressLevel(Enum):
    """Stress test severity levels"""
    MILD = "mild"           # 1-2 standard deviations
    MODERATE = "moderate"   # 2-3 standard deviations
    SEVERE = "severe"       # 3-4 standard deviations
    EXTREME = "extreme"     # 4+ standard deviations

class RiskFactor(Enum):
    """Risk factors for stress testing"""
    EQUITY_PRICE = "equity_price"
    VOLATILITY = "volatility"
    INTEREST_RATE = "interest_rate"
    CREDIT_SPREAD = "credit_spread"
    CURRENCY = "currency"
    COMMODITY = "commodity"
    CORRELATION = "correlation"
    LIQUIDITY = "liquidity"

@dataclass
class StressScenario:
    """Stress test scenario definition"""
    scenario_id: str
    scenario_name: str
    scenario_type: ScenarioType
    stress_level: StressLevel
    description: str
    risk_factors: Dict[RiskFactor, float]  # Factor -> shock magnitude
    duration_days: int
    probability: float  # Estimated probability of occurrence
    created_at: datetime
    is_active: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['scenario_type'] = self.scenario_type.value
        data['stress_level'] = self.stress_level.value
        data['risk_factors'] = {k.value: v for k, v in self.risk_factors.items()}
        data['created_at'] = self.created_at.isoformat()
        return data

@dataclass
class StressTestResult:
    """Stress test result"""
    test_id: str
    portfolio_id: str
    scenario_id: str
    scenario_name: str
    test_timestamp: datetime
    
    # Portfolio metrics before stress
    initial_value: float
    initial_var_95: float
    initial_var_99: float
    initial_expected_shortfall: float
    
    # Portfolio metrics after stress
    stressed_value: float
    stressed_var_95: float
    stressed_var_99: float
    stressed_expected_shortfall: float
    
    # Stress impact
    absolute_loss: float
    percentage_loss: float
    var_increase: float
    expected_shortfall_increase: float
    
    # Position-level impacts
    position_impacts: Dict[str, float]  # symbol -> impact
    sector_impacts: Dict[str, float]    # sector -> impact
    strategy_impacts: Dict[str, float]  # strategy -> impact
    
    # Risk metrics
    stress_ratio: float  # stressed_loss / normal_var
    tail_risk_ratio: float
    concentration_risk: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['test_timestamp'] = self.test_timestamp.isoformat()
        return data

@dataclass
class MonteCarloResult:
    """Monte Carlo simulation result"""
    simulation_id: str
    portfolio_id: str
    num_simulations: int
    time_horizon_days: int
    confidence_level: float
    
    # Distribution statistics
    mean_return: float
    std_return: float
    skewness: float
    kurtosis: float
    
    # Risk metrics
    var_95: float
    var_99: float
    expected_shortfall_95: float
    expected_shortfall_99: float
    
    # Percentiles
    percentile_5: float
    percentile_10: float
    percentile_25: float
    percentile_75: float
    percentile_90: float
    percentile_95: float
    
    # Probability metrics
    prob_loss: float
    prob_large_loss: float  # > 10%
    prob_extreme_loss: float  # > 20%
    
    simulation_timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['simulation_timestamp'] = self.simulation_timestamp.isoformat()
        return data

class ScenarioGenerator:
    """Generates stress test scenarios"""
    
    def __init__(self):
        self.historical_events = self._load_historical_events()
    
    def generate_market_crash_scenarios(self) -> List[StressScenario]:
        """Generate market crash scenarios"""
        scenarios = []
        
        # Mild crash (2008-style)
        mild_crash = StressScenario(
            scenario_id=str(uuid.uuid4()),
            scenario_name="Mild Market Crash",
            scenario_type=ScenarioType.MARKET_CRASH,
            stress_level=StressLevel.MILD,
            description="2008-style financial crisis with 20-30% market decline",
            risk_factors={
                RiskFactor.EQUITY_PRICE: -0.25,  # 25% decline
                RiskFactor.VOLATILITY: 2.0,      # 100% increase in volatility
                RiskFactor.CORRELATION: 0.3,     # Increased correlation
                RiskFactor.LIQUIDITY: -0.4       # 40% liquidity reduction
            },
            duration_days=90,
            probability=0.05,  # 5% annual probability
            created_at=datetime.now()
        )
        scenarios.append(mild_crash)
        
        # Severe crash (1987-style)
        severe_crash = StressScenario(
            scenario_id=str(uuid.uuid4()),
            scenario_name="Severe Market Crash",
            scenario_type=ScenarioType.MARKET_CRASH,
            stress_level=StressLevel.SEVERE,
            description="1987-style black Monday with 40-50% market decline",
            risk_factors={
                RiskFactor.EQUITY_PRICE: -0.45,  # 45% decline
                RiskFactor.VOLATILITY: 3.0,      # 200% increase in volatility
                RiskFactor.CORRELATION: 0.5,     # High correlation
                RiskFactor.LIQUIDITY: -0.6       # 60% liquidity reduction
            },
            duration_days=30,
            probability=0.01,  # 1% annual probability
            created_at=datetime.now()
        )
        scenarios.append(severe_crash)
        
        # Extreme crash (1929-style)
        extreme_crash = StressScenario(
            scenario_id=str(uuid.uuid4()),
            scenario_name="Extreme Market Crash",
            scenario_type=ScenarioType.MARKET_CRASH,
            stress_level=StressLevel.EXTREME,
            description="1929-style great depression with 60%+ market decline",
            risk_factors={
                RiskFactor.EQUITY_PRICE: -0.65,  # 65% decline
                RiskFactor.VOLATILITY: 4.0,      # 300% increase in volatility
                RiskFactor.CORRELATION: 0.7,     # Very high correlation
                RiskFactor.LIQUIDITY: -0.8       # 80% liquidity reduction
            },
            duration_days=180,
            probability=0.002,  # 0.2% annual probability
            created_at=datetime.now()
        )
        scenarios.append(extreme_crash)
        
        return scenarios
    
    def generate_volatility_spike_scenarios(self) -> List[StressScenario]:
        """Generate volatility spike scenarios"""
        scenarios = []
        
        # VIX spike to 40
        vix_40 = StressScenario(
            scenario_id=str(uuid.uuid4()),
            scenario_name="VIX Spike to 40",
            scenario_type=ScenarioType.VOLATILITY_SPIKE,
            stress_level=StressLevel.MODERATE,
            description="VIX spikes to 40 with increased market volatility",
            risk_factors={
                RiskFactor.VOLATILITY: 1.5,      # 50% increase
                RiskFactor.EQUITY_PRICE: -0.10,  # 10% decline
                RiskFactor.CORRELATION: 0.2      # Increased correlation
            },
            duration_days=14,
            probability=0.15,  # 15% annual probability
            created_at=datetime.now()
        )
        scenarios.append(vix_40)
        
        # VIX spike to 60
        vix_60 = StressScenario(
            scenario_id=str(uuid.uuid4()),
            scenario_name="VIX Spike to 60",
            scenario_type=ScenarioType.VOLATILITY_SPIKE,
            stress_level=StressLevel.SEVERE,
            description="VIX spikes to 60 with extreme market volatility",
            risk_factors={
                RiskFactor.VOLATILITY: 2.5,      # 150% increase
                RiskFactor.EQUITY_PRICE: -0.15,  # 15% decline
                RiskFactor.CORRELATION: 0.4      # High correlation
            },
            duration_days=7,
            probability=0.05,  # 5% annual probability
            created_at=datetime.now()
        )
        scenarios.append(vix_60)
        
        return scenarios
    
    def generate_interest_rate_scenarios(self) -> List[StressScenario]:
        """Generate interest rate scenarios"""
        scenarios = []
        
        # Rate hike scenario
        rate_hike = StressScenario(
            scenario_id=str(uuid.uuid4()),
            scenario_name="Aggressive Rate Hikes",
            scenario_type=ScenarioType.INTEREST_RATE,
            stress_level=StressLevel.MODERATE,
            description="Fed raises rates by 200 basis points rapidly",
            risk_factors={
                RiskFactor.INTEREST_RATE: 0.02,  # 200 basis points
                RiskFactor.EQUITY_PRICE: -0.12,  # 12% decline
                RiskFactor.VOLATILITY: 0.5       # 50% vol increase
            },
            duration_days=60,
            probability=0.10,  # 10% annual probability
            created_at=datetime.now()
        )
        scenarios.append(rate_hike)
        
        # Rate cut scenario
        rate_cut = StressScenario(
            scenario_id=str(uuid.uuid4()),
            scenario_name="Emergency Rate Cuts",
            scenario_type=ScenarioType.INTEREST_RATE,
            stress_level=StressLevel.MODERATE,
            description="Fed cuts rates to zero in emergency response",
            risk_factors={
                RiskFactor.INTEREST_RATE: -0.025,  # 250 basis points cut
                RiskFactor.EQUITY_PRICE: -0.08,    # 8% initial decline
                RiskFactor.VOLATILITY: 1.0         # 100% vol increase
            },
            duration_days=30,
            probability=0.08,  # 8% annual probability
            created_at=datetime.now()
        )
        scenarios.append(rate_cut)
        
        return scenarios
    
    def generate_sector_rotation_scenarios(self) -> List[StressScenario]:
        """Generate sector rotation scenarios"""
        scenarios = []
        
        # Tech selloff
        tech_selloff = StressScenario(
            scenario_id=str(uuid.uuid4()),
            scenario_name="Technology Sector Selloff",
            scenario_type=ScenarioType.SECTOR_ROTATION,
            stress_level=StressLevel.MODERATE,
            description="Major rotation out of technology stocks",
            risk_factors={
                RiskFactor.EQUITY_PRICE: -0.20,  # 20% tech decline
                RiskFactor.VOLATILITY: 0.8,      # 80% vol increase
                RiskFactor.CORRELATION: -0.2     # Decreased correlation
            },
            duration_days=45,
            probability=0.12,  # 12% annual probability
            created_at=datetime.now()
        )
        scenarios.append(tech_selloff)
        
        return scenarios
    
    def generate_custom_scenario(self, scenario_config: Dict[str, Any]) -> StressScenario:
        """Generate custom stress scenario"""
        return StressScenario(
            scenario_id=str(uuid.uuid4()),
            scenario_name=scenario_config['name'],
            scenario_type=ScenarioType.CUSTOM,
            stress_level=StressLevel(scenario_config['stress_level']),
            description=scenario_config['description'],
            risk_factors={
                RiskFactor(k): v for k, v in scenario_config['risk_factors'].items()
            },
            duration_days=scenario_config['duration_days'],
            probability=scenario_config.get('probability', 0.05),
            created_at=datetime.now()
        )
    
    def _load_historical_events(self) -> Dict[str, Dict[str, Any]]:
        """Load historical market events for replay scenarios"""
        return {
            "black_monday_1987": {
                "date": "1987-10-19",
                "equity_shock": -0.22,
                "volatility_shock": 3.0,
                "duration": 5
            },
            "dot_com_crash_2000": {
                "date": "2000-03-10",
                "equity_shock": -0.78,
                "volatility_shock": 2.5,
                "duration": 365
            },
            "financial_crisis_2008": {
                "date": "2008-09-15",
                "equity_shock": -0.57,
                "volatility_shock": 2.8,
                "duration": 180
            },
            "covid_crash_2020": {
                "date": "2020-02-20",
                "equity_shock": -0.34,
                "volatility_shock": 4.0,
                "duration": 30
            }
        }

class PortfolioStressTester:
    """Performs stress tests on portfolios"""
    
    def __init__(self, db_connection, redis_client):
        self.db_connection = db_connection
        self.redis_client = redis_client
        self.scenario_generator = ScenarioGenerator()
    
    def run_stress_test(self, portfolio_id: str, scenario: StressScenario) -> StressTestResult:
        """Run stress test on portfolio"""
        try:
            # Get portfolio data
            portfolio_data = self._get_portfolio_data(portfolio_id)
            if not portfolio_data:
                raise ValueError(f"Portfolio {portfolio_id} not found")
            
            # Calculate initial metrics
            initial_metrics = self._calculate_initial_metrics(portfolio_data)
            
            # Apply stress scenario
            stressed_portfolio = self._apply_stress_scenario(portfolio_data, scenario)
            
            # Calculate stressed metrics
            stressed_metrics = self._calculate_stressed_metrics(stressed_portfolio)
            
            # Calculate impacts
            impacts = self._calculate_impacts(portfolio_data, stressed_portfolio)
            
            # Create result
            result = StressTestResult(
                test_id=str(uuid.uuid4()),
                portfolio_id=portfolio_id,
                scenario_id=scenario.scenario_id,
                scenario_name=scenario.scenario_name,
                test_timestamp=datetime.now(),
                
                # Initial metrics
                initial_value=initial_metrics['value'],
                initial_var_95=initial_metrics['var_95'],
                initial_var_99=initial_metrics['var_99'],
                initial_expected_shortfall=initial_metrics['expected_shortfall'],
                
                # Stressed metrics
                stressed_value=stressed_metrics['value'],
                stressed_var_95=stressed_metrics['var_95'],
                stressed_var_99=stressed_metrics['var_99'],
                stressed_expected_shortfall=stressed_metrics['expected_shortfall'],
                
                # Impact metrics
                absolute_loss=initial_metrics['value'] - stressed_metrics['value'],
                percentage_loss=(initial_metrics['value'] - stressed_metrics['value']) / initial_metrics['value'],
                var_increase=stressed_metrics['var_95'] - initial_metrics['var_95'],
                expected_shortfall_increase=stressed_metrics['expected_shortfall'] - initial_metrics['expected_shortfall'],
                
                # Detailed impacts
                position_impacts=impacts['positions'],
                sector_impacts=impacts['sectors'],
                strategy_impacts=impacts['strategies'],
                
                # Risk ratios
                stress_ratio=(initial_metrics['value'] - stressed_metrics['value']) / initial_metrics['var_95'],
                tail_risk_ratio=stressed_metrics['expected_shortfall'] / initial_metrics['expected_shortfall'],
                concentration_risk=impacts['concentration_risk']
            )
            
            # Store result
            self._store_stress_test_result(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error running stress test: {e}")
            raise
    
    def run_monte_carlo_simulation(self, portfolio_id: str, num_simulations: int = 10000,
                                 time_horizon_days: int = 252) -> MonteCarloResult:
        """Run Monte Carlo simulation on portfolio"""
        try:
            # Get portfolio data
            portfolio_data = self._get_portfolio_data(portfolio_id)
            if not portfolio_data:
                raise ValueError(f"Portfolio {portfolio_id} not found")
            
            # Get historical returns
            returns_data = self._get_historical_returns(portfolio_data, lookback_days=252)
            
            # Run simulation
            simulated_returns = self._run_monte_carlo(returns_data, num_simulations, time_horizon_days)
            
            # Calculate statistics
            result = self._calculate_monte_carlo_statistics(
                portfolio_id, simulated_returns, num_simulations, time_horizon_days
            )
            
            # Store result
            self._store_monte_carlo_result(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error running Monte Carlo simulation: {e}")
            raise
    
    def _get_portfolio_data(self, portfolio_id: str) -> Optional[Dict[str, Any]]:
        """Get portfolio data for stress testing"""
        try:
            with self.db_connection.cursor(cursor_factory=RealDictCursor) as cursor:
                # Get positions
                cursor.execute("""
                    SELECT p.symbol, p.quantity, p.market_value, p.cost_basis,
                           p.strategy_id, p.position_type,
                           s.sector, s.market_cap, s.beta, s.volatility
                    FROM positions p
                    LEFT JOIN symbols s ON p.symbol = s.symbol
                    WHERE p.portfolio_id = %s AND p.status = 'open'
                """, (portfolio_id,))
                
                positions = [dict(row) for row in cursor.fetchall()]
                
                if not positions:
                    return None
                
                # Get portfolio summary
                cursor.execute("""
                    SELECT portfolio_id, account_id, total_value, cash_balance,
                           unrealized_pnl, realized_pnl
                    FROM portfolios
                    WHERE portfolio_id = %s
                """, (portfolio_id,))
                
                portfolio_summary = dict(cursor.fetchone())
                
                return {
                    'portfolio_summary': portfolio_summary,
                    'positions': positions
                }
                
        except Exception as e:
            logger.error(f"Error getting portfolio data: {e}")
            return None
    
    def _calculate_initial_metrics(self, portfolio_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate initial portfolio risk metrics"""
        positions = portfolio_data['positions']
        total_value = sum(pos['market_value'] for pos in positions)
        
        # Calculate portfolio volatility (simplified)
        weights = [pos['market_value'] / total_value for pos in positions]
        volatilities = [pos.get('volatility', 0.20) for pos in positions]
        
        # Simplified portfolio volatility calculation
        portfolio_vol = np.sqrt(sum(w * w * vol * vol for w, vol in zip(weights, volatilities)))
        
        # Calculate VaR (assuming normal distribution)
        var_95 = total_value * portfolio_vol * stats.norm.ppf(0.05) * np.sqrt(252/252)
        var_99 = total_value * portfolio_vol * stats.norm.ppf(0.01) * np.sqrt(252/252)
        
        # Expected shortfall (simplified)
        expected_shortfall = total_value * portfolio_vol * stats.norm.pdf(stats.norm.ppf(0.05)) / 0.05
        
        return {
            'value': total_value,
            'volatility': portfolio_vol,
            'var_95': abs(var_95),
            'var_99': abs(var_99),
            'expected_shortfall': abs(expected_shortfall)
        }
    
    def _apply_stress_scenario(self, portfolio_data: Dict[str, Any], 
                             scenario: StressScenario) -> Dict[str, Any]:
        """Apply stress scenario to portfolio"""
        stressed_portfolio = portfolio_data.copy()
        stressed_positions = []
        
        for position in portfolio_data['positions']:
            stressed_position = position.copy()
            
            # Apply equity price shock
            if RiskFactor.EQUITY_PRICE in scenario.risk_factors:
                price_shock = scenario.risk_factors[RiskFactor.EQUITY_PRICE]
                
                # Apply sector-specific adjustments
                sector_multiplier = self._get_sector_multiplier(position.get('sector'), scenario)
                adjusted_shock = price_shock * sector_multiplier
                
                # Apply shock to market value
                stressed_position['market_value'] = position['market_value'] * (1 + adjusted_shock)
                
                # Update unrealized P&L
                original_unrealized = position['market_value'] - position['cost_basis']
                new_unrealized = stressed_position['market_value'] - position['cost_basis']
                stressed_position['unrealized_pnl'] = new_unrealized
            
            # Apply volatility shock
            if RiskFactor.VOLATILITY in scenario.risk_factors:
                vol_shock = scenario.risk_factors[RiskFactor.VOLATILITY]
                original_vol = position.get('volatility', 0.20)
                stressed_position['volatility'] = original_vol * (1 + vol_shock)
            
            # Apply liquidity shock (affects bid-ask spread)
            if RiskFactor.LIQUIDITY in scenario.risk_factors:
                liquidity_shock = scenario.risk_factors[RiskFactor.LIQUIDITY]
                # Reduce market value by liquidity impact
                liquidity_impact = abs(liquidity_shock) * 0.01  # 1% impact per 100% liquidity reduction
                stressed_position['market_value'] *= (1 - liquidity_impact)
            
            stressed_positions.append(stressed_position)
        
        stressed_portfolio['positions'] = stressed_positions
        
        # Update portfolio summary
        new_total_value = sum(pos['market_value'] for pos in stressed_positions)
        stressed_portfolio['portfolio_summary']['total_value'] = new_total_value
        
        return stressed_portfolio
    
    def _get_sector_multiplier(self, sector: str, scenario: StressScenario) -> float:
        """Get sector-specific stress multiplier"""
        sector_multipliers = {
            ScenarioType.MARKET_CRASH: {
                'Technology': 1.2,
                'Finance': 1.5,
                'Energy': 1.3,
                'Healthcare': 0.8,
                'Utilities': 0.6
            },
            ScenarioType.INTEREST_RATE: {
                'Finance': 1.8,
                'Real Estate': 1.5,
                'Utilities': 1.3,
                'Technology': 0.9,
                'Healthcare': 0.8
            },
            ScenarioType.SECTOR_ROTATION: {
                'Technology': 2.0 if scenario.scenario_name == "Technology Sector Selloff" else 1.0
            }
        }
        
        multipliers = sector_multipliers.get(scenario.scenario_type, {})
        return multipliers.get(sector, 1.0)
    
    def _calculate_stressed_metrics(self, stressed_portfolio: Dict[str, Any]) -> Dict[str, float]:
        """Calculate metrics for stressed portfolio"""
        return self._calculate_initial_metrics(stressed_portfolio)
    
    def _calculate_impacts(self, original_portfolio: Dict[str, Any], 
                         stressed_portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate detailed impact analysis"""
        original_positions = {pos['symbol']: pos for pos in original_portfolio['positions']}
        stressed_positions = {pos['symbol']: pos for pos in stressed_portfolio['positions']}
        
        # Position impacts
        position_impacts = {}
        for symbol in original_positions:
            original_value = original_positions[symbol]['market_value']
            stressed_value = stressed_positions[symbol]['market_value']
            impact = (stressed_value - original_value) / original_value
            position_impacts[symbol] = impact
        
        # Sector impacts
        sector_impacts = {}
        for position in original_portfolio['positions']:
            sector = position.get('sector', 'Unknown')
            if sector not in sector_impacts:
                sector_impacts[sector] = {'original': 0, 'stressed': 0}
            
            sector_impacts[sector]['original'] += position['market_value']
            sector_impacts[sector]['stressed'] += stressed_positions[position['symbol']]['market_value']
        
        for sector in sector_impacts:
            original = sector_impacts[sector]['original']
            stressed = sector_impacts[sector]['stressed']
            sector_impacts[sector] = (stressed - original) / original if original > 0 else 0
        
        # Strategy impacts
        strategy_impacts = {}
        for position in original_portfolio['positions']:
            strategy = position.get('strategy_id', 'Unknown')
            if strategy not in strategy_impacts:
                strategy_impacts[strategy] = {'original': 0, 'stressed': 0}
            
            strategy_impacts[strategy]['original'] += position['market_value']
            strategy_impacts[strategy]['stressed'] += stressed_positions[position['symbol']]['market_value']
        
        for strategy in strategy_impacts:
            original = strategy_impacts[strategy]['original']
            stressed = strategy_impacts[strategy]['stressed']
            strategy_impacts[strategy] = (stressed - original) / original if original > 0 else 0
        
        # Concentration risk
        total_original = sum(pos['market_value'] for pos in original_portfolio['positions'])
        largest_position = max(pos['market_value'] for pos in original_portfolio['positions'])
        concentration_risk = largest_position / total_original if total_original > 0 else 0
        
        return {
            'positions': position_impacts,
            'sectors': sector_impacts,
            'strategies': strategy_impacts,
            'concentration_risk': concentration_risk
        }
    
    def _get_historical_returns(self, portfolio_data: Dict[str, Any], 
                              lookback_days: int = 252) -> np.ndarray:
        """Get historical returns for Monte Carlo simulation"""
        # Simplified: generate synthetic returns based on portfolio characteristics
        positions = portfolio_data['positions']
        total_value = sum(pos['market_value'] for pos in positions)
        
        # Calculate portfolio-level characteristics
        weights = [pos['market_value'] / total_value for pos in positions]
        volatilities = [pos.get('volatility', 0.20) for pos in positions]
        
        # Portfolio volatility
        portfolio_vol = np.sqrt(sum(w * w * vol * vol for w, vol in zip(weights, volatilities)))
        
        # Generate synthetic daily returns
        np.random.seed(42)  # For reproducibility
        daily_returns = np.random.normal(0.0008, portfolio_vol / np.sqrt(252), lookback_days)
        
        return daily_returns
    
    def _run_monte_carlo(self, historical_returns: np.ndarray, num_simulations: int,
                        time_horizon_days: int) -> np.ndarray:
        """Run Monte Carlo simulation"""
        # Calculate return statistics
        mean_return = np.mean(historical_returns)
        std_return = np.std(historical_returns)
        
        # Generate random returns for simulation
        np.random.seed(42)  # For reproducibility
        simulated_returns = np.random.normal(
            mean_return, std_return, 
            (num_simulations, time_horizon_days)
        )
        
        # Calculate cumulative returns for each simulation
        cumulative_returns = np.cumprod(1 + simulated_returns, axis=1)[:, -1] - 1
        
        return cumulative_returns
    
    def _calculate_monte_carlo_statistics(self, portfolio_id: str, simulated_returns: np.ndarray,
                                        num_simulations: int, time_horizon_days: int) -> MonteCarloResult:
        """Calculate Monte Carlo simulation statistics"""
        # Basic statistics
        mean_return = np.mean(simulated_returns)
        std_return = np.std(simulated_returns)
        skewness = stats.skew(simulated_returns)
        kurtosis = stats.kurtosis(simulated_returns)
        
        # Risk metrics
        var_95 = np.percentile(simulated_returns, 5)
        var_99 = np.percentile(simulated_returns, 1)
        
        # Expected shortfall
        expected_shortfall_95 = np.mean(simulated_returns[simulated_returns <= var_95])
        expected_shortfall_99 = np.mean(simulated_returns[simulated_returns <= var_99])
        
        # Percentiles
        percentiles = np.percentile(simulated_returns, [5, 10, 25, 75, 90, 95])
        
        # Probability metrics
        prob_loss = np.sum(simulated_returns < 0) / len(simulated_returns)
        prob_large_loss = np.sum(simulated_returns < -0.10) / len(simulated_returns)
        prob_extreme_loss = np.sum(simulated_returns < -0.20) / len(simulated_returns)
        
        return MonteCarloResult(
            simulation_id=str(uuid.uuid4()),
            portfolio_id=portfolio_id,
            num_simulations=num_simulations,
            time_horizon_days=time_horizon_days,
            confidence_level=0.95,
            
            mean_return=mean_return,
            std_return=std_return,
            skewness=skewness,
            kurtosis=kurtosis,
            
            var_95=abs(var_95),
            var_99=abs(var_99),
            expected_shortfall_95=abs(expected_shortfall_95),
            expected_shortfall_99=abs(expected_shortfall_99),
            
            percentile_5=percentiles[0],
            percentile_10=percentiles[1],
            percentile_25=percentiles[2],
            percentile_75=percentiles[3],
            percentile_90=percentiles[4],
            percentile_95=percentiles[5],
            
            prob_loss=prob_loss,
            prob_large_loss=prob_large_loss,
            prob_extreme_loss=prob_extreme_loss,
            
            simulation_timestamp=datetime.now()
        )
    
    def _store_stress_test_result(self, result: StressTestResult):
        """Store stress test result in database"""
        try:
            with self.db_connection.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO stress_test_results (
                        test_id, portfolio_id, scenario_id, scenario_name, test_timestamp,
                        initial_value, initial_var_95, initial_var_99, initial_expected_shortfall,
                        stressed_value, stressed_var_95, stressed_var_99, stressed_expected_shortfall,
                        absolute_loss, percentage_loss, var_increase, expected_shortfall_increase,
                        position_impacts, sector_impacts, strategy_impacts,
                        stress_ratio, tail_risk_ratio, concentration_risk
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                """, (
                    result.test_id, result.portfolio_id, result.scenario_id, result.scenario_name,
                    result.test_timestamp, result.initial_value, result.initial_var_95,
                    result.initial_var_99, result.initial_expected_shortfall, result.stressed_value,
                    result.stressed_var_95, result.stressed_var_99, result.stressed_expected_shortfall,
                    result.absolute_loss, result.percentage_loss, result.var_increase,
                    result.expected_shortfall_increase, json.dumps(result.position_impacts),
                    json.dumps(result.sector_impacts), json.dumps(result.strategy_impacts),
                    result.stress_ratio, result.tail_risk_ratio, result.concentration_risk
                ))
                self.db_connection.commit()
        except Exception as e:
            logger.error(f"Error storing stress test result: {e}")
            self.db_connection.rollback()
    
    def _store_monte_carlo_result(self, result: MonteCarloResult):
        """Store Monte Carlo result in database"""
        try:
            with self.db_connection.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO monte_carlo_results (
                        simulation_id, portfolio_id, num_simulations, time_horizon_days,
                        confidence_level, mean_return, std_return, skewness, kurtosis,
                        var_95, var_99, expected_shortfall_95, expected_shortfall_99,
                        percentile_5, percentile_10, percentile_25, percentile_75,
                        percentile_90, percentile_95, prob_loss, prob_large_loss,
                        prob_extreme_loss, simulation_timestamp
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                """, (
                    result.simulation_id, result.portfolio_id, result.num_simulations,
                    result.time_horizon_days, result.confidence_level, result.mean_return,
                    result.std_return, result.skewness, result.kurtosis, result.var_95,
                    result.var_99, result.expected_shortfall_95, result.expected_shortfall_99,
                    result.percentile_5, result.percentile_10, result.percentile_25,
                    result.percentile_75, result.percentile_90, result.percentile_95,
                    result.prob_loss, result.prob_large_loss, result.prob_extreme_loss,
                    result.simulation_timestamp
                ))
                self.db_connection.commit()
        except Exception as e:
            logger.error(f"Error storing Monte Carlo result: {e}")
            self.db_connection.rollback()

class StressTestManager:
    """Main stress testing management system"""
    
    def __init__(self, db_connection, redis_client):
        self.db_connection = db_connection
        self.redis_client = redis_client
        self.scenario_generator = ScenarioGenerator()
        self.stress_tester = PortfolioStressTester(db_connection, redis_client)
    
    def run_comprehensive_stress_test(self, portfolio_id: str) -> Dict[str, Any]:
        """Run comprehensive stress test suite"""
        try:
            results = {}
            
            # Generate standard scenarios
            scenarios = []
            scenarios.extend(self.scenario_generator.generate_market_crash_scenarios())
            scenarios.extend(self.scenario_generator.generate_volatility_spike_scenarios())
            scenarios.extend(self.scenario_generator.generate_interest_rate_scenarios())
            scenarios.extend(self.scenario_generator.generate_sector_rotation_scenarios())
            
            # Run stress tests in parallel
            with ThreadPoolExecutor(max_workers=4) as executor:
                future_to_scenario = {
                    executor.submit(self.stress_tester.run_stress_test, portfolio_id, scenario): scenario
                    for scenario in scenarios
                }
                
                stress_results = []
                for future in concurrent.futures.as_completed(future_to_scenario):
                    try:
                        result = future.result()
                        stress_results.append(result)
                    except Exception as e:
                        logger.error(f"Stress test failed: {e}")
            
            results['stress_tests'] = stress_results
            
            # Run Monte Carlo simulation
            try:
                monte_carlo_result = self.stress_tester.run_monte_carlo_simulation(portfolio_id)
                results['monte_carlo'] = monte_carlo_result
            except Exception as e:
                logger.error(f"Monte Carlo simulation failed: {e}")
                results['monte_carlo'] = None
            
            # Calculate summary statistics
            results['summary'] = self._calculate_stress_test_summary(stress_results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error running comprehensive stress test: {e}")
            raise
    
    def _calculate_stress_test_summary(self, stress_results: List[StressTestResult]) -> Dict[str, Any]:
        """Calculate summary statistics from stress test results"""
        if not stress_results:
            return {}
        
        # Calculate worst-case scenarios
        worst_absolute_loss = min(result.absolute_loss for result in stress_results)
        worst_percentage_loss = min(result.percentage_loss for result in stress_results)
        
        # Find most impactful scenario
        worst_scenario = min(stress_results, key=lambda x: x.percentage_loss)
        
        # Calculate average impacts
        avg_percentage_loss = np.mean([result.percentage_loss for result in stress_results])
        avg_var_increase = np.mean([result.var_increase for result in stress_results])
        
        # Calculate stress test coverage
        scenarios_tested = len(stress_results)
        
        return {
            'scenarios_tested': scenarios_tested,
            'worst_absolute_loss': worst_absolute_loss,
            'worst_percentage_loss': worst_percentage_loss,
            'worst_scenario_name': worst_scenario.scenario_name,
            'average_percentage_loss': avg_percentage_loss,
            'average_var_increase': avg_var_increase,
            'max_stress_ratio': max(result.stress_ratio for result in stress_results),
            'max_tail_risk_ratio': max(result.tail_risk_ratio for result in stress_results)
        }
    
    def get_stress_test_history(self, portfolio_id: str, days: int = 30) -> List[StressTestResult]:
        """Get stress test history for a portfolio"""
        try:
            with self.db_connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT test_id, portfolio_id, scenario_id, scenario_name, test_timestamp,
                           initial_value, stressed_value, absolute_loss, percentage_loss,
                           stress_ratio, tail_risk_ratio
                    FROM stress_test_results
                    WHERE portfolio_id = %s 
                    AND test_timestamp >= %s
                    ORDER BY test_timestamp DESC
                """, (portfolio_id, datetime.now() - timedelta(days=days)))
                
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Error getting stress test history: {e}")
            return []
    
    def create_custom_scenario(self, scenario_config: Dict[str, Any]) -> StressScenario:
        """Create and store custom stress scenario"""
        scenario = self.scenario_generator.generate_custom_scenario(scenario_config)
        
        # Store scenario in database
        try:
            with self.db_connection.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO stress_scenarios (
                        scenario_id, scenario_name, scenario_type, stress_level,
                        description, risk_factors, duration_days, probability,
                        created_at, is_active
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                """, (
                    scenario.scenario_id, scenario.scenario_name, scenario.scenario_type.value,
                    scenario.stress_level.value, scenario.description,
                    json.dumps({k.value: v for k, v in scenario.risk_factors.items()}),
                    scenario.duration_days, scenario.probability, scenario.created_at,
                    scenario.is_active
                ))
                self.db_connection.commit()
        except Exception as e:
            logger.error(f"Error storing custom scenario: {e}")
            self.db_connection.rollback()
            raise
        
        return scenario

