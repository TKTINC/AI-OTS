"""
Portfolio Optimization Engine
Advanced portfolio optimization with diversification, correlation management, and risk budgeting
"""

import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import scipy.optimize as opt
from scipy import linalg
import warnings

logger = logging.getLogger(__name__)

class OptimizationObjective(Enum):
    """Portfolio optimization objective"""
    MAX_SHARPE = "max_sharpe"
    MIN_VARIANCE = "min_variance"
    MAX_RETURN = "max_return"
    RISK_PARITY = "risk_parity"
    MAX_DIVERSIFICATION = "max_diversification"
    BLACK_LITTERMAN = "black_litterman"

class ConstraintType(Enum):
    """Portfolio constraint types"""
    WEIGHT_BOUNDS = "weight_bounds"
    SECTOR_LIMITS = "sector_limits"
    TURNOVER_LIMITS = "turnover_limits"
    TRACKING_ERROR = "tracking_error"
    CONCENTRATION = "concentration"

@dataclass
class AssetData:
    """Asset data for optimization"""
    symbol: str
    expected_return: float
    volatility: float
    beta: float
    sector: str
    market_cap: float
    
    # Current position
    current_weight: float = 0.0
    current_value: float = 0.0
    
    # Constraints
    min_weight: float = 0.0
    max_weight: float = 1.0
    
    # Strategy attribution
    strategy_id: Optional[str] = None
    signal_confidence: float = 0.0

@dataclass
class OptimizationConstraints:
    """Portfolio optimization constraints"""
    # Weight constraints
    min_weight: float = 0.0
    max_weight: float = 0.15  # 15% max per asset
    
    # Sector constraints
    max_sector_weight: float = 0.30  # 30% max per sector
    min_sector_weight: float = 0.0
    
    # Risk constraints
    max_portfolio_volatility: float = 0.20  # 20% max volatility
    max_tracking_error: float = 0.05  # 5% max tracking error
    target_beta: float = 1.0
    beta_tolerance: float = 0.2
    
    # Diversification constraints
    max_concentration: float = 0.50  # 50% max in top 5 holdings
    min_assets: int = 5
    max_assets: int = 20
    
    # Turnover constraints
    max_turnover: float = 0.20  # 20% max turnover
    transaction_cost: float = 0.001  # 0.1% transaction cost

@dataclass
class OptimizationResult:
    """Portfolio optimization result"""
    objective_value: float
    optimal_weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    
    # Risk metrics
    portfolio_beta: float
    max_drawdown: float
    var_95: float
    
    # Diversification metrics
    concentration_ratio: float
    effective_assets: float
    sector_diversification: float
    
    # Constraint satisfaction
    constraints_satisfied: bool
    constraint_violations: List[str]
    
    # Optimization details
    optimization_method: OptimizationObjective
    iterations: int
    convergence_status: str
    
    # Rebalancing
    trades_required: Dict[str, float]  # symbol -> trade amount
    total_turnover: float
    estimated_costs: float

class PortfolioOptimizer:
    """
    Advanced portfolio optimization engine
    Implements multiple optimization objectives with comprehensive constraints
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        
        # Risk parameters
        self.risk_free_rate = self.config.get('risk_free_rate', 0.02)
        self.lookback_period = self.config.get('lookback_period', 252)  # 1 year
        self.min_correlation_period = self.config.get('min_correlation_period', 60)
        
        # Optimization parameters
        self.max_iterations = self.config.get('max_iterations', 1000)
        self.tolerance = self.config.get('tolerance', 1e-8)
        self.regularization = self.config.get('regularization', 1e-5)
        
        # Data storage
        self.correlation_matrix: Optional[np.ndarray] = None
        self.covariance_matrix: Optional[np.ndarray] = None
        self.asset_returns: Optional[pd.DataFrame] = None
        
        logger.info("Portfolio Optimizer initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for portfolio optimization"""
        return {
            'risk_free_rate': 0.02,
            'lookback_period': 252,
            'min_correlation_period': 60,
            'max_iterations': 1000,
            'tolerance': 1e-8,
            'regularization': 1e-5,
            'shrinkage_factor': 0.1,  # For covariance shrinkage
            'black_litterman_tau': 0.025,
            'confidence_scaling': 2.0
        }
    
    def optimize_portfolio(self, assets: List[AssetData], 
                         objective: OptimizationObjective = OptimizationObjective.MAX_SHARPE,
                         constraints: OptimizationConstraints = None,
                         benchmark_weights: Dict[str, float] = None) -> OptimizationResult:
        """
        Optimize portfolio allocation
        
        Args:
            assets: List of assets to optimize
            objective: Optimization objective
            constraints: Portfolio constraints
            benchmark_weights: Benchmark weights for tracking error
            
        Returns:
            Optimization result
        """
        try:
            if not assets:
                raise ValueError("No assets provided for optimization")
            
            # Set default constraints
            if constraints is None:
                constraints = OptimizationConstraints()
            
            # Prepare data
            symbols = [asset.symbol for asset in assets]
            expected_returns = np.array([asset.expected_return for asset in assets])
            volatilities = np.array([asset.volatility for asset in assets])
            current_weights = np.array([asset.current_weight for asset in assets])
            
            # Build covariance matrix
            cov_matrix = self._build_covariance_matrix(assets)
            
            # Set up optimization problem
            n_assets = len(assets)
            
            # Initial guess (equal weights or current weights)
            if np.sum(current_weights) > 0:
                x0 = current_weights / np.sum(current_weights)
            else:
                x0 = np.ones(n_assets) / n_assets
            
            # Define bounds
            bounds = [(max(asset.min_weight, constraints.min_weight), 
                      min(asset.max_weight, constraints.max_weight)) 
                     for asset in assets]
            
            # Define constraints
            constraint_list = self._build_constraints(assets, constraints, benchmark_weights)
            
            # Define objective function
            objective_func = self._get_objective_function(objective, expected_returns, cov_matrix, constraints)
            
            # Solve optimization
            result = opt.minimize(
                objective_func,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraint_list,
                options={'maxiter': self.max_iterations, 'ftol': self.tolerance}
            )
            
            if not result.success:
                logger.warning(f"Optimization did not converge: {result.message}")
            
            # Process results
            optimal_weights = result.x
            optimal_weights = optimal_weights / np.sum(optimal_weights)  # Normalize
            
            # Calculate portfolio metrics
            portfolio_return = np.dot(optimal_weights, expected_returns)
            portfolio_variance = np.dot(optimal_weights, np.dot(cov_matrix, optimal_weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
            
            # Calculate additional metrics
            portfolio_beta = self._calculate_portfolio_beta(optimal_weights, assets)
            var_95 = self._calculate_var(optimal_weights, expected_returns, cov_matrix)
            concentration_ratio = self._calculate_concentration(optimal_weights)
            effective_assets = self._calculate_effective_assets(optimal_weights)
            sector_diversification = self._calculate_sector_diversification(optimal_weights, assets)
            
            # Check constraint satisfaction
            constraints_satisfied, violations = self._check_constraints(optimal_weights, assets, constraints)
            
            # Calculate trades and turnover
            trades = {}
            total_turnover = 0
            for i, asset in enumerate(assets):
                trade_amount = optimal_weights[i] - asset.current_weight
                if abs(trade_amount) > 1e-6:  # Minimum trade threshold
                    trades[asset.symbol] = trade_amount
                    total_turnover += abs(trade_amount)
            
            estimated_costs = total_turnover * constraints.transaction_cost
            
            # Create result
            optimization_result = OptimizationResult(
                objective_value=-result.fun if objective in [OptimizationObjective.MAX_SHARPE, OptimizationObjective.MAX_RETURN] else result.fun,
                optimal_weights={asset.symbol: optimal_weights[i] for i, asset in enumerate(assets)},
                expected_return=portfolio_return,
                expected_volatility=portfolio_volatility,
                sharpe_ratio=sharpe_ratio,
                portfolio_beta=portfolio_beta,
                max_drawdown=0.0,  # Would need historical simulation
                var_95=var_95,
                concentration_ratio=concentration_ratio,
                effective_assets=effective_assets,
                sector_diversification=sector_diversification,
                constraints_satisfied=constraints_satisfied,
                constraint_violations=violations,
                optimization_method=objective,
                iterations=result.nit,
                convergence_status="converged" if result.success else "failed",
                trades_required=trades,
                total_turnover=total_turnover,
                estimated_costs=estimated_costs
            )
            
            return optimization_result
            
        except Exception as e:
            logger.error(f"Error in portfolio optimization: {e}")
            # Return empty result
            return OptimizationResult(
                objective_value=0.0,
                optimal_weights={},
                expected_return=0.0,
                expected_volatility=0.0,
                sharpe_ratio=0.0,
                portfolio_beta=1.0,
                max_drawdown=0.0,
                var_95=0.0,
                concentration_ratio=0.0,
                effective_assets=0.0,
                sector_diversification=0.0,
                constraints_satisfied=False,
                constraint_violations=[f"Optimization error: {str(e)}"],
                optimization_method=objective,
                iterations=0,
                convergence_status="error",
                trades_required={},
                total_turnover=0.0,
                estimated_costs=0.0
            )
    
    def _build_covariance_matrix(self, assets: List[AssetData]) -> np.ndarray:
        """Build covariance matrix from asset data"""
        try:
            n_assets = len(assets)
            
            # If we have historical data, use it
            if self.covariance_matrix is not None and self.covariance_matrix.shape[0] == n_assets:
                return self.covariance_matrix
            
            # Otherwise, build from volatilities and correlations
            volatilities = np.array([asset.volatility for asset in assets])
            
            # Create correlation matrix (simplified approach)
            correlation_matrix = self._estimate_correlation_matrix(assets)
            
            # Convert to covariance matrix
            cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix
            
            # Add regularization to ensure positive definiteness
            cov_matrix += np.eye(n_assets) * self.regularization
            
            return cov_matrix
            
        except Exception as e:
            logger.error(f"Error building covariance matrix: {e}")
            # Return identity matrix as fallback
            n_assets = len(assets)
            volatilities = np.array([asset.volatility for asset in assets])
            return np.diag(volatilities ** 2)
    
    def _estimate_correlation_matrix(self, assets: List[AssetData]) -> np.ndarray:
        """Estimate correlation matrix between assets"""
        try:
            n_assets = len(assets)
            
            # If we have stored correlation matrix, use it
            if self.correlation_matrix is not None and self.correlation_matrix.shape[0] == n_assets:
                return self.correlation_matrix
            
            # Create correlation matrix based on sector and market cap similarity
            correlation_matrix = np.eye(n_assets)
            
            for i in range(n_assets):
                for j in range(i + 1, n_assets):
                    asset_i = assets[i]
                    asset_j = assets[j]
                    
                    # Base correlation
                    correlation = 0.1  # Low base correlation
                    
                    # Same sector increases correlation
                    if asset_i.sector == asset_j.sector:
                        correlation += 0.3
                    
                    # Similar market cap increases correlation
                    if asset_i.market_cap and asset_j.market_cap:
                        size_ratio = min(asset_i.market_cap, asset_j.market_cap) / max(asset_i.market_cap, asset_j.market_cap)
                        correlation += 0.2 * size_ratio
                    
                    # Beta similarity
                    beta_similarity = 1 - abs(asset_i.beta - asset_j.beta) / 2
                    correlation += 0.1 * beta_similarity
                    
                    # Cap correlation
                    correlation = min(correlation, 0.8)
                    
                    correlation_matrix[i, j] = correlation
                    correlation_matrix[j, i] = correlation
            
            # Ensure positive definiteness
            eigenvals, eigenvecs = linalg.eigh(correlation_matrix)
            eigenvals = np.maximum(eigenvals, 0.01)  # Minimum eigenvalue
            correlation_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
            
            # Normalize diagonal to 1
            diag_sqrt = np.sqrt(np.diag(correlation_matrix))
            correlation_matrix = correlation_matrix / np.outer(diag_sqrt, diag_sqrt)
            
            return correlation_matrix
            
        except Exception as e:
            logger.error(f"Error estimating correlation matrix: {e}")
            return np.eye(len(assets))
    
    def _build_constraints(self, assets: List[AssetData], constraints: OptimizationConstraints,
                          benchmark_weights: Dict[str, float] = None) -> List[Dict]:
        """Build optimization constraints"""
        constraint_list = []
        
        try:
            n_assets = len(assets)
            
            # Sum of weights = 1
            constraint_list.append({
                'type': 'eq',
                'fun': lambda x: np.sum(x) - 1.0
            })
            
            # Sector constraints
            sectors = list(set(asset.sector for asset in assets))
            for sector in sectors:
                sector_indices = [i for i, asset in enumerate(assets) if asset.sector == sector]
                
                if sector_indices:
                    # Maximum sector weight
                    constraint_list.append({
                        'type': 'ineq',
                        'fun': lambda x, indices=sector_indices: constraints.max_sector_weight - np.sum(x[indices])
                    })
                    
                    # Minimum sector weight (if any assets in sector)
                    if constraints.min_sector_weight > 0:
                        constraint_list.append({
                            'type': 'ineq',
                            'fun': lambda x, indices=sector_indices: np.sum(x[indices]) - constraints.min_sector_weight
                        })
            
            # Portfolio volatility constraint
            if hasattr(self, 'covariance_matrix') and self.covariance_matrix is not None:
                cov_matrix = self.covariance_matrix
            else:
                cov_matrix = self._build_covariance_matrix(assets)
            
            constraint_list.append({
                'type': 'ineq',
                'fun': lambda x: constraints.max_portfolio_volatility**2 - np.dot(x, np.dot(cov_matrix, x))
            })
            
            # Beta constraint
            if constraints.target_beta is not None:
                betas = np.array([asset.beta for asset in assets])
                
                # Beta upper bound
                constraint_list.append({
                    'type': 'ineq',
                    'fun': lambda x: (constraints.target_beta + constraints.beta_tolerance) - np.dot(x, betas)
                })
                
                # Beta lower bound
                constraint_list.append({
                    'type': 'ineq',
                    'fun': lambda x: np.dot(x, betas) - (constraints.target_beta - constraints.beta_tolerance)
                })
            
            # Turnover constraint
            if constraints.max_turnover is not None:
                current_weights = np.array([asset.current_weight for asset in assets])
                constraint_list.append({
                    'type': 'ineq',
                    'fun': lambda x: constraints.max_turnover - np.sum(np.abs(x - current_weights))
                })
            
            # Tracking error constraint (if benchmark provided)
            if benchmark_weights and constraints.max_tracking_error is not None:
                benchmark_array = np.array([benchmark_weights.get(asset.symbol, 0) for asset in assets])
                constraint_list.append({
                    'type': 'ineq',
                    'fun': lambda x: constraints.max_tracking_error**2 - np.dot(x - benchmark_array, np.dot(cov_matrix, x - benchmark_array))
                })
            
            return constraint_list
            
        except Exception as e:
            logger.error(f"Error building constraints: {e}")
            # Return basic constraint (sum = 1)
            return [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
    
    def _get_objective_function(self, objective: OptimizationObjective, 
                               expected_returns: np.ndarray, cov_matrix: np.ndarray,
                               constraints: OptimizationConstraints):
        """Get objective function for optimization"""
        
        if objective == OptimizationObjective.MAX_SHARPE:
            def sharpe_objective(weights):
                portfolio_return = np.dot(weights, expected_returns)
                portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
                portfolio_volatility = np.sqrt(portfolio_variance)
                if portfolio_volatility == 0:
                    return -np.inf
                sharpe = (portfolio_return - self.risk_free_rate) / portfolio_volatility
                return -sharpe  # Negative because we minimize
            return sharpe_objective
        
        elif objective == OptimizationObjective.MIN_VARIANCE:
            def variance_objective(weights):
                return np.dot(weights, np.dot(cov_matrix, weights))
            return variance_objective
        
        elif objective == OptimizationObjective.MAX_RETURN:
            def return_objective(weights):
                return -np.dot(weights, expected_returns)  # Negative because we minimize
            return return_objective
        
        elif objective == OptimizationObjective.RISK_PARITY:
            def risk_parity_objective(weights):
                portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
                marginal_contrib = np.dot(cov_matrix, weights)
                contrib = weights * marginal_contrib / portfolio_variance if portfolio_variance > 0 else weights
                target_contrib = 1.0 / len(weights)
                return np.sum((contrib - target_contrib) ** 2)
            return risk_parity_objective
        
        elif objective == OptimizationObjective.MAX_DIVERSIFICATION:
            def diversification_objective(weights):
                portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
                portfolio_volatility = np.sqrt(portfolio_variance)
                weighted_avg_volatility = np.dot(weights, np.sqrt(np.diag(cov_matrix)))
                if portfolio_volatility == 0:
                    return 0
                diversification_ratio = weighted_avg_volatility / portfolio_volatility
                return -diversification_ratio  # Negative because we minimize
            return diversification_objective
        
        else:
            # Default to max Sharpe
            return self._get_objective_function(OptimizationObjective.MAX_SHARPE, expected_returns, cov_matrix, constraints)
    
    def _calculate_portfolio_beta(self, weights: np.ndarray, assets: List[AssetData]) -> float:
        """Calculate portfolio beta"""
        try:
            betas = np.array([asset.beta for asset in assets])
            return np.dot(weights, betas)
        except Exception as e:
            logger.error(f"Error calculating portfolio beta: {e}")
            return 1.0
    
    def _calculate_var(self, weights: np.ndarray, expected_returns: np.ndarray, 
                      cov_matrix: np.ndarray, confidence: float = 0.95) -> float:
        """Calculate Value at Risk"""
        try:
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            
            # Assuming normal distribution
            from scipy.stats import norm
            var = norm.ppf(1 - confidence) * portfolio_volatility + portfolio_return
            return -var  # VaR is typically reported as positive loss
            
        except Exception as e:
            logger.error(f"Error calculating VaR: {e}")
            return 0.0
    
    def _calculate_concentration(self, weights: np.ndarray) -> float:
        """Calculate concentration ratio (sum of top 5 holdings)"""
        try:
            sorted_weights = np.sort(weights)[::-1]  # Descending order
            top_5 = sorted_weights[:min(5, len(weights))]
            return np.sum(top_5)
        except Exception as e:
            logger.error(f"Error calculating concentration: {e}")
            return 0.0
    
    def _calculate_effective_assets(self, weights: np.ndarray) -> float:
        """Calculate effective number of assets (inverse of Herfindahl index)"""
        try:
            herfindahl = np.sum(weights ** 2)
            return 1.0 / herfindahl if herfindahl > 0 else 0.0
        except Exception as e:
            logger.error(f"Error calculating effective assets: {e}")
            return 0.0
    
    def _calculate_sector_diversification(self, weights: np.ndarray, assets: List[AssetData]) -> float:
        """Calculate sector diversification score"""
        try:
            sectors = {}
            for i, asset in enumerate(assets):
                if asset.sector not in sectors:
                    sectors[asset.sector] = 0
                sectors[asset.sector] += weights[i]
            
            sector_weights = np.array(list(sectors.values()))
            sector_herfindahl = np.sum(sector_weights ** 2)
            return 1.0 / sector_herfindahl if sector_herfindahl > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating sector diversification: {e}")
            return 0.0
    
    def _check_constraints(self, weights: np.ndarray, assets: List[AssetData], 
                          constraints: OptimizationConstraints) -> Tuple[bool, List[str]]:
        """Check if solution satisfies all constraints"""
        violations = []
        
        try:
            # Weight bounds
            for i, (weight, asset) in enumerate(zip(weights, assets)):
                if weight < asset.min_weight - 1e-6:
                    violations.append(f"{asset.symbol} weight {weight:.3f} below minimum {asset.min_weight:.3f}")
                if weight > asset.max_weight + 1e-6:
                    violations.append(f"{asset.symbol} weight {weight:.3f} above maximum {asset.max_weight:.3f}")
            
            # Sector constraints
            sectors = {}
            for i, asset in enumerate(assets):
                if asset.sector not in sectors:
                    sectors[asset.sector] = 0
                sectors[asset.sector] += weights[i]
            
            for sector, sector_weight in sectors.items():
                if sector_weight > constraints.max_sector_weight + 1e-6:
                    violations.append(f"Sector {sector} weight {sector_weight:.3f} above maximum {constraints.max_sector_weight:.3f}")
            
            # Portfolio volatility
            cov_matrix = self._build_covariance_matrix(assets)
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            
            if portfolio_volatility > constraints.max_portfolio_volatility + 1e-6:
                violations.append(f"Portfolio volatility {portfolio_volatility:.3f} above maximum {constraints.max_portfolio_volatility:.3f}")
            
            # Beta constraint
            if constraints.target_beta is not None:
                portfolio_beta = self._calculate_portfolio_beta(weights, assets)
                if abs(portfolio_beta - constraints.target_beta) > constraints.beta_tolerance + 1e-6:
                    violations.append(f"Portfolio beta {portfolio_beta:.3f} outside target {constraints.target_beta:.3f} ± {constraints.beta_tolerance:.3f}")
            
            return len(violations) == 0, violations
            
        except Exception as e:
            logger.error(f"Error checking constraints: {e}")
            return False, [f"Constraint check error: {str(e)}"]
    
    def rebalance_portfolio(self, current_assets: List[AssetData], target_weights: Dict[str, float],
                           constraints: OptimizationConstraints = None) -> Dict[str, Any]:
        """
        Calculate rebalancing trades to achieve target weights
        
        Args:
            current_assets: Current portfolio assets
            target_weights: Target weight allocation
            constraints: Rebalancing constraints
            
        Returns:
            Rebalancing plan with trades and costs
        """
        try:
            if constraints is None:
                constraints = OptimizationConstraints()
            
            rebalancing_plan = {
                'trades': {},
                'total_turnover': 0.0,
                'estimated_costs': 0.0,
                'cash_required': 0.0,
                'cash_generated': 0.0
            }
            
            total_value = sum(asset.current_value for asset in current_assets)
            
            for asset in current_assets:
                current_weight = asset.current_weight
                target_weight = target_weights.get(asset.symbol, 0.0)
                
                weight_diff = target_weight - current_weight
                
                if abs(weight_diff) > 1e-6:  # Minimum trade threshold
                    trade_value = weight_diff * total_value
                    
                    rebalancing_plan['trades'][asset.symbol] = {
                        'current_weight': current_weight,
                        'target_weight': target_weight,
                        'weight_change': weight_diff,
                        'trade_value': trade_value,
                        'trade_shares': int(trade_value / self._get_current_price(asset.symbol)) if trade_value != 0 else 0
                    }
                    
                    rebalancing_plan['total_turnover'] += abs(weight_diff)
                    
                    if trade_value > 0:
                        rebalancing_plan['cash_required'] += trade_value
                    else:
                        rebalancing_plan['cash_generated'] += abs(trade_value)
            
            # Calculate costs
            rebalancing_plan['estimated_costs'] = rebalancing_plan['total_turnover'] * total_value * constraints.transaction_cost
            
            # Check turnover constraint
            if rebalancing_plan['total_turnover'] > constraints.max_turnover:
                rebalancing_plan['warning'] = f"Turnover {rebalancing_plan['total_turnover']:.2%} exceeds limit {constraints.max_turnover:.2%}"
            
            return rebalancing_plan
            
        except Exception as e:
            logger.error(f"Error calculating rebalancing plan: {e}")
            return {'error': str(e)}
    
    def _get_current_price(self, symbol: str) -> float:
        """Get current price for symbol (mock implementation)"""
        # In production, this would fetch from market data service
        mock_prices = {
            "AAPL": 150.0, "MSFT": 300.0, "GOOGL": 2500.0, "AMZN": 3000.0,
            "TSLA": 200.0, "NVDA": 400.0, "META": 250.0, "SPY": 400.0, "QQQ": 350.0
        }
        return mock_prices.get(symbol, 100.0)

# Utility functions
def create_asset_data(symbol: str, expected_return: float, volatility: float, 
                     beta: float = 1.0, sector: str = "technology", **kwargs) -> AssetData:
    """Create AssetData with defaults"""
    return AssetData(
        symbol=symbol,
        expected_return=expected_return,
        volatility=volatility,
        beta=beta,
        sector=sector,
        market_cap=kwargs.get('market_cap', 1e9),
        current_weight=kwargs.get('current_weight', 0.0),
        current_value=kwargs.get('current_value', 0.0),
        min_weight=kwargs.get('min_weight', 0.0),
        max_weight=kwargs.get('max_weight', 0.15),
        strategy_id=kwargs.get('strategy_id'),
        signal_confidence=kwargs.get('signal_confidence', 0.0)
    )

def create_optimization_constraints(**kwargs) -> OptimizationConstraints:
    """Create OptimizationConstraints with custom values"""
    return OptimizationConstraints(
        min_weight=kwargs.get('min_weight', 0.0),
        max_weight=kwargs.get('max_weight', 0.15),
        max_sector_weight=kwargs.get('max_sector_weight', 0.30),
        min_sector_weight=kwargs.get('min_sector_weight', 0.0),
        max_portfolio_volatility=kwargs.get('max_portfolio_volatility', 0.20),
        max_tracking_error=kwargs.get('max_tracking_error', 0.05),
        target_beta=kwargs.get('target_beta', 1.0),
        beta_tolerance=kwargs.get('beta_tolerance', 0.2),
        max_concentration=kwargs.get('max_concentration', 0.50),
        min_assets=kwargs.get('min_assets', 5),
        max_assets=kwargs.get('max_assets', 20),
        max_turnover=kwargs.get('max_turnover', 0.20),
        transaction_cost=kwargs.get('transaction_cost', 0.001)
    )

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create optimizer
    optimizer = PortfolioOptimizer()
    
    # Create test assets
    assets = [
        create_asset_data("AAPL", 0.08, 0.25, 1.2, "technology", current_weight=0.1),
        create_asset_data("MSFT", 0.07, 0.22, 1.1, "technology", current_weight=0.15),
        create_asset_data("GOOGL", 0.09, 0.28, 1.3, "technology", current_weight=0.05),
        create_asset_data("AMZN", 0.10, 0.30, 1.4, "consumer", current_weight=0.08),
        create_asset_data("TSLA", 0.12, 0.45, 1.8, "automotive", current_weight=0.03),
        create_asset_data("SPY", 0.06, 0.15, 1.0, "etf", current_weight=0.59)
    ]
    
    # Create constraints
    constraints = create_optimization_constraints(
        max_weight=0.20,
        max_sector_weight=0.40,
        max_portfolio_volatility=0.18
    )
    
    # Test different optimization objectives
    objectives = [
        OptimizationObjective.MAX_SHARPE,
        OptimizationObjective.MIN_VARIANCE,
        OptimizationObjective.RISK_PARITY
    ]
    
    print("Portfolio Optimization Results:")
    print("=" * 50)
    
    for objective in objectives:
        result = optimizer.optimize_portfolio(assets, objective, constraints)
        
        print(f"\n{objective.value.upper()}:")
        print(f"  Expected Return: {result.expected_return:.2%}")
        print(f"  Expected Volatility: {result.expected_volatility:.2%}")
        print(f"  Sharpe Ratio: {result.sharpe_ratio:.3f}")
        print(f"  Portfolio Beta: {result.portfolio_beta:.3f}")
        print(f"  Concentration: {result.concentration_ratio:.2%}")
        print(f"  Effective Assets: {result.effective_assets:.1f}")
        print(f"  Total Turnover: {result.total_turnover:.2%}")
        print(f"  Constraints Satisfied: {result.constraints_satisfied}")
        
        print("  Optimal Weights:")
        for symbol, weight in result.optimal_weights.items():
            if weight > 0.01:  # Only show significant weights
                print(f"    {symbol}: {weight:.2%}")
    
    print("\nPortfolio optimization example completed")

