"""
Portfolio Management Service
Core service for managing trading portfolios with real-time tracking and IBKR integration
"""

import os
import sys
import time
import logging
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json
import uuid

from flask import Flask, request, jsonify, g
from flask_cors import CORS
import sqlite3
import redis
import pandas as pd
import numpy as np

# Import IBKR components
from ..ibkr.ibkr_client import IBKRClient, AccountType, Position, AccountInfo, OrderInfo
from ..ibkr.account_manager import AccountManager, AccountProfile, TradingSession

logger = logging.getLogger(__name__)

class PortfolioStatus(Enum):
    """Portfolio status enumeration"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    LIQUIDATING = "liquidating"
    SUSPENDED = "suspended"

class PositionStatus(Enum):
    """Position status enumeration"""
    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"

@dataclass
class PortfolioPosition:
    """Enhanced portfolio position with additional metadata"""
    position_id: str
    symbol: str
    quantity: float
    average_cost: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float
    
    # Strategy attribution
    strategy_id: Optional[str] = None
    signal_id: Optional[str] = None
    entry_time: Optional[datetime] = None
    
    # Risk metrics
    position_risk: float = 0.0
    var_contribution: float = 0.0
    beta: float = 1.0
    
    # Status and metadata
    status: PositionStatus = PositionStatus.OPEN
    last_updated: datetime = None
    
    def __post_init__(self):
        if self.position_id is None:
            self.position_id = str(uuid.uuid4())
        if self.last_updated is None:
            self.last_updated = datetime.now(timezone.utc)

@dataclass
class Portfolio:
    """Portfolio data structure"""
    portfolio_id: str
    account_id: str
    portfolio_name: str
    
    # Portfolio metrics
    total_value: float = 0.0
    cash_balance: float = 0.0
    buying_power: float = 0.0
    total_pnl: float = 0.0
    day_pnl: float = 0.0
    
    # Risk metrics
    portfolio_var: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    beta: float = 1.0
    
    # Positions
    positions: Dict[str, PortfolioPosition] = None
    
    # Status and metadata
    status: PortfolioStatus = PortfolioStatus.ACTIVE
    created_at: datetime = None
    last_updated: datetime = None
    
    def __post_init__(self):
        if self.positions is None:
            self.positions = {}
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
        if self.last_updated is None:
            self.last_updated = datetime.now(timezone.utc)

class PortfolioManager:
    """
    Core portfolio management system
    Handles portfolio tracking, position management, and real-time updates
    """
    
    def __init__(self, db_path: str = "data/portfolio.db", redis_host: str = "localhost", redis_port: int = 6379):
        self.db_path = db_path
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        
        # Portfolio storage
        self.portfolios: Dict[str, Portfolio] = {}
        self.account_manager: Optional[AccountManager] = None
        self.ibkr_client: Optional[IBKRClient] = None
        
        # Real-time tracking
        self.update_thread: Optional[threading.Thread] = None
        self.is_running = False
        self.update_interval = 1.0  # seconds
        
        # Initialize database
        self._init_database()
        
        # Load existing portfolios
        self._load_portfolios()
        
        logger.info("Portfolio Manager initialized")
    
    def _init_database(self):
        """Initialize SQLite database for portfolio storage"""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Portfolios table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS portfolios (
                        portfolio_id TEXT PRIMARY KEY,
                        account_id TEXT NOT NULL,
                        portfolio_name TEXT NOT NULL,
                        total_value REAL DEFAULT 0.0,
                        cash_balance REAL DEFAULT 0.0,
                        buying_power REAL DEFAULT 0.0,
                        total_pnl REAL DEFAULT 0.0,
                        day_pnl REAL DEFAULT 0.0,
                        portfolio_var REAL DEFAULT 0.0,
                        max_drawdown REAL DEFAULT 0.0,
                        sharpe_ratio REAL DEFAULT 0.0,
                        beta REAL DEFAULT 1.0,
                        status TEXT DEFAULT 'active',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Positions table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS positions (
                        position_id TEXT PRIMARY KEY,
                        portfolio_id TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        quantity REAL NOT NULL,
                        average_cost REAL NOT NULL,
                        current_price REAL NOT NULL,
                        market_value REAL NOT NULL,
                        unrealized_pnl REAL NOT NULL,
                        realized_pnl REAL DEFAULT 0.0,
                        strategy_id TEXT,
                        signal_id TEXT,
                        entry_time TIMESTAMP,
                        position_risk REAL DEFAULT 0.0,
                        var_contribution REAL DEFAULT 0.0,
                        beta REAL DEFAULT 1.0,
                        status TEXT DEFAULT 'open',
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (portfolio_id) REFERENCES portfolios (portfolio_id)
                    )
                """)
                
                # Portfolio history table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS portfolio_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        portfolio_id TEXT NOT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        total_value REAL,
                        total_pnl REAL,
                        day_pnl REAL,
                        portfolio_var REAL,
                        position_count INTEGER,
                        FOREIGN KEY (portfolio_id) REFERENCES portfolios (portfolio_id)
                    )
                """)
                
                conn.commit()
                
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    def _load_portfolios(self):
        """Load existing portfolios from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Load portfolios
                cursor.execute("SELECT * FROM portfolios")
                portfolio_rows = cursor.fetchall()
                
                for row in portfolio_rows:
                    portfolio = Portfolio(
                        portfolio_id=row[0],
                        account_id=row[1],
                        portfolio_name=row[2],
                        total_value=row[3],
                        cash_balance=row[4],
                        buying_power=row[5],
                        total_pnl=row[6],
                        day_pnl=row[7],
                        portfolio_var=row[8],
                        max_drawdown=row[9],
                        sharpe_ratio=row[10],
                        beta=row[11],
                        status=PortfolioStatus(row[12]),
                        created_at=datetime.fromisoformat(row[13]),
                        last_updated=datetime.fromisoformat(row[14])
                    )
                    
                    # Load positions for this portfolio
                    cursor.execute("SELECT * FROM positions WHERE portfolio_id = ?", (portfolio.portfolio_id,))
                    position_rows = cursor.fetchall()
                    
                    for pos_row in position_rows:
                        position = PortfolioPosition(
                            position_id=pos_row[0],
                            symbol=pos_row[2],
                            quantity=pos_row[3],
                            average_cost=pos_row[4],
                            current_price=pos_row[5],
                            market_value=pos_row[6],
                            unrealized_pnl=pos_row[7],
                            realized_pnl=pos_row[8],
                            strategy_id=pos_row[9],
                            signal_id=pos_row[10],
                            entry_time=datetime.fromisoformat(pos_row[11]) if pos_row[11] else None,
                            position_risk=pos_row[12],
                            var_contribution=pos_row[13],
                            beta=pos_row[14],
                            status=PositionStatus(pos_row[15]),
                            last_updated=datetime.fromisoformat(pos_row[16])
                        )
                        
                        portfolio.positions[position.symbol] = position
                    
                    self.portfolios[portfolio.portfolio_id] = portfolio
                
                logger.info(f"Loaded {len(self.portfolios)} portfolios from database")
                
        except Exception as e:
            logger.error(f"Error loading portfolios: {e}")
    
    def set_account_manager(self, account_manager: AccountManager):
        """Set account manager for IBKR integration"""
        self.account_manager = account_manager
        self.ibkr_client = account_manager.ibkr_client
        
        # Set up IBKR callbacks
        if self.ibkr_client:
            self.ibkr_client.add_position_callback(self._on_position_update)
            self.ibkr_client.add_account_callback(self._on_account_update)
            self.ibkr_client.add_order_callback(self._on_order_update)
    
    def create_portfolio(self, account_id: str, portfolio_name: str) -> str:
        """
        Create new portfolio
        
        Args:
            account_id: Account ID
            portfolio_name: Portfolio name
            
        Returns:
            Portfolio ID
        """
        try:
            portfolio_id = str(uuid.uuid4())
            
            portfolio = Portfolio(
                portfolio_id=portfolio_id,
                account_id=account_id,
                portfolio_name=portfolio_name
            )
            
            # Save to database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO portfolios (
                        portfolio_id, account_id, portfolio_name, status, created_at, last_updated
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    portfolio.portfolio_id,
                    portfolio.account_id,
                    portfolio.portfolio_name,
                    portfolio.status.value,
                    portfolio.created_at.isoformat(),
                    portfolio.last_updated.isoformat()
                ))
                conn.commit()
            
            # Store in memory
            self.portfolios[portfolio_id] = portfolio
            
            # Cache in Redis
            self._cache_portfolio(portfolio)
            
            logger.info(f"Created portfolio {portfolio_id} for account {account_id}")
            return portfolio_id
            
        except Exception as e:
            logger.error(f"Error creating portfolio: {e}")
            return ""
    
    def get_portfolio(self, portfolio_id: str) -> Optional[Portfolio]:
        """Get portfolio by ID"""
        return self.portfolios.get(portfolio_id)
    
    def get_portfolios_by_account(self, account_id: str) -> List[Portfolio]:
        """Get all portfolios for an account"""
        return [p for p in self.portfolios.values() if p.account_id == account_id]
    
    def update_position(self, portfolio_id: str, symbol: str, quantity: float, 
                       price: float, strategy_id: str = None, signal_id: str = None) -> bool:
        """
        Update position in portfolio
        
        Args:
            portfolio_id: Portfolio ID
            symbol: Trading symbol
            quantity: Position quantity (positive for long, negative for short)
            price: Current price
            strategy_id: Strategy that created this position
            signal_id: Signal that triggered this position
            
        Returns:
            True if update successful
        """
        try:
            portfolio = self.portfolios.get(portfolio_id)
            if not portfolio:
                logger.error(f"Portfolio {portfolio_id} not found")
                return False
            
            # Get or create position
            if symbol in portfolio.positions:
                position = portfolio.positions[symbol]
                
                # Update existing position
                old_quantity = position.quantity
                old_cost = position.average_cost
                
                if quantity == 0:
                    # Closing position
                    position.status = PositionStatus.CLOSED
                    position.realized_pnl += (price - position.average_cost) * old_quantity
                    position.quantity = 0
                    position.market_value = 0
                    position.unrealized_pnl = 0
                else:
                    # Update position
                    if (old_quantity > 0 and quantity > 0) or (old_quantity < 0 and quantity < 0):
                        # Same direction - update average cost
                        total_cost = (old_quantity * old_cost) + ((quantity - old_quantity) * price)
                        position.average_cost = total_cost / quantity
                    else:
                        # Direction change or partial close
                        if abs(quantity) < abs(old_quantity):
                            # Partial close
                            closed_quantity = old_quantity - quantity
                            position.realized_pnl += (price - position.average_cost) * closed_quantity
                        else:
                            # Direction change
                            position.realized_pnl += (price - position.average_cost) * old_quantity
                            position.average_cost = price
                    
                    position.quantity = quantity
                    position.current_price = price
                    position.market_value = quantity * price
                    position.unrealized_pnl = quantity * (price - position.average_cost)
                
            else:
                # Create new position
                position = PortfolioPosition(
                    symbol=symbol,
                    quantity=quantity,
                    average_cost=price,
                    current_price=price,
                    market_value=quantity * price,
                    unrealized_pnl=0.0,
                    strategy_id=strategy_id,
                    signal_id=signal_id,
                    entry_time=datetime.now(timezone.utc)
                )
                
                portfolio.positions[symbol] = position
            
            position.last_updated = datetime.now(timezone.utc)
            
            # Update portfolio metrics
            self._update_portfolio_metrics(portfolio)
            
            # Save to database
            self._save_position(portfolio_id, position)
            self._save_portfolio(portfolio)
            
            # Cache in Redis
            self._cache_portfolio(portfolio)
            
            logger.info(f"Updated position {symbol} in portfolio {portfolio_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating position: {e}")
            return False
    
    def get_portfolio_summary(self, portfolio_id: str) -> Dict[str, Any]:
        """
        Get portfolio summary with key metrics
        
        Args:
            portfolio_id: Portfolio ID
            
        Returns:
            Portfolio summary dictionary
        """
        try:
            portfolio = self.portfolios.get(portfolio_id)
            if not portfolio:
                return {}
            
            # Calculate additional metrics
            open_positions = [p for p in portfolio.positions.values() if p.status == PositionStatus.OPEN]
            total_positions = len(open_positions)
            
            # Sector/symbol diversification
            symbols = [p.symbol for p in open_positions]
            unique_symbols = len(set(symbols))
            
            # Risk metrics
            total_risk = sum(p.position_risk for p in open_positions)
            max_position_risk = max([p.position_risk for p in open_positions], default=0)
            
            return {
                "portfolio_id": portfolio.portfolio_id,
                "account_id": portfolio.account_id,
                "portfolio_name": portfolio.portfolio_name,
                "status": portfolio.status.value,
                
                # Value metrics
                "total_value": portfolio.total_value,
                "cash_balance": portfolio.cash_balance,
                "buying_power": portfolio.buying_power,
                "total_pnl": portfolio.total_pnl,
                "day_pnl": portfolio.day_pnl,
                
                # Risk metrics
                "portfolio_var": portfolio.portfolio_var,
                "max_drawdown": portfolio.max_drawdown,
                "sharpe_ratio": portfolio.sharpe_ratio,
                "beta": portfolio.beta,
                "total_risk": total_risk,
                "max_position_risk": max_position_risk,
                
                # Position metrics
                "total_positions": total_positions,
                "unique_symbols": unique_symbols,
                "positions": [asdict(p) for p in open_positions],
                
                # Timestamps
                "created_at": portfolio.created_at.isoformat(),
                "last_updated": portfolio.last_updated.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting portfolio summary: {e}")
            return {}
    
    def start_real_time_tracking(self):
        """Start real-time portfolio tracking"""
        try:
            if self.is_running:
                logger.warning("Real-time tracking already running")
                return
            
            self.is_running = True
            self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
            self.update_thread.start()
            
            logger.info("Started real-time portfolio tracking")
            
        except Exception as e:
            logger.error(f"Error starting real-time tracking: {e}")
    
    def stop_real_time_tracking(self):
        """Stop real-time portfolio tracking"""
        try:
            self.is_running = False
            
            if self.update_thread and self.update_thread.is_alive():
                self.update_thread.join(timeout=5)
            
            logger.info("Stopped real-time portfolio tracking")
            
        except Exception as e:
            logger.error(f"Error stopping real-time tracking: {e}")
    
    def _update_loop(self):
        """Main update loop for real-time tracking"""
        while self.is_running:
            try:
                # Update all portfolios
                for portfolio in self.portfolios.values():
                    if portfolio.status == PortfolioStatus.ACTIVE:
                        self._update_portfolio_real_time(portfolio)
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in update loop: {e}")
                time.sleep(self.update_interval)
    
    def _update_portfolio_real_time(self, portfolio: Portfolio):
        """Update portfolio with real-time data"""
        try:
            # Update position prices from market data
            for position in portfolio.positions.values():
                if position.status == PositionStatus.OPEN and self.ibkr_client:
                    market_data = self.ibkr_client.get_market_data(position.symbol)
                    if market_data and 'last' in market_data:
                        new_price = market_data['last']
                        if new_price != position.current_price:
                            position.current_price = new_price
                            position.market_value = position.quantity * new_price
                            position.unrealized_pnl = position.quantity * (new_price - position.average_cost)
                            position.last_updated = datetime.now(timezone.utc)
            
            # Update portfolio metrics
            self._update_portfolio_metrics(portfolio)
            
            # Cache updated portfolio
            self._cache_portfolio(portfolio)
            
            # Save to database periodically (every 10 updates)
            if int(time.time()) % 10 == 0:
                self._save_portfolio(portfolio)
                for position in portfolio.positions.values():
                    self._save_position(portfolio.portfolio_id, position)
            
        except Exception as e:
            logger.error(f"Error updating portfolio real-time: {e}")
    
    def _update_portfolio_metrics(self, portfolio: Portfolio):
        """Update portfolio-level metrics"""
        try:
            # Calculate total values
            portfolio.total_value = sum(p.market_value for p in portfolio.positions.values() if p.status == PositionStatus.OPEN)
            portfolio.total_pnl = sum(p.unrealized_pnl + p.realized_pnl for p in portfolio.positions.values())
            
            # Update account info if available
            if self.ibkr_client and self.ibkr_client.account_info:
                account_info = self.ibkr_client.account_info
                portfolio.cash_balance = account_info.total_cash
                portfolio.buying_power = account_info.buying_power
            
            # Calculate risk metrics (simplified)
            open_positions = [p for p in portfolio.positions.values() if p.status == PositionStatus.OPEN]
            if open_positions:
                position_values = [abs(p.market_value) for p in open_positions]
                portfolio.portfolio_var = np.std(position_values) * 1.65  # 95% VaR approximation
                
                # Calculate beta (simplified)
                betas = [p.beta for p in open_positions if p.beta]
                if betas:
                    weights = [abs(p.market_value) / portfolio.total_value for p in open_positions if p.beta]
                    portfolio.beta = np.average(betas, weights=weights)
            
            portfolio.last_updated = datetime.now(timezone.utc)
            
        except Exception as e:
            logger.error(f"Error updating portfolio metrics: {e}")
    
    def _save_portfolio(self, portfolio: Portfolio):
        """Save portfolio to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE portfolios SET
                        total_value = ?, cash_balance = ?, buying_power = ?,
                        total_pnl = ?, day_pnl = ?, portfolio_var = ?,
                        max_drawdown = ?, sharpe_ratio = ?, beta = ?,
                        status = ?, last_updated = ?
                    WHERE portfolio_id = ?
                """, (
                    portfolio.total_value, portfolio.cash_balance, portfolio.buying_power,
                    portfolio.total_pnl, portfolio.day_pnl, portfolio.portfolio_var,
                    portfolio.max_drawdown, portfolio.sharpe_ratio, portfolio.beta,
                    portfolio.status.value, portfolio.last_updated.isoformat(),
                    portfolio.portfolio_id
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error saving portfolio: {e}")
    
    def _save_position(self, portfolio_id: str, position: PortfolioPosition):
        """Save position to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if position exists
                cursor.execute("SELECT position_id FROM positions WHERE position_id = ?", (position.position_id,))
                exists = cursor.fetchone()
                
                if exists:
                    # Update existing position
                    cursor.execute("""
                        UPDATE positions SET
                            quantity = ?, average_cost = ?, current_price = ?,
                            market_value = ?, unrealized_pnl = ?, realized_pnl = ?,
                            position_risk = ?, var_contribution = ?, beta = ?,
                            status = ?, last_updated = ?
                        WHERE position_id = ?
                    """, (
                        position.quantity, position.average_cost, position.current_price,
                        position.market_value, position.unrealized_pnl, position.realized_pnl,
                        position.position_risk, position.var_contribution, position.beta,
                        position.status.value, position.last_updated.isoformat(),
                        position.position_id
                    ))
                else:
                    # Insert new position
                    cursor.execute("""
                        INSERT INTO positions (
                            position_id, portfolio_id, symbol, quantity, average_cost,
                            current_price, market_value, unrealized_pnl, realized_pnl,
                            strategy_id, signal_id, entry_time, position_risk,
                            var_contribution, beta, status, last_updated
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        position.position_id, portfolio_id, position.symbol,
                        position.quantity, position.average_cost, position.current_price,
                        position.market_value, position.unrealized_pnl, position.realized_pnl,
                        position.strategy_id, position.signal_id,
                        position.entry_time.isoformat() if position.entry_time else None,
                        position.position_risk, position.var_contribution, position.beta,
                        position.status.value, position.last_updated.isoformat()
                    ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error saving position: {e}")
    
    def _cache_portfolio(self, portfolio: Portfolio):
        """Cache portfolio in Redis"""
        try:
            portfolio_data = asdict(portfolio)
            
            # Convert datetime objects to ISO strings
            portfolio_data['created_at'] = portfolio.created_at.isoformat()
            portfolio_data['last_updated'] = portfolio.last_updated.isoformat()
            
            # Convert positions
            positions_data = {}
            for symbol, position in portfolio.positions.items():
                pos_data = asdict(position)
                pos_data['last_updated'] = position.last_updated.isoformat()
                if position.entry_time:
                    pos_data['entry_time'] = position.entry_time.isoformat()
                positions_data[symbol] = pos_data
            
            portfolio_data['positions'] = positions_data
            
            # Cache with 1 hour expiration
            self.redis_client.setex(
                f"portfolio:{portfolio.portfolio_id}",
                3600,
                json.dumps(portfolio_data, default=str)
            )
            
        except Exception as e:
            logger.error(f"Error caching portfolio: {e}")
    
    # IBKR callback handlers
    def _on_position_update(self, position: Position):
        """Handle IBKR position updates"""
        try:
            # Find portfolio for current account
            if not self.account_manager or not self.account_manager.current_account:
                return
            
            account_id = self.account_manager.current_account.account_id
            portfolios = self.get_portfolios_by_account(account_id)
            
            if portfolios:
                # Update position in first active portfolio
                portfolio = portfolios[0]
                self.update_position(
                    portfolio.portfolio_id,
                    position.symbol,
                    position.quantity,
                    position.current_price
                )
                
        except Exception as e:
            logger.error(f"Error handling position update: {e}")
    
    def _on_account_update(self, account_info: AccountInfo):
        """Handle IBKR account updates"""
        try:
            # Update portfolios for this account
            portfolios = self.get_portfolios_by_account(account_info.account_id)
            
            for portfolio in portfolios:
                portfolio.cash_balance = account_info.total_cash
                portfolio.buying_power = account_info.buying_power
                portfolio.last_updated = datetime.now(timezone.utc)
                
                # Cache updated portfolio
                self._cache_portfolio(portfolio)
                
        except Exception as e:
            logger.error(f"Error handling account update: {e}")
    
    def _on_order_update(self, order_info: OrderInfo):
        """Handle IBKR order updates"""
        try:
            # Update position when order is filled
            if order_info.status.value == "filled":
                if not self.account_manager or not self.account_manager.current_account:
                    return
                
                account_id = self.account_manager.current_account.account_id
                portfolios = self.get_portfolios_by_account(account_id)
                
                if portfolios:
                    portfolio = portfolios[0]
                    
                    # Calculate new quantity based on order
                    current_position = portfolio.positions.get(order_info.symbol)
                    current_qty = current_position.quantity if current_position else 0
                    
                    if order_info.action == "BUY":
                        new_qty = current_qty + order_info.filled_quantity
                    else:  # SELL
                        new_qty = current_qty - order_info.filled_quantity
                    
                    self.update_position(
                        portfolio.portfolio_id,
                        order_info.symbol,
                        new_qty,
                        order_info.avg_fill_price
                    )
                    
        except Exception as e:
            logger.error(f"Error handling order update: {e}")

# Flask application
def create_app():
    """Create Flask application for Portfolio Management Service"""
    app = Flask(__name__)
    CORS(app)  # Enable CORS for all routes
    
    # Configuration
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key')
    
    # Initialize portfolio manager
    portfolio_manager = PortfolioManager()
    
    # Initialize account manager
    account_manager = AccountManager()
    portfolio_manager.set_account_manager(account_manager)
    
    @app.before_request
    def before_request():
        g.portfolio_manager = portfolio_manager
        g.account_manager = account_manager
    
    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint"""
        return jsonify({
            "status": "healthy",
            "service": "portfolio-management",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "1.0.0"
        })
    
    @app.route('/api/v1/portfolios', methods=['POST'])
    def create_portfolio():
        """Create new portfolio"""
        try:
            data = request.get_json()
            
            account_id = data.get('account_id')
            portfolio_name = data.get('portfolio_name')
            
            if not account_id or not portfolio_name:
                return jsonify({"error": "account_id and portfolio_name required"}), 400
            
            portfolio_id = g.portfolio_manager.create_portfolio(account_id, portfolio_name)
            
            if portfolio_id:
                return jsonify({
                    "portfolio_id": portfolio_id,
                    "message": "Portfolio created successfully"
                }), 201
            else:
                return jsonify({"error": "Failed to create portfolio"}), 500
                
        except Exception as e:
            logger.error(f"Error creating portfolio: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/v1/portfolios/<portfolio_id>', methods=['GET'])
    def get_portfolio(portfolio_id):
        """Get portfolio summary"""
        try:
            summary = g.portfolio_manager.get_portfolio_summary(portfolio_id)
            
            if summary:
                return jsonify(summary)
            else:
                return jsonify({"error": "Portfolio not found"}), 404
                
        except Exception as e:
            logger.error(f"Error getting portfolio: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/v1/portfolios/account/<account_id>', methods=['GET'])
    def get_portfolios_by_account(account_id):
        """Get all portfolios for an account"""
        try:
            portfolios = g.portfolio_manager.get_portfolios_by_account(account_id)
            
            summaries = []
            for portfolio in portfolios:
                summary = g.portfolio_manager.get_portfolio_summary(portfolio.portfolio_id)
                if summary:
                    summaries.append(summary)
            
            return jsonify({
                "account_id": account_id,
                "portfolios": summaries,
                "total_portfolios": len(summaries)
            })
            
        except Exception as e:
            logger.error(f"Error getting portfolios by account: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/v1/portfolios/<portfolio_id>/positions', methods=['POST'])
    def update_position(portfolio_id):
        """Update position in portfolio"""
        try:
            data = request.get_json()
            
            symbol = data.get('symbol')
            quantity = data.get('quantity')
            price = data.get('price')
            strategy_id = data.get('strategy_id')
            signal_id = data.get('signal_id')
            
            if symbol is None or quantity is None or price is None:
                return jsonify({"error": "symbol, quantity, and price required"}), 400
            
            success = g.portfolio_manager.update_position(
                portfolio_id, symbol, quantity, price, strategy_id, signal_id
            )
            
            if success:
                return jsonify({"message": "Position updated successfully"})
            else:
                return jsonify({"error": "Failed to update position"}), 500
                
        except Exception as e:
            logger.error(f"Error updating position: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/v1/accounts', methods=['GET'])
    def get_accounts():
        """Get all accounts"""
        try:
            accounts = g.account_manager.get_accounts()
            
            account_list = []
            for account_id, profile in accounts.items():
                account_data = asdict(profile)
                account_data['account_type'] = profile.account_type.value
                account_data['status'] = profile.status.value
                account_data['created_at'] = profile.created_at.isoformat()
                if profile.last_used:
                    account_data['last_used'] = profile.last_used.isoformat()
                account_list.append(account_data)
            
            return jsonify({
                "accounts": account_list,
                "total_accounts": len(account_list)
            })
            
        except Exception as e:
            logger.error(f"Error getting accounts: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/v1/accounts/<account_id>/switch', methods=['POST'])
    def switch_account(account_id):
        """Switch to different account"""
        try:
            success = g.account_manager.switch_account(account_id)
            
            if success:
                # Connect to IBKR
                connected = g.account_manager.connect_current_account()
                
                return jsonify({
                    "message": f"Switched to account {account_id}",
                    "connected": connected,
                    "account_type": g.account_manager.current_account.account_type.value
                })
            else:
                return jsonify({"error": "Failed to switch account"}), 500
                
        except Exception as e:
            logger.error(f"Error switching account: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/v1/tracking/start', methods=['POST'])
    def start_tracking():
        """Start real-time portfolio tracking"""
        try:
            g.portfolio_manager.start_real_time_tracking()
            return jsonify({"message": "Real-time tracking started"})
            
        except Exception as e:
            logger.error(f"Error starting tracking: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/v1/tracking/stop', methods=['POST'])
    def stop_tracking():
        """Stop real-time portfolio tracking"""
        try:
            g.portfolio_manager.stop_real_time_tracking()
            return jsonify({"message": "Real-time tracking stopped"})
            
        except Exception as e:
            logger.error(f"Error stopping tracking: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/v1/status', methods=['GET'])
    def get_status():
        """Get service status"""
        try:
            current_account = g.account_manager.get_current_account()
            current_session = g.account_manager.get_current_session()
            
            ibkr_connected = (g.account_manager.ibkr_client and 
                            g.account_manager.ibkr_client.is_connected())
            
            return jsonify({
                "service_status": "running",
                "real_time_tracking": g.portfolio_manager.is_running,
                "current_account": {
                    "account_id": current_account.account_id if current_account else None,
                    "account_type": current_account.account_type.value if current_account else None,
                    "account_name": current_account.account_name if current_account else None
                } if current_account else None,
                "current_session": {
                    "session_id": current_session.session_id if current_session else None,
                    "is_active": current_session.is_active if current_session else False,
                    "connection_status": current_session.connection_status if current_session else None
                } if current_session else None,
                "ibkr_connected": ibkr_connected,
                "total_portfolios": len(g.portfolio_manager.portfolios),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return jsonify({"error": str(e)}), 500
    
    return app

# Main execution
if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create and run Flask app
    app = create_app()
    app.run(host='0.0.0.0', port=8005, debug=True)

