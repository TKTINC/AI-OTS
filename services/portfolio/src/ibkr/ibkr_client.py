"""
Interactive Brokers (IBKR) Trading Interface
Runtime-configurable paper/live account switching for AI Options Trading System
"""

import os
import sys
import time
import logging
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import json

# IBKR API imports
try:
    from ibapi.client import EClient
    from ibapi.wrapper import EWrapper
    from ibapi.contract import Contract
    from ibapi.order import Order
    from ibapi.common import TickerId, OrderId
    from ibapi.ticktype import TickType
except ImportError:
    print("Warning: IBKR API not installed. Install with: pip install ibapi")
    # Mock classes for development
    class EClient:
        pass
    class EWrapper:
        pass
    class Contract:
        pass
    class Order:
        pass
    TickerId = int
    OrderId = int
    TickType = object

logger = logging.getLogger(__name__)

class AccountType(Enum):
    """Account type enumeration"""
    PAPER = "paper"
    LIVE = "live"

class OrderStatus(Enum):
    """Order status enumeration"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

class ConnectionStatus(Enum):
    """Connection status enumeration"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    FAILED = "failed"

@dataclass
class IBKRConfig:
    """IBKR configuration settings"""
    # Connection settings
    host: str = "127.0.0.1"
    paper_port: int = 7497  # TWS Paper Trading
    live_port: int = 7496   # TWS Live Trading
    client_id: int = 1
    timeout: int = 30
    
    # Account settings
    paper_account: str = "DU123456"  # Demo account ID
    live_account: str = ""           # Live account ID (to be configured)
    
    # Risk settings
    max_order_value: float = 50000.0
    max_daily_trades: int = 100
    max_position_size: float = 10000.0
    
    # Allowed symbols for trading
    allowed_symbols: List[str] = None
    
    def __post_init__(self):
        if self.allowed_symbols is None:
            self.allowed_symbols = [
                "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", 
                "NVDA", "META", "SPY", "QQQ"
            ]

@dataclass
class Position:
    """Portfolio position data"""
    symbol: str
    quantity: float
    average_cost: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float
    last_updated: datetime

@dataclass
class AccountInfo:
    """Account information data"""
    account_id: str
    account_type: AccountType
    total_cash: float
    buying_power: float
    net_liquidation: float
    total_pnl: float
    day_trades_remaining: int
    last_updated: datetime

@dataclass
class OrderInfo:
    """Order information data"""
    order_id: int
    symbol: str
    action: str  # BUY/SELL
    quantity: float
    order_type: str  # MKT/LMT/STP
    limit_price: Optional[float]
    stop_price: Optional[float]
    status: OrderStatus
    filled_quantity: float
    avg_fill_price: float
    commission: float
    timestamp: datetime

class IBKRClient(EWrapper, EClient):
    """
    Interactive Brokers client with runtime account switching
    Supports both paper and live trading accounts
    """
    
    def __init__(self, config: IBKRConfig = None):
        EClient.__init__(self, self)
        
        self.config = config or IBKRConfig()
        self.account_type = AccountType.PAPER  # Default to paper trading
        self.connection_status = ConnectionStatus.DISCONNECTED
        
        # Data storage
        self.positions: Dict[str, Position] = {}
        self.account_info: Optional[AccountInfo] = None
        self.orders: Dict[int, OrderInfo] = {}
        self.market_data: Dict[str, Dict[str, float]] = {}
        
        # Request tracking
        self.next_order_id = 1
        self.request_id_counter = 1000
        self.pending_requests: Dict[int, str] = {}
        
        # Threading
        self.api_thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        
        # Callbacks
        self.position_callbacks: List[Callable] = []
        self.order_callbacks: List[Callable] = []
        self.account_callbacks: List[Callable] = []
        self.market_data_callbacks: List[Callable] = []
        
        logger.info(f"IBKR Client initialized with {self.account_type.value} account")
    
    def switch_account_type(self, account_type: AccountType) -> bool:
        """
        Switch between paper and live trading accounts at runtime
        
        Args:
            account_type: Target account type (PAPER or LIVE)
            
        Returns:
            bool: True if switch successful, False otherwise
        """
        try:
            if account_type == self.account_type:
                logger.info(f"Already using {account_type.value} account")
                return True
            
            # Disconnect if currently connected
            if self.connection_status == ConnectionStatus.CONNECTED:
                logger.info(f"Disconnecting from {self.account_type.value} account")
                self.disconnect()
                time.sleep(2)  # Allow clean disconnect
            
            # Update account type
            old_type = self.account_type
            self.account_type = account_type
            
            # Clear cached data
            self.positions.clear()
            self.orders.clear()
            self.market_data.clear()
            self.account_info = None
            
            logger.info(f"Switched from {old_type.value} to {account_type.value} account")
            
            # Reconnect with new account type
            return self.connect()
            
        except Exception as e:
            logger.error(f"Error switching account type: {e}")
            return False
    
    def connect(self) -> bool:
        """
        Connect to IBKR TWS/Gateway
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            if self.connection_status == ConnectionStatus.CONNECTED:
                logger.warning("Already connected to IBKR")
                return True
            
            self.connection_status = ConnectionStatus.CONNECTING
            
            # Determine port based on account type
            port = (self.config.paper_port if self.account_type == AccountType.PAPER 
                   else self.config.live_port)
            
            logger.info(f"Connecting to IBKR {self.account_type.value} account on port {port}")
            
            # Connect to TWS/Gateway
            EClient.connect(self, self.config.host, port, self.config.client_id)
            
            # Start API thread
            self.api_thread = threading.Thread(target=self.run, daemon=True)
            self.api_thread.start()
            
            # Wait for connection
            timeout = time.time() + self.config.timeout
            while (self.connection_status == ConnectionStatus.CONNECTING and 
                   time.time() < timeout):
                time.sleep(0.1)
            
            if self.connection_status == ConnectionStatus.CONNECTED:
                logger.info(f"Successfully connected to IBKR {self.account_type.value} account")
                
                # Request initial data
                self._request_initial_data()
                return True
            else:
                logger.error(f"Failed to connect to IBKR {self.account_type.value} account")
                self.connection_status = ConnectionStatus.FAILED
                return False
                
        except Exception as e:
            logger.error(f"Error connecting to IBKR: {e}")
            self.connection_status = ConnectionStatus.FAILED
            return False
    
    def disconnect(self):
        """Disconnect from IBKR"""
        try:
            if self.connection_status == ConnectionStatus.CONNECTED:
                logger.info("Disconnecting from IBKR")
                EClient.disconnect(self)
                
                if self.api_thread and self.api_thread.is_alive():
                    self.api_thread.join(timeout=5)
                
                self.connection_status = ConnectionStatus.DISCONNECTED
                logger.info("Disconnected from IBKR")
                
        except Exception as e:
            logger.error(f"Error disconnecting from IBKR: {e}")
    
    def is_connected(self) -> bool:
        """Check if connected to IBKR"""
        return self.connection_status == ConnectionStatus.CONNECTED
    
    def get_current_account_id(self) -> str:
        """Get current account ID based on account type"""
        return (self.config.paper_account if self.account_type == AccountType.PAPER 
                else self.config.live_account)
    
    # EWrapper callback methods
    def connectAck(self):
        """Connection acknowledgment callback"""
        logger.info("IBKR connection acknowledged")
    
    def nextValidId(self, orderId: int):
        """Next valid order ID callback"""
        self.next_order_id = orderId
        self.connection_status = ConnectionStatus.CONNECTED
        logger.info(f"Connected to IBKR, next order ID: {orderId}")
    
    def error(self, reqId: TickerId, errorCode: int, errorString: str):
        """Error callback"""
        if errorCode in [2104, 2106, 2158]:  # Informational messages
            logger.info(f"IBKR Info [{errorCode}]: {errorString}")
        elif errorCode in [502, 503, 504]:  # Connection errors
            logger.error(f"IBKR Connection Error [{errorCode}]: {errorString}")
            self.connection_status = ConnectionStatus.FAILED
        else:
            logger.error(f"IBKR Error [{errorCode}]: {errorString}")
    
    def position(self, account: str, contract: Contract, position: float, avgCost: float):
        """Position update callback"""
        try:
            symbol = contract.symbol
            
            # Get current market price (use last known price or avgCost as fallback)
            current_price = self.market_data.get(symbol, {}).get('last', avgCost)
            
            # Calculate market value and unrealized P&L
            market_value = position * current_price
            unrealized_pnl = position * (current_price - avgCost) if position != 0 else 0
            
            position_obj = Position(
                symbol=symbol,
                quantity=position,
                average_cost=avgCost,
                current_price=current_price,
                market_value=market_value,
                unrealized_pnl=unrealized_pnl,
                realized_pnl=0.0,  # Will be updated separately
                last_updated=datetime.now(timezone.utc)
            )
            
            with self.lock:
                self.positions[symbol] = position_obj
            
            # Notify callbacks
            for callback in self.position_callbacks:
                try:
                    callback(position_obj)
                except Exception as e:
                    logger.error(f"Error in position callback: {e}")
                    
        except Exception as e:
            logger.error(f"Error processing position update: {e}")
    
    def positionEnd(self):
        """Position updates complete callback"""
        logger.info(f"Position updates complete. Total positions: {len(self.positions)}")
    
    def accountSummary(self, reqId: int, account: str, tag: str, value: str, currency: str):
        """Account summary callback"""
        try:
            if not self.account_info:
                self.account_info = AccountInfo(
                    account_id=account,
                    account_type=self.account_type,
                    total_cash=0.0,
                    buying_power=0.0,
                    net_liquidation=0.0,
                    total_pnl=0.0,
                    day_trades_remaining=0,
                    last_updated=datetime.now(timezone.utc)
                )
            
            # Update account info based on tag
            if tag == "TotalCashValue":
                self.account_info.total_cash = float(value)
            elif tag == "BuyingPower":
                self.account_info.buying_power = float(value)
            elif tag == "NetLiquidation":
                self.account_info.net_liquidation = float(value)
            elif tag == "UnrealizedPnL":
                self.account_info.total_pnl = float(value)
            elif tag == "DayTradesRemaining":
                self.account_info.day_trades_remaining = int(float(value))
            
            # Notify callbacks
            for callback in self.account_callbacks:
                try:
                    callback(self.account_info)
                except Exception as e:
                    logger.error(f"Error in account callback: {e}")
                    
        except Exception as e:
            logger.error(f"Error processing account summary: {e}")
    
    def orderStatus(self, orderId: OrderId, status: str, filled: float, 
                   remaining: float, avgFillPrice: float, permId: int,
                   parentId: int, lastFillPrice: float, clientId: int, whyHeld: str, mktCapPrice: float):
        """Order status callback"""
        try:
            if orderId in self.orders:
                order_info = self.orders[orderId]
                order_info.status = OrderStatus(status.lower())
                order_info.filled_quantity = filled
                order_info.avg_fill_price = avgFillPrice
                
                logger.info(f"Order {orderId} status: {status}, filled: {filled}")
                
                # Notify callbacks
                for callback in self.order_callbacks:
                    try:
                        callback(order_info)
                    except Exception as e:
                        logger.error(f"Error in order callback: {e}")
                        
        except Exception as e:
            logger.error(f"Error processing order status: {e}")
    
    def tickPrice(self, reqId: TickerId, tickType: TickType, price: float, attrib):
        """Market data price tick callback"""
        try:
            symbol = self.pending_requests.get(reqId)
            if symbol:
                if symbol not in self.market_data:
                    self.market_data[symbol] = {}
                
                # Map tick types to data fields
                if tickType == 1:  # Bid
                    self.market_data[symbol]['bid'] = price
                elif tickType == 2:  # Ask
                    self.market_data[symbol]['ask'] = price
                elif tickType == 4:  # Last
                    self.market_data[symbol]['last'] = price
                elif tickType == 6:  # High
                    self.market_data[symbol]['high'] = price
                elif tickType == 7:  # Low
                    self.market_data[symbol]['low'] = price
                elif tickType == 9:  # Close
                    self.market_data[symbol]['close'] = price
                
                # Notify callbacks
                for callback in self.market_data_callbacks:
                    try:
                        callback(symbol, self.market_data[symbol])
                    except Exception as e:
                        logger.error(f"Error in market data callback: {e}")
                        
        except Exception as e:
            logger.error(f"Error processing tick price: {e}")
    
    # Trading methods
    def place_order(self, symbol: str, action: str, quantity: float, 
                   order_type: str = "MKT", limit_price: Optional[float] = None,
                   stop_price: Optional[float] = None) -> Optional[int]:
        """
        Place a trading order
        
        Args:
            symbol: Trading symbol
            action: BUY or SELL
            quantity: Number of shares
            order_type: Order type (MKT, LMT, STP, etc.)
            limit_price: Limit price for limit orders
            stop_price: Stop price for stop orders
            
        Returns:
            Order ID if successful, None otherwise
        """
        try:
            if not self.is_connected():
                logger.error("Not connected to IBKR")
                return None
            
            # Validate order
            if not self._validate_order(symbol, action, quantity, order_type):
                return None
            
            # Create contract
            contract = Contract()
            contract.symbol = symbol
            contract.secType = "STK"
            contract.exchange = "SMART"
            contract.currency = "USD"
            
            # Create order
            order = Order()
            order.action = action.upper()
            order.totalQuantity = quantity
            order.orderType = order_type.upper()
            
            if limit_price and order_type.upper() in ["LMT", "STP LMT"]:
                order.lmtPrice = limit_price
            
            if stop_price and order_type.upper() in ["STP", "STP LMT"]:
                order.auxPrice = stop_price
            
            # Get order ID
            order_id = self.next_order_id
            self.next_order_id += 1
            
            # Store order info
            order_info = OrderInfo(
                order_id=order_id,
                symbol=symbol,
                action=action.upper(),
                quantity=quantity,
                order_type=order_type.upper(),
                limit_price=limit_price,
                stop_price=stop_price,
                status=OrderStatus.PENDING,
                filled_quantity=0.0,
                avg_fill_price=0.0,
                commission=0.0,
                timestamp=datetime.now(timezone.utc)
            )
            
            with self.lock:
                self.orders[order_id] = order_info
            
            # Place order
            self.placeOrder(order_id, contract, order)
            
            logger.info(f"Placed {action} order for {quantity} shares of {symbol} (Order ID: {order_id})")
            return order_id
            
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return None
    
    def cancel_order(self, order_id: int) -> bool:
        """
        Cancel an order
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if cancellation request sent successfully
        """
        try:
            if not self.is_connected():
                logger.error("Not connected to IBKR")
                return False
            
            if order_id not in self.orders:
                logger.error(f"Order {order_id} not found")
                return False
            
            self.cancelOrder(order_id)
            logger.info(f"Cancellation requested for order {order_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    def get_positions(self) -> Dict[str, Position]:
        """Get current positions"""
        with self.lock:
            return self.positions.copy()
    
    def get_account_info(self) -> Optional[AccountInfo]:
        """Get account information"""
        return self.account_info
    
    def get_orders(self) -> Dict[int, OrderInfo]:
        """Get order history"""
        with self.lock:
            return self.orders.copy()
    
    def get_market_data(self, symbol: str) -> Optional[Dict[str, float]]:
        """Get market data for symbol"""
        return self.market_data.get(symbol)
    
    def subscribe_market_data(self, symbol: str) -> bool:
        """
        Subscribe to real-time market data
        
        Args:
            symbol: Trading symbol
            
        Returns:
            True if subscription successful
        """
        try:
            if not self.is_connected():
                logger.error("Not connected to IBKR")
                return False
            
            # Create contract
            contract = Contract()
            contract.symbol = symbol
            contract.secType = "STK"
            contract.exchange = "SMART"
            contract.currency = "USD"
            
            # Get request ID
            req_id = self.request_id_counter
            self.request_id_counter += 1
            
            # Store request mapping
            self.pending_requests[req_id] = symbol
            
            # Request market data
            self.reqMktData(req_id, contract, "", False, False, [])
            
            logger.info(f"Subscribed to market data for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error subscribing to market data for {symbol}: {e}")
            return False
    
    def unsubscribe_market_data(self, symbol: str) -> bool:
        """
        Unsubscribe from market data
        
        Args:
            symbol: Trading symbol
            
        Returns:
            True if unsubscription successful
        """
        try:
            # Find request ID for symbol
            req_id = None
            for rid, sym in self.pending_requests.items():
                if sym == symbol:
                    req_id = rid
                    break
            
            if req_id:
                self.cancelMktData(req_id)
                del self.pending_requests[req_id]
                logger.info(f"Unsubscribed from market data for {symbol}")
                return True
            else:
                logger.warning(f"No active subscription found for {symbol}")
                return False
                
        except Exception as e:
            logger.error(f"Error unsubscribing from market data for {symbol}: {e}")
            return False
    
    # Callback registration methods
    def add_position_callback(self, callback: Callable[[Position], None]):
        """Add position update callback"""
        self.position_callbacks.append(callback)
    
    def add_order_callback(self, callback: Callable[[OrderInfo], None]):
        """Add order update callback"""
        self.order_callbacks.append(callback)
    
    def add_account_callback(self, callback: Callable[[AccountInfo], None]):
        """Add account update callback"""
        self.account_callbacks.append(callback)
    
    def add_market_data_callback(self, callback: Callable[[str, Dict[str, float]], None]):
        """Add market data callback"""
        self.market_data_callbacks.append(callback)
    
    # Private methods
    def _request_initial_data(self):
        """Request initial account and position data"""
        try:
            # Request positions
            self.reqPositions()
            
            # Request account summary
            account_id = self.get_current_account_id()
            tags = "TotalCashValue,BuyingPower,NetLiquidation,UnrealizedPnL,DayTradesRemaining"
            self.reqAccountSummary(9001, "All", tags)
            
            # Subscribe to market data for allowed symbols
            for symbol in self.config.allowed_symbols:
                self.subscribe_market_data(symbol)
                
        except Exception as e:
            logger.error(f"Error requesting initial data: {e}")
    
    def _validate_order(self, symbol: str, action: str, quantity: float, order_type: str) -> bool:
        """
        Validate order parameters
        
        Args:
            symbol: Trading symbol
            action: BUY or SELL
            quantity: Number of shares
            order_type: Order type
            
        Returns:
            True if order is valid
        """
        try:
            # Check symbol
            if symbol not in self.config.allowed_symbols:
                logger.error(f"Symbol {symbol} not in allowed symbols list")
                return False
            
            # Check action
            if action.upper() not in ["BUY", "SELL"]:
                logger.error(f"Invalid action: {action}")
                return False
            
            # Check quantity
            if quantity <= 0:
                logger.error(f"Invalid quantity: {quantity}")
                return False
            
            # Check order type
            valid_order_types = ["MKT", "LMT", "STP", "STP LMT"]
            if order_type.upper() not in valid_order_types:
                logger.error(f"Invalid order type: {order_type}")
                return False
            
            # Check order value against limits
            current_price = self.market_data.get(symbol, {}).get('last', 0)
            if current_price > 0:
                order_value = quantity * current_price
                if order_value > self.config.max_order_value:
                    logger.error(f"Order value {order_value} exceeds limit {self.config.max_order_value}")
                    return False
            
            # Check daily trade limit
            today_orders = [o for o in self.orders.values() 
                          if o.timestamp.date() == datetime.now().date()]
            if len(today_orders) >= self.config.max_daily_trades:
                logger.error(f"Daily trade limit {self.config.max_daily_trades} exceeded")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating order: {e}")
            return False

# Utility functions
def create_ibkr_client(account_type: AccountType = AccountType.PAPER, 
                      config: IBKRConfig = None) -> IBKRClient:
    """
    Create and configure IBKR client
    
    Args:
        account_type: Account type (PAPER or LIVE)
        config: IBKR configuration
        
    Returns:
        Configured IBKR client
    """
    client = IBKRClient(config)
    client.account_type = account_type
    return client

def load_config_from_env() -> IBKRConfig:
    """Load IBKR configuration from environment variables"""
    return IBKRConfig(
        host=os.getenv("IBKR_HOST", "127.0.0.1"),
        paper_port=int(os.getenv("IBKR_PAPER_PORT", "7497")),
        live_port=int(os.getenv("IBKR_LIVE_PORT", "7496")),
        client_id=int(os.getenv("IBKR_CLIENT_ID", "1")),
        timeout=int(os.getenv("IBKR_TIMEOUT", "30")),
        paper_account=os.getenv("IBKR_PAPER_ACCOUNT", "DU123456"),
        live_account=os.getenv("IBKR_LIVE_ACCOUNT", ""),
        max_order_value=float(os.getenv("IBKR_MAX_ORDER_VALUE", "50000")),
        max_daily_trades=int(os.getenv("IBKR_MAX_DAILY_TRADES", "100")),
        max_position_size=float(os.getenv("IBKR_MAX_POSITION_SIZE", "10000"))
    )

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Load configuration
    config = load_config_from_env()
    
    # Create client
    client = create_ibkr_client(AccountType.PAPER, config)
    
    # Add callbacks
    def on_position_update(position: Position):
        print(f"Position update: {position.symbol} = {position.quantity} @ {position.current_price}")
    
    def on_order_update(order: OrderInfo):
        print(f"Order update: {order.order_id} {order.symbol} {order.status.value}")
    
    def on_account_update(account: AccountInfo):
        print(f"Account update: {account.account_id} buying power = {account.buying_power}")
    
    client.add_position_callback(on_position_update)
    client.add_order_callback(on_order_update)
    client.add_account_callback(on_account_update)
    
    try:
        # Connect to paper trading
        if client.connect():
            print("Connected to IBKR paper trading")
            
            # Wait for initial data
            time.sleep(5)
            
            # Switch to live trading (if configured)
            if config.live_account:
                print("Switching to live trading...")
                if client.switch_account_type(AccountType.LIVE):
                    print("Successfully switched to live trading")
                else:
                    print("Failed to switch to live trading")
            
            # Keep running
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("Shutting down...")
                
        else:
            print("Failed to connect to IBKR")
            
    finally:
        client.disconnect()

