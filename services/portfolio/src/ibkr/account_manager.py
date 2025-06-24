"""
IBKR Account Manager and Configuration
Handles account switching, configuration management, and account validation
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
import yaml

from .ibkr_client import AccountType, IBKRConfig, IBKRClient

logger = logging.getLogger(__name__)

class AccountStatus(Enum):
    """Account status enumeration"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    UNKNOWN = "unknown"

@dataclass
class AccountProfile:
    """Account profile configuration"""
    account_id: str
    account_type: AccountType
    account_name: str
    description: str
    status: AccountStatus
    
    # Trading permissions
    can_trade_stocks: bool = True
    can_trade_options: bool = True
    can_trade_futures: bool = False
    can_short_sell: bool = True
    
    # Risk limits
    max_order_value: float = 50000.0
    max_daily_trades: int = 100
    max_position_size: float = 10000.0
    max_portfolio_value: float = 1000000.0
    
    # Allowed symbols
    allowed_symbols: List[str] = None
    restricted_symbols: List[str] = None
    
    # Metadata
    created_at: datetime = None
    last_used: datetime = None
    
    def __post_init__(self):
        if self.allowed_symbols is None:
            self.allowed_symbols = [
                "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", 
                "NVDA", "META", "SPY", "QQQ"
            ]
        if self.restricted_symbols is None:
            self.restricted_symbols = []
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)

@dataclass
class TradingSession:
    """Trading session information"""
    session_id: str
    account_id: str
    account_type: AccountType
    start_time: datetime
    end_time: Optional[datetime] = None
    
    # Session statistics
    orders_placed: int = 0
    orders_filled: int = 0
    total_volume: float = 0.0
    total_pnl: float = 0.0
    
    # Session status
    is_active: bool = True
    connection_status: str = "connected"

class AccountManager:
    """
    Manages IBKR account configurations and switching
    Provides runtime account switching capabilities
    """
    
    def __init__(self, config_file: str = None):
        self.config_file = config_file or "config/accounts.yaml"
        self.accounts: Dict[str, AccountProfile] = {}
        self.current_account: Optional[AccountProfile] = None
        self.current_session: Optional[TradingSession] = None
        self.ibkr_client: Optional[IBKRClient] = None
        
        # Load account configurations
        self.load_accounts()
        
        logger.info(f"Account Manager initialized with {len(self.accounts)} accounts")
    
    def load_accounts(self):
        """Load account configurations from file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config_data = yaml.safe_load(f)
                
                for account_id, account_data in config_data.get('accounts', {}).items():
                    # Convert string dates back to datetime objects
                    if 'created_at' in account_data:
                        account_data['created_at'] = datetime.fromisoformat(account_data['created_at'])
                    if 'last_used' in account_data:
                        account_data['last_used'] = datetime.fromisoformat(account_data['last_used'])
                    
                    # Convert account_type and status to enums
                    account_data['account_type'] = AccountType(account_data['account_type'])
                    account_data['status'] = AccountStatus(account_data['status'])
                    
                    profile = AccountProfile(**account_data)
                    self.accounts[account_id] = profile
                
                logger.info(f"Loaded {len(self.accounts)} account configurations")
            else:
                # Create default accounts if no config file exists
                self._create_default_accounts()
                self.save_accounts()
                
        except Exception as e:
            logger.error(f"Error loading account configurations: {e}")
            self._create_default_accounts()
    
    def save_accounts(self):
        """Save account configurations to file"""
        try:
            # Ensure config directory exists
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            
            # Convert accounts to serializable format
            config_data = {'accounts': {}}
            
            for account_id, profile in self.accounts.items():
                account_data = asdict(profile)
                
                # Convert datetime objects to ISO strings
                if account_data['created_at']:
                    account_data['created_at'] = account_data['created_at'].isoformat()
                if account_data['last_used']:
                    account_data['last_used'] = account_data['last_used'].isoformat()
                
                # Convert enums to strings
                account_data['account_type'] = account_data['account_type'].value
                account_data['status'] = account_data['status'].value
                
                config_data['accounts'][account_id] = account_data
            
            with open(self.config_file, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
            
            logger.info(f"Saved {len(self.accounts)} account configurations")
            
        except Exception as e:
            logger.error(f"Error saving account configurations: {e}")
    
    def _create_default_accounts(self):
        """Create default account configurations"""
        # Paper trading account
        paper_account = AccountProfile(
            account_id="DU123456",
            account_type=AccountType.PAPER,
            account_name="Paper Trading Account",
            description="Default paper trading account for testing and development",
            status=AccountStatus.ACTIVE,
            max_order_value=50000.0,
            max_daily_trades=100,
            max_position_size=10000.0,
            max_portfolio_value=1000000.0
        )
        
        # Live trading account (placeholder)
        live_account = AccountProfile(
            account_id="U123456",
            account_type=AccountType.LIVE,
            account_name="Live Trading Account",
            description="Live trading account - configure with real account ID",
            status=AccountStatus.INACTIVE,  # Inactive until properly configured
            max_order_value=25000.0,
            max_daily_trades=50,
            max_position_size=5000.0,
            max_portfolio_value=500000.0
        )
        
        self.accounts[paper_account.account_id] = paper_account
        self.accounts[live_account.account_id] = live_account
        
        # Set paper account as default
        self.current_account = paper_account
        
        logger.info("Created default account configurations")
    
    def get_accounts(self) -> Dict[str, AccountProfile]:
        """Get all account profiles"""
        return self.accounts.copy()
    
    def get_account(self, account_id: str) -> Optional[AccountProfile]:
        """Get specific account profile"""
        return self.accounts.get(account_id)
    
    def get_current_account(self) -> Optional[AccountProfile]:
        """Get current active account"""
        return self.current_account
    
    def add_account(self, profile: AccountProfile) -> bool:
        """
        Add new account profile
        
        Args:
            profile: Account profile to add
            
        Returns:
            True if account added successfully
        """
        try:
            if profile.account_id in self.accounts:
                logger.warning(f"Account {profile.account_id} already exists")
                return False
            
            self.accounts[profile.account_id] = profile
            self.save_accounts()
            
            logger.info(f"Added account {profile.account_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding account {profile.account_id}: {e}")
            return False
    
    def update_account(self, account_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update account profile
        
        Args:
            account_id: Account ID to update
            updates: Dictionary of fields to update
            
        Returns:
            True if account updated successfully
        """
        try:
            if account_id not in self.accounts:
                logger.error(f"Account {account_id} not found")
                return False
            
            profile = self.accounts[account_id]
            
            # Update fields
            for field, value in updates.items():
                if hasattr(profile, field):
                    setattr(profile, field, value)
                else:
                    logger.warning(f"Unknown field {field} for account {account_id}")
            
            self.save_accounts()
            
            logger.info(f"Updated account {account_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating account {account_id}: {e}")
            return False
    
    def remove_account(self, account_id: str) -> bool:
        """
        Remove account profile
        
        Args:
            account_id: Account ID to remove
            
        Returns:
            True if account removed successfully
        """
        try:
            if account_id not in self.accounts:
                logger.error(f"Account {account_id} not found")
                return False
            
            # Don't remove current account
            if self.current_account and self.current_account.account_id == account_id:
                logger.error(f"Cannot remove current active account {account_id}")
                return False
            
            del self.accounts[account_id]
            self.save_accounts()
            
            logger.info(f"Removed account {account_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing account {account_id}: {e}")
            return False
    
    def switch_account(self, account_id: str) -> bool:
        """
        Switch to different account
        
        Args:
            account_id: Target account ID
            
        Returns:
            True if switch successful
        """
        try:
            if account_id not in self.accounts:
                logger.error(f"Account {account_id} not found")
                return False
            
            target_account = self.accounts[account_id]
            
            # Check account status
            if target_account.status != AccountStatus.ACTIVE:
                logger.error(f"Account {account_id} is not active (status: {target_account.status.value})")
                return False
            
            # End current session if active
            if self.current_session and self.current_session.is_active:
                self.end_session()
            
            # Disconnect current IBKR client if connected
            if self.ibkr_client and self.ibkr_client.is_connected():
                logger.info("Disconnecting from current IBKR account")
                self.ibkr_client.disconnect()
            
            # Update current account
            old_account = self.current_account
            self.current_account = target_account
            
            # Update last used timestamp
            target_account.last_used = datetime.now(timezone.utc)
            self.save_accounts()
            
            # Create new IBKR client with target account configuration
            ibkr_config = self._create_ibkr_config(target_account)
            self.ibkr_client = IBKRClient(ibkr_config)
            self.ibkr_client.account_type = target_account.account_type
            
            # Start new session
            self.start_session()
            
            logger.info(f"Switched from {old_account.account_id if old_account else 'None'} to {account_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error switching to account {account_id}: {e}")
            return False
    
    def connect_current_account(self) -> bool:
        """
        Connect to current account via IBKR
        
        Returns:
            True if connection successful
        """
        try:
            if not self.current_account:
                logger.error("No current account selected")
                return False
            
            if not self.ibkr_client:
                # Create IBKR client if not exists
                ibkr_config = self._create_ibkr_config(self.current_account)
                self.ibkr_client = IBKRClient(ibkr_config)
                self.ibkr_client.account_type = self.current_account.account_type
            
            # Connect to IBKR
            if self.ibkr_client.connect():
                logger.info(f"Connected to IBKR account {self.current_account.account_id}")
                
                # Update session status
                if self.current_session:
                    self.current_session.connection_status = "connected"
                
                return True
            else:
                logger.error(f"Failed to connect to IBKR account {self.current_account.account_id}")
                
                # Update session status
                if self.current_session:
                    self.current_session.connection_status = "failed"
                
                return False
                
        except Exception as e:
            logger.error(f"Error connecting to current account: {e}")
            return False
    
    def disconnect_current_account(self):
        """Disconnect from current IBKR account"""
        try:
            if self.ibkr_client and self.ibkr_client.is_connected():
                self.ibkr_client.disconnect()
                logger.info("Disconnected from IBKR account")
                
                # Update session status
                if self.current_session:
                    self.current_session.connection_status = "disconnected"
                    
        except Exception as e:
            logger.error(f"Error disconnecting from current account: {e}")
    
    def start_session(self) -> str:
        """
        Start new trading session
        
        Returns:
            Session ID
        """
        try:
            if not self.current_account:
                raise ValueError("No current account selected")
            
            # End previous session if active
            if self.current_session and self.current_session.is_active:
                self.end_session()
            
            # Create new session
            session_id = f"session_{self.current_account.account_id}_{int(datetime.now().timestamp())}"
            
            self.current_session = TradingSession(
                session_id=session_id,
                account_id=self.current_account.account_id,
                account_type=self.current_account.account_type,
                start_time=datetime.now(timezone.utc)
            )
            
            logger.info(f"Started trading session {session_id} for account {self.current_account.account_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Error starting trading session: {e}")
            return ""
    
    def end_session(self):
        """End current trading session"""
        try:
            if self.current_session and self.current_session.is_active:
                self.current_session.end_time = datetime.now(timezone.utc)
                self.current_session.is_active = False
                
                logger.info(f"Ended trading session {self.current_session.session_id}")
                
                # Log session statistics
                duration = (self.current_session.end_time - self.current_session.start_time).total_seconds()
                logger.info(f"Session duration: {duration:.1f} seconds")
                logger.info(f"Orders placed: {self.current_session.orders_placed}")
                logger.info(f"Orders filled: {self.current_session.orders_filled}")
                logger.info(f"Total volume: {self.current_session.total_volume}")
                logger.info(f"Total P&L: {self.current_session.total_pnl}")
                
        except Exception as e:
            logger.error(f"Error ending trading session: {e}")
    
    def get_current_session(self) -> Optional[TradingSession]:
        """Get current trading session"""
        return self.current_session
    
    def validate_account_access(self, account_id: str) -> bool:
        """
        Validate account access and permissions
        
        Args:
            account_id: Account ID to validate
            
        Returns:
            True if account is accessible
        """
        try:
            account = self.accounts.get(account_id)
            if not account:
                logger.error(f"Account {account_id} not found")
                return False
            
            if account.status != AccountStatus.ACTIVE:
                logger.error(f"Account {account_id} is not active")
                return False
            
            # Additional validation checks can be added here
            # e.g., check account balance, permissions, etc.
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating account access for {account_id}: {e}")
            return False
    
    def get_account_permissions(self, account_id: str) -> Dict[str, bool]:
        """
        Get account trading permissions
        
        Args:
            account_id: Account ID
            
        Returns:
            Dictionary of permissions
        """
        account = self.accounts.get(account_id)
        if not account:
            return {}
        
        return {
            "can_trade_stocks": account.can_trade_stocks,
            "can_trade_options": account.can_trade_options,
            "can_trade_futures": account.can_trade_futures,
            "can_short_sell": account.can_short_sell
        }
    
    def get_account_limits(self, account_id: str) -> Dict[str, float]:
        """
        Get account risk limits
        
        Args:
            account_id: Account ID
            
        Returns:
            Dictionary of limits
        """
        account = self.accounts.get(account_id)
        if not account:
            return {}
        
        return {
            "max_order_value": account.max_order_value,
            "max_daily_trades": account.max_daily_trades,
            "max_position_size": account.max_position_size,
            "max_portfolio_value": account.max_portfolio_value
        }
    
    def _create_ibkr_config(self, account: AccountProfile) -> IBKRConfig:
        """
        Create IBKR configuration from account profile
        
        Args:
            account: Account profile
            
        Returns:
            IBKR configuration
        """
        return IBKRConfig(
            host=os.getenv("IBKR_HOST", "127.0.0.1"),
            paper_port=int(os.getenv("IBKR_PAPER_PORT", "7497")),
            live_port=int(os.getenv("IBKR_LIVE_PORT", "7496")),
            client_id=int(os.getenv("IBKR_CLIENT_ID", "1")),
            timeout=int(os.getenv("IBKR_TIMEOUT", "30")),
            paper_account=account.account_id if account.account_type == AccountType.PAPER else os.getenv("IBKR_PAPER_ACCOUNT", "DU123456"),
            live_account=account.account_id if account.account_type == AccountType.LIVE else os.getenv("IBKR_LIVE_ACCOUNT", ""),
            max_order_value=account.max_order_value,
            max_daily_trades=account.max_daily_trades,
            max_position_size=account.max_position_size,
            allowed_symbols=account.allowed_symbols
        )

# Configuration utilities
def load_account_manager(config_file: str = None) -> AccountManager:
    """Load account manager with configuration"""
    return AccountManager(config_file)

def create_account_profile(account_id: str, account_type: AccountType, 
                         account_name: str, **kwargs) -> AccountProfile:
    """Create new account profile"""
    return AccountProfile(
        account_id=account_id,
        account_type=account_type,
        account_name=account_name,
        description=kwargs.get('description', ''),
        status=kwargs.get('status', AccountStatus.ACTIVE),
        **{k: v for k, v in kwargs.items() if k not in ['description', 'status']}
    )

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create account manager
    manager = AccountManager("config/test_accounts.yaml")
    
    # List accounts
    print("Available accounts:")
    for account_id, profile in manager.get_accounts().items():
        print(f"  {account_id}: {profile.account_name} ({profile.account_type.value})")
    
    # Switch to paper account
    if manager.switch_account("DU123456"):
        print("Switched to paper trading account")
        
        # Connect to IBKR
        if manager.connect_current_account():
            print("Connected to IBKR")
            
            # Keep running for a bit
            import time
            time.sleep(5)
            
            # Disconnect
            manager.disconnect_current_account()
            print("Disconnected from IBKR")
        
        # End session
        manager.end_session()
    
    print("Account manager example completed")

