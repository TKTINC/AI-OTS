"""
Databento Client Wrapper for AI Options Trading System
Provides unified interface for both real Databento API and mock data
"""

import os
import json
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Iterator, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
import aiohttp
import websockets
from concurrent.futures import ThreadPoolExecutor
import threading

from mock_data_generator import MockDataGenerator, generate_sample_data

logger = logging.getLogger(__name__)

class DatabentoMode(Enum):
    """Databento client modes"""
    LIVE = "live"
    HISTORICAL = "historical"
    MOCK = "mock"

class DatabentoDataset(Enum):
    """Databento dataset identifiers"""
    XNAS_ITCH = "XNAS.ITCH"  # NASDAQ stocks
    OPRA_PILLAR = "OPRA.PILLAR"  # Options
    GLBX_MDP3 = "GLBX.MDP3"  # CME futures
    DBEQ_BASIC = "DBEQ.BASIC"  # Databento Equities

class DatabentoSchema(Enum):
    """Databento schema types"""
    MBO = "mbo"
    MBP_1 = "mbp-1"
    MBP_10 = "mbp-10"
    TBBO = "tbbo"
    TRADES = "trades"
    OHLCV_1S = "ohlcv-1s"
    OHLCV_1M = "ohlcv-1m"
    OHLCV_1H = "ohlcv-1h"
    OHLCV_1D = "ohlcv-1d"

@dataclass
class DatabentoConfig:
    """Databento client configuration"""
    api_key: str
    mode: DatabentoMode = DatabentoMode.MOCK
    base_url: str = "https://hist.databento.com"
    live_url: str = "wss://live.databento.com/v0/stream"
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Mock data settings
    mock_symbols: List[str] = None
    mock_update_interval: float = 1.0
    mock_data_file: Optional[str] = None
    
    # Rate limiting
    requests_per_second: int = 10
    burst_limit: int = 100

@dataclass
class SubscriptionRequest:
    """Data subscription request"""
    dataset: str
    schema: str
    symbols: List[str]
    start: Optional[str] = None
    end: Optional[str] = None
    stype_in: Optional[str] = None
    stype_out: Optional[str] = None

class DatabentoClient:
    """Unified Databento client with mock data support"""
    
    def __init__(self, config: DatabentoConfig):
        """Initialize Databento client"""
        self.config = config
        self.mode = config.mode
        self.session = None
        self.websocket = None
        self.mock_generator = None
        self.subscriptions = {}
        self.callbacks = {}
        self.running = False
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="databento")
        
        # Rate limiting
        self.request_times = []
        self.rate_limit_lock = threading.Lock()
        
        # Initialize based on mode
        if self.mode == DatabentoMode.MOCK:
            self._init_mock_mode()
        else:
            self._init_api_mode()
    
    def _init_mock_mode(self):
        """Initialize mock data mode"""
        symbols = self.config.mock_symbols or ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
        self.mock_generator = MockDataGenerator(symbols)
        
        # Load mock data from file if specified
        if self.config.mock_data_file and os.path.exists(self.config.mock_data_file):
            try:
                with open(self.config.mock_data_file, 'r') as f:
                    self.mock_data = json.load(f)
                logger.info(f"Loaded mock data from {self.config.mock_data_file}")
            except Exception as e:
                logger.warning(f"Failed to load mock data file: {e}")
                self.mock_data = None
        else:
            self.mock_data = None
        
        logger.info(f"Initialized Databento client in MOCK mode with {len(symbols)} symbols")
    
    def _init_api_mode(self):
        """Initialize API mode"""
        if not self.config.api_key:
            raise ValueError("API key is required for live/historical mode")
        
        logger.info(f"Initialized Databento client in {self.mode.value.upper()} mode")
    
    async def _create_session(self):
        """Create HTTP session for API requests"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json"
            }
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers=headers
            )
    
    async def _close_session(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None
    
    def _check_rate_limit(self):
        """Check and enforce rate limiting"""
        with self.rate_limit_lock:
            now = time.time()
            
            # Remove old requests outside the window
            self.request_times = [t for t in self.request_times if now - t < 1.0]
            
            # Check if we're within limits
            if len(self.request_times) >= self.config.requests_per_second:
                sleep_time = 1.0 - (now - self.request_times[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            # Add current request
            self.request_times.append(now)
    
    async def get_instruments(self, dataset: str, symbols: List[str] = None) -> List[Dict[str, Any]]:
        """Get instrument definitions"""
        if self.mode == DatabentoMode.MOCK:
            return self._get_mock_instruments(symbols)
        
        # Real API call
        await self._create_session()
        self._check_rate_limit()
        
        url = f"{self.config.base_url}/v0/metadata.list_symbols"
        params = {
            "dataset": dataset,
            "symbols": ",".join(symbols) if symbols else None
        }
        
        try:
            async with self.session.get(url, params=params) as response:
                response.raise_for_status()
                data = await response.json()
                return data.get("symbols", [])
        
        except Exception as e:
            logger.error(f"Failed to get instruments: {e}")
            return []
    
    def _get_mock_instruments(self, symbols: List[str] = None) -> List[Dict[str, Any]]:
        """Get mock instrument definitions"""
        if symbols:
            # Filter mock generator symbols
            filtered_symbols = [s for s in symbols if s in self.mock_generator.symbols]
            if filtered_symbols != self.mock_generator.symbols:
                self.mock_generator.symbols = filtered_symbols
                self.mock_generator._generate_option_instruments()
        
        instruments = []
        instruments.extend(self.mock_generator.generate_stock_instruments())
        instruments.extend(self.mock_generator.generate_option_instruments())
        
        return instruments
    
    async def get_historical_data(self, request: SubscriptionRequest) -> List[Dict[str, Any]]:
        """Get historical market data"""
        if self.mode == DatabentoMode.MOCK:
            return self._get_mock_historical_data(request)
        
        # Real API call
        await self._create_session()
        self._check_rate_limit()
        
        url = f"{self.config.base_url}/v0/timeseries.get_range"
        params = {
            "dataset": request.dataset,
            "schema": request.schema,
            "symbols": ",".join(request.symbols),
            "start": request.start,
            "end": request.end,
            "stype_in": request.stype_in,
            "stype_out": request.stype_out
        }
        
        try:
            async with self.session.get(url, params=params) as response:
                response.raise_for_status()
                
                # Handle different response formats
                content_type = response.headers.get("content-type", "")
                
                if "application/json" in content_type:
                    return await response.json()
                else:
                    # Binary data (DBN format)
                    data = await response.read()
                    return self._parse_dbn_data(data)
        
        except Exception as e:
            logger.error(f"Failed to get historical data: {e}")
            return []
    
    def _get_mock_historical_data(self, request: SubscriptionRequest) -> List[Dict[str, Any]]:
        """Get mock historical data"""
        data = []
        
        for symbol in request.symbols:
            if symbol not in self.mock_generator.symbols:
                continue
            
            if request.schema == "trades":
                data.extend(self.mock_generator.generate_stock_trades(symbol, count=100))
            elif request.schema == "tbbo":
                data.extend(self.mock_generator.generate_stock_quotes(symbol, count=100))
            elif request.schema.startswith("ohlcv"):
                interval = request.schema.split("-")[1] if "-" in request.schema else "1m"
                data.extend(self.mock_generator.generate_ohlcv_data(symbol, interval, count=100))
        
        return data
    
    async def subscribe_live(self, request: SubscriptionRequest, 
                           callback: Callable[[Dict[str, Any]], None]):
        """Subscribe to live market data"""
        if self.mode == DatabentoMode.MOCK:
            return await self._subscribe_mock_live(request, callback)
        
        # Real WebSocket subscription
        subscription_id = f"{request.dataset}:{request.schema}:{','.join(request.symbols)}"
        self.subscriptions[subscription_id] = request
        self.callbacks[subscription_id] = callback
        
        if not self.websocket:
            await self._connect_websocket()
        
        # Send subscription message
        subscribe_msg = {
            "action": "subscribe",
            "dataset": request.dataset,
            "schema": request.schema,
            "symbols": request.symbols
        }
        
        await self.websocket.send(json.dumps(subscribe_msg))
        logger.info(f"Subscribed to live data: {subscription_id}")
    
    async def _subscribe_mock_live(self, request: SubscriptionRequest,
                                 callback: Callable[[Dict[str, Any]], None]):
        """Subscribe to mock live data"""
        subscription_id = f"mock:{request.schema}:{','.join(request.symbols)}"
        self.subscriptions[subscription_id] = request
        self.callbacks[subscription_id] = callback
        
        # Start mock data generation in background
        asyncio.create_task(self._generate_mock_live_data(subscription_id))
        
        logger.info(f"Subscribed to mock live data: {subscription_id}")
    
    async def _generate_mock_live_data(self, subscription_id: str):
        """Generate mock live data continuously"""
        request = self.subscriptions[subscription_id]
        callback = self.callbacks[subscription_id]
        
        while subscription_id in self.subscriptions and self.running:
            try:
                for symbol in request.symbols:
                    if symbol not in self.mock_generator.symbols:
                        continue
                    
                    # Generate data based on schema
                    if request.schema == "trades":
                        data = self.mock_generator.generate_stock_trades(symbol, count=1)
                    elif request.schema == "tbbo":
                        data = self.mock_generator.generate_stock_quotes(symbol, count=1)
                    elif request.schema.startswith("ohlcv"):
                        interval = request.schema.split("-")[1] if "-" in request.schema else "1m"
                        data = self.mock_generator.generate_ohlcv_data(symbol, interval, count=1)
                    else:
                        continue
                    
                    # Send data to callback
                    for record in data:
                        record["subscription_id"] = subscription_id
                        record["symbol"] = symbol
                        await asyncio.get_event_loop().run_in_executor(
                            self.executor, callback, record
                        )
                
                await asyncio.sleep(self.config.mock_update_interval)
            
            except Exception as e:
                logger.error(f"Error generating mock live data: {e}")
                await asyncio.sleep(1)
    
    async def _connect_websocket(self):
        """Connect to Databento WebSocket"""
        try:
            headers = {
                "Authorization": f"Bearer {self.config.api_key}"
            }
            
            self.websocket = await websockets.connect(
                self.config.live_url,
                extra_headers=headers
            )
            
            # Start message handler
            asyncio.create_task(self._handle_websocket_messages())
            
            logger.info("Connected to Databento WebSocket")
        
        except Exception as e:
            logger.error(f"Failed to connect to WebSocket: {e}")
            raise
    
    async def _handle_websocket_messages(self):
        """Handle incoming WebSocket messages"""
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    
                    # Route message to appropriate callback
                    for subscription_id, callback in self.callbacks.items():
                        # Simple routing based on subscription
                        if self._message_matches_subscription(data, subscription_id):
                            await asyncio.get_event_loop().run_in_executor(
                                self.executor, callback, data
                            )
                
                except Exception as e:
                    logger.error(f"Error processing WebSocket message: {e}")
        
        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket connection closed")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
    
    def _message_matches_subscription(self, message: Dict[str, Any], 
                                    subscription_id: str) -> bool:
        """Check if message matches subscription"""
        # Simple matching logic - can be enhanced
        return True
    
    async def unsubscribe(self, subscription_id: str):
        """Unsubscribe from data feed"""
        if subscription_id in self.subscriptions:
            del self.subscriptions[subscription_id]
            del self.callbacks[subscription_id]
            
            if self.mode != DatabentoMode.MOCK and self.websocket:
                # Send unsubscribe message
                request = self.subscriptions.get(subscription_id)
                if request:
                    unsubscribe_msg = {
                        "action": "unsubscribe",
                        "dataset": request.dataset,
                        "schema": request.schema,
                        "symbols": request.symbols
                    }
                    await self.websocket.send(json.dumps(unsubscribe_msg))
            
            logger.info(f"Unsubscribed from: {subscription_id}")
    
    def _parse_dbn_data(self, data: bytes) -> List[Dict[str, Any]]:
        """Parse Databento binary format (simplified)"""
        # This is a placeholder - real implementation would parse DBN format
        logger.warning("DBN parsing not implemented - returning empty data")
        return []
    
    async def get_market_status(self) -> Dict[str, Any]:
        """Get current market status"""
        if self.mode == DatabentoMode.MOCK:
            return self.mock_generator.generate_market_status()
        
        # Real API call for market status
        await self._create_session()
        self._check_rate_limit()
        
        try:
            # This would be the actual Databento market status endpoint
            url = f"{self.config.base_url}/v0/metadata.get_market_status"
            
            async with self.session.get(url) as response:
                response.raise_for_status()
                return await response.json()
        
        except Exception as e:
            logger.error(f"Failed to get market status: {e}")
            return {"is_open": False, "error": str(e)}
    
    async def start(self):
        """Start the client"""
        self.running = True
        
        if self.mode != DatabentoMode.MOCK:
            await self._create_session()
        
        logger.info(f"Databento client started in {self.mode.value} mode")
    
    async def stop(self):
        """Stop the client"""
        self.running = False
        
        # Clear subscriptions
        self.subscriptions.clear()
        self.callbacks.clear()
        
        # Close WebSocket
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
        
        # Close HTTP session
        await self._close_session()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("Databento client stopped")
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get client usage statistics"""
        return {
            "mode": self.mode.value,
            "active_subscriptions": len(self.subscriptions),
            "subscription_ids": list(self.subscriptions.keys()),
            "requests_in_last_second": len([
                t for t in self.request_times 
                if time.time() - t < 1.0
            ]),
            "websocket_connected": self.websocket is not None,
            "running": self.running
        }

# Factory functions
def create_databento_client(api_key: str = None, mode: str = "mock", 
                          symbols: List[str] = None) -> DatabentoClient:
    """Create Databento client with configuration"""
    
    # Get API key from environment if not provided
    if not api_key:
        api_key = os.getenv("DATABENTO_API_KEY", "mock_api_key")
    
    # Parse mode
    try:
        client_mode = DatabentoMode(mode.lower())
    except ValueError:
        logger.warning(f"Invalid mode '{mode}', defaulting to mock")
        client_mode = DatabentoMode.MOCK
    
    config = DatabentoConfig(
        api_key=api_key,
        mode=client_mode,
        mock_symbols=symbols or ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA", "META"]
    )
    
    return DatabentoClient(config)

def create_mock_client(symbols: List[str] = None, 
                      update_interval: float = 1.0) -> DatabentoClient:
    """Create mock Databento client for testing"""
    config = DatabentoConfig(
        api_key="mock_api_key",
        mode=DatabentoMode.MOCK,
        mock_symbols=symbols,
        mock_update_interval=update_interval
    )
    
    return DatabentoClient(config)

# Example usage
async def example_usage():
    """Example of how to use the Databento client"""
    
    # Create mock client
    client = create_mock_client(symbols=["AAPL", "GOOGL"])
    
    try:
        # Start client
        await client.start()
        
        # Get instruments
        instruments = await client.get_instruments("XNAS.ITCH", ["AAPL", "GOOGL"])
        print(f"Found {len(instruments)} instruments")
        
        # Get historical data
        request = SubscriptionRequest(
            dataset="XNAS.ITCH",
            schema="trades",
            symbols=["AAPL"],
            start="2023-12-01",
            end="2023-12-01"
        )
        
        historical_data = await client.get_historical_data(request)
        print(f"Retrieved {len(historical_data)} historical records")
        
        # Subscribe to live data
        def data_callback(data):
            print(f"Received live data: {data.get('symbol')} @ {data.get('price', 'N/A')}")
        
        await client.subscribe_live(request, data_callback)
        
        # Let it run for a few seconds
        await asyncio.sleep(5)
        
        # Get market status
        market_status = await client.get_market_status()
        print(f"Market status: {market_status}")
        
        # Get usage stats
        stats = client.get_usage_stats()
        print(f"Client stats: {stats}")
    
    finally:
        # Stop client
        await client.stop()

if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())

