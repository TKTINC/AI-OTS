"""
Databento Mock Data Generator for AI Options Trading System
Generates realistic market data for development and testing
"""

import json
import random
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional
import numpy as np
from dataclasses import dataclass, asdict
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class InstrumentClass(Enum):
    """Databento instrument classes"""
    STOCK = "stock"
    OPTION = "option"
    FUTURE = "future"
    INDEX = "index"

class SType(Enum):
    """Databento schema types"""
    MBO = "mbo"  # Market by Order
    MBP_1 = "mbp-1"  # Market by Price (Level 1)
    MBP_10 = "mbp-10"  # Market by Price (Level 10)
    TBBO = "tbbo"  # Top of Book
    TRADES = "trades"
    OHLCV_1S = "ohlcv-1s"
    OHLCV_1M = "ohlcv-1m"
    OHLCV_1H = "ohlcv-1h"
    OHLCV_1D = "ohlcv-1d"

@dataclass
class MockInstrument:
    """Mock instrument definition"""
    instrument_id: int
    symbol: str
    raw_symbol: str
    publisher_id: int
    instrument_class: str
    exchange: str
    currency: str
    tick_size: float
    multiplier: int
    expiration: Optional[str] = None
    strike_price: Optional[float] = None
    option_type: Optional[str] = None
    underlying_symbol: Optional[str] = None

@dataclass
class MockTrade:
    """Mock trade record"""
    ts_event: int  # Nanoseconds since Unix epoch
    ts_recv: int   # Nanoseconds since Unix epoch
    instrument_id: int
    action: str    # 'T' for trade
    side: str      # 'A' for ask, 'B' for bid
    price: int     # Price in fixed-point (multiply by 1e-9)
    size: int      # Size in lots
    flags: int     # Trade flags
    sequence: int  # Sequence number

@dataclass
class MockQuote:
    """Mock quote record (TBBO)"""
    ts_event: int
    ts_recv: int
    instrument_id: int
    action: str    # 'R' for replace
    side: str      # 'A' for ask, 'B' for bid
    price: int     # Price in fixed-point
    size: int      # Size in lots
    flags: int
    sequence: int
    levels: List[Dict[str, Any]]  # Price levels

@dataclass
class MockOHLCV:
    """Mock OHLCV record"""
    ts_event: int
    ts_recv: int
    instrument_id: int
    open: int      # Open price in fixed-point
    high: int      # High price in fixed-point
    low: int       # Low price in fixed-point
    close: int     # Close price in fixed-point
    volume: int    # Volume

class MockDataGenerator:
    """Generates realistic mock market data"""
    
    def __init__(self, symbols: List[str] = None):
        """Initialize mock data generator"""
        self.symbols = symbols or [
            "AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA", "META"
        ]
        
        # Base prices for symbols (in dollars)
        self.base_prices = {
            "AAPL": 150.0,
            "GOOGL": 2800.0,
            "MSFT": 350.0,
            "AMZN": 3200.0,
            "TSLA": 800.0,
            "NVDA": 450.0,
            "META": 300.0
        }
        
        # Current prices (will fluctuate)
        self.current_prices = self.base_prices.copy()
        
        # Volatility for each symbol
        self.volatilities = {
            "AAPL": 0.25,
            "GOOGL": 0.30,
            "MSFT": 0.22,
            "AMZN": 0.35,
            "TSLA": 0.60,
            "NVDA": 0.45,
            "META": 0.40
        }
        
        # Instrument ID mapping
        self.instrument_ids = {symbol: i + 1 for i, symbol in enumerate(self.symbols)}
        
        # Options instrument IDs start from 1000
        self.next_option_id = 1000
        self.option_instruments = {}
        
        # Sequence numbers
        self.sequence_numbers = {symbol: 1 for symbol in self.symbols}
        
        # Market state
        self.market_open = True
        self.last_update = time.time()
        
        # Generate option instruments
        self._generate_option_instruments()
    
    def _generate_option_instruments(self):
        """Generate option instruments for each underlying"""
        for symbol in self.symbols:
            base_price = self.base_prices[symbol]
            
            # Generate strikes around current price
            strikes = []
            for i in range(-10, 11):  # 21 strikes total
                if base_price < 100:
                    strike = base_price + (i * 5)  # $5 intervals
                elif base_price < 500:
                    strike = base_price + (i * 10)  # $10 intervals
                else:
                    strike = base_price + (i * 50)  # $50 intervals
                
                if strike > 0:
                    strikes.append(round(strike, 2))
            
            # Generate expiration dates (next 4 Fridays)
            expiries = []
            today = datetime.now()
            days_until_friday = (4 - today.weekday()) % 7
            if days_until_friday == 0:
                days_until_friday = 7  # Next Friday if today is Friday
            
            for i in range(4):
                expiry = today + timedelta(days=days_until_friday + (i * 7))
                expiries.append(expiry.strftime("%Y-%m-%d"))
            
            # Create option instruments
            for expiry in expiries:
                for strike in strikes:
                    for option_type in ["C", "P"]:  # Call and Put
                        option_symbol = f"{symbol}{expiry.replace('-', '')}{option_type}{int(strike*1000):08d}"
                        
                        instrument = MockInstrument(
                            instrument_id=self.next_option_id,
                            symbol=option_symbol,
                            raw_symbol=option_symbol,
                            publisher_id=1,
                            instrument_class="option",
                            exchange="OPRA",
                            currency="USD",
                            tick_size=0.01,
                            multiplier=100,
                            expiration=expiry,
                            strike_price=strike,
                            option_type="CALL" if option_type == "C" else "PUT",
                            underlying_symbol=symbol
                        )
                        
                        self.option_instruments[option_symbol] = instrument
                        self.sequence_numbers[option_symbol] = 1
                        self.next_option_id += 1
    
    def _price_to_fixed_point(self, price: float) -> int:
        """Convert price to fixed-point representation (multiply by 1e9)"""
        return int(price * 1e9)
    
    def _fixed_point_to_price(self, fixed_point: int) -> float:
        """Convert fixed-point to price (divide by 1e9)"""
        return fixed_point / 1e9
    
    def _get_current_timestamp(self) -> int:
        """Get current timestamp in nanoseconds"""
        return int(time.time() * 1e9)
    
    def _update_stock_price(self, symbol: str) -> float:
        """Update stock price using random walk"""
        current_price = self.current_prices[symbol]
        volatility = self.volatilities[symbol]
        
        # Random walk with mean reversion
        dt = 1.0 / (252 * 24 * 60)  # 1 minute in trading years
        drift = -0.1 * (current_price - self.base_prices[symbol]) / self.base_prices[symbol]
        
        random_change = np.random.normal(0, volatility * np.sqrt(dt))
        price_change = current_price * (drift * dt + random_change)
        
        new_price = max(current_price + price_change, 0.01)  # Minimum price $0.01
        self.current_prices[symbol] = new_price
        
        return new_price
    
    def _calculate_option_price(self, underlying_price: float, strike: float, 
                               option_type: str, time_to_expiry: float) -> float:
        """Calculate option price using simplified Black-Scholes"""
        # Simplified option pricing for mock data
        intrinsic_value = 0
        
        if option_type == "CALL":
            intrinsic_value = max(underlying_price - strike, 0)
        else:  # PUT
            intrinsic_value = max(strike - underlying_price, 0)
        
        # Time value (simplified)
        time_value = max(0.01, time_to_expiry * 0.1 * underlying_price * 0.01)
        
        # Add some randomness
        noise = random.uniform(0.9, 1.1)
        
        return max(0.01, (intrinsic_value + time_value) * noise)
    
    def generate_stock_instruments(self) -> List[Dict[str, Any]]:
        """Generate stock instrument definitions"""
        instruments = []
        
        for symbol in self.symbols:
            instrument = MockInstrument(
                instrument_id=self.instrument_ids[symbol],
                symbol=symbol,
                raw_symbol=symbol,
                publisher_id=1,
                instrument_class="stock",
                exchange="NASDAQ",
                currency="USD",
                tick_size=0.01,
                multiplier=1
            )
            instruments.append(asdict(instrument))
        
        return instruments
    
    def generate_option_instruments(self) -> List[Dict[str, Any]]:
        """Generate option instrument definitions"""
        return [asdict(instrument) for instrument in self.option_instruments.values()]
    
    def generate_stock_trades(self, symbol: str, count: int = 10) -> List[Dict[str, Any]]:
        """Generate mock stock trade data"""
        if symbol not in self.symbols:
            raise ValueError(f"Unknown symbol: {symbol}")
        
        trades = []
        current_time = self._get_current_timestamp()
        instrument_id = self.instrument_ids[symbol]
        
        for i in range(count):
            # Update price
            price = self._update_stock_price(symbol)
            
            # Generate trade
            trade = MockTrade(
                ts_event=current_time + (i * 1000000),  # 1ms apart
                ts_recv=current_time + (i * 1000000) + 100000,  # 100μs latency
                instrument_id=instrument_id,
                action='T',
                side=random.choice(['A', 'B']),
                price=self._price_to_fixed_point(price),
                size=random.randint(100, 10000),
                flags=0,
                sequence=self.sequence_numbers[symbol]
            )
            
            trades.append(asdict(trade))
            self.sequence_numbers[symbol] += 1
        
        return trades
    
    def generate_stock_quotes(self, symbol: str, count: int = 10) -> List[Dict[str, Any]]:
        """Generate mock stock quote data (TBBO)"""
        if symbol not in self.symbols:
            raise ValueError(f"Unknown symbol: {symbol}")
        
        quotes = []
        current_time = self._get_current_timestamp()
        instrument_id = self.instrument_ids[symbol]
        
        for i in range(count):
            # Update price
            mid_price = self._update_stock_price(symbol)
            
            # Generate bid/ask spread
            spread = random.uniform(0.01, 0.05)
            bid_price = mid_price - spread / 2
            ask_price = mid_price + spread / 2
            
            # Generate bid quote
            bid_quote = MockQuote(
                ts_event=current_time + (i * 2000000),  # 2ms apart
                ts_recv=current_time + (i * 2000000) + 50000,
                instrument_id=instrument_id,
                action='R',
                side='B',
                price=self._price_to_fixed_point(bid_price),
                size=random.randint(100, 5000),
                flags=0,
                sequence=self.sequence_numbers[symbol],
                levels=[{
                    'price': self._price_to_fixed_point(bid_price),
                    'size': random.randint(100, 5000),
                    'count': random.randint(1, 10)
                }]
            )
            
            # Generate ask quote
            ask_quote = MockQuote(
                ts_event=current_time + (i * 2000000) + 1000000,
                ts_recv=current_time + (i * 2000000) + 1050000,
                instrument_id=instrument_id,
                action='R',
                side='A',
                price=self._price_to_fixed_point(ask_price),
                size=random.randint(100, 5000),
                flags=0,
                sequence=self.sequence_numbers[symbol] + 1,
                levels=[{
                    'price': self._price_to_fixed_point(ask_price),
                    'size': random.randint(100, 5000),
                    'count': random.randint(1, 10)
                }]
            )
            
            quotes.extend([asdict(bid_quote), asdict(ask_quote)])
            self.sequence_numbers[symbol] += 2
        
        return quotes
    
    def generate_option_trades(self, underlying_symbol: str, count: int = 5) -> List[Dict[str, Any]]:
        """Generate mock option trade data"""
        if underlying_symbol not in self.symbols:
            raise ValueError(f"Unknown underlying symbol: {underlying_symbol}")
        
        trades = []
        current_time = self._get_current_timestamp()
        underlying_price = self.current_prices[underlying_symbol]
        
        # Select random options for this underlying
        underlying_options = [
            (symbol, instrument) for symbol, instrument in self.option_instruments.items()
            if instrument.underlying_symbol == underlying_symbol
        ]
        
        if not underlying_options:
            return trades
        
        selected_options = random.sample(
            underlying_options, 
            min(count, len(underlying_options))
        )
        
        for i, (option_symbol, option_instrument) in enumerate(selected_options):
            # Calculate time to expiry
            expiry_date = datetime.strptime(option_instrument.expiration, "%Y-%m-%d")
            time_to_expiry = max(0.01, (expiry_date - datetime.now()).days / 365.0)
            
            # Calculate option price
            option_price = self._calculate_option_price(
                underlying_price,
                option_instrument.strike_price,
                option_instrument.option_type,
                time_to_expiry
            )
            
            # Generate trade
            trade = MockTrade(
                ts_event=current_time + (i * 5000000),  # 5ms apart
                ts_recv=current_time + (i * 5000000) + 200000,  # 200μs latency
                instrument_id=option_instrument.instrument_id,
                action='T',
                side=random.choice(['A', 'B']),
                price=self._price_to_fixed_point(option_price),
                size=random.randint(1, 100),  # Options trade in smaller sizes
                flags=0,
                sequence=self.sequence_numbers[option_symbol]
            )
            
            trades.append(asdict(trade))
            self.sequence_numbers[option_symbol] += 1
        
        return trades
    
    def generate_ohlcv_data(self, symbol: str, interval: str = "1m", count: int = 100) -> List[Dict[str, Any]]:
        """Generate mock OHLCV data"""
        if symbol not in self.symbols:
            raise ValueError(f"Unknown symbol: {symbol}")
        
        ohlcv_data = []
        current_time = self._get_current_timestamp()
        
        # Interval mapping to nanoseconds
        interval_ns = {
            "1s": 1_000_000_000,
            "1m": 60_000_000_000,
            "1h": 3_600_000_000_000,
            "1d": 86_400_000_000_000
        }
        
        if interval not in interval_ns:
            raise ValueError(f"Unsupported interval: {interval}")
        
        interval_duration = interval_ns[interval]
        instrument_id = self.instrument_ids[symbol]
        
        # Start from count intervals ago
        start_time = current_time - (count * interval_duration)
        
        for i in range(count):
            # Generate OHLCV for this interval
            open_price = self._update_stock_price(symbol)
            
            # Generate high, low, close within reasonable bounds
            volatility = self.volatilities[symbol]
            price_range = open_price * volatility * 0.1  # 10% of daily volatility per interval
            
            high_price = open_price + random.uniform(0, price_range)
            low_price = open_price - random.uniform(0, price_range)
            close_price = random.uniform(low_price, high_price)
            
            # Ensure logical price relationships
            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)
            
            # Generate volume
            base_volume = 1000000 if interval == "1d" else 10000
            volume = random.randint(base_volume // 2, base_volume * 2)
            
            ohlcv = MockOHLCV(
                ts_event=start_time + (i * interval_duration),
                ts_recv=start_time + (i * interval_duration) + 100000,
                instrument_id=instrument_id,
                open=self._price_to_fixed_point(open_price),
                high=self._price_to_fixed_point(high_price),
                low=self._price_to_fixed_point(low_price),
                close=self._price_to_fixed_point(close_price),
                volume=volume
            )
            
            ohlcv_data.append(asdict(ohlcv))
            
            # Update current price for next iteration
            self.current_prices[symbol] = close_price
        
        return ohlcv_data
    
    def generate_market_status(self) -> Dict[str, Any]:
        """Generate mock market status"""
        now = datetime.now(timezone.utc)
        
        # Simple market hours: 9:30 AM - 4:00 PM ET (14:30 - 21:00 UTC)
        market_open_utc = now.replace(hour=14, minute=30, second=0, microsecond=0)
        market_close_utc = now.replace(hour=21, minute=0, second=0, microsecond=0)
        
        # Check if market is open
        is_open = market_open_utc <= now <= market_close_utc
        
        # Next market event
        if is_open:
            next_event = "close"
            next_time = market_close_utc
        else:
            if now < market_open_utc:
                next_event = "open"
                next_time = market_open_utc
            else:
                # After close, next open is tomorrow
                next_event = "open"
                next_time = market_open_utc + timedelta(days=1)
        
        return {
            "is_open": is_open,
            "session": "regular" if is_open else "closed",
            "next_event": next_event,
            "next_time": next_time.isoformat(),
            "timezone": "UTC",
            "timestamp": now.isoformat()
        }
    
    def generate_batch_data(self, symbols: List[str] = None, 
                           data_types: List[str] = None) -> Dict[str, Any]:
        """Generate a batch of mixed market data"""
        symbols = symbols or self.symbols
        data_types = data_types or ["trades", "quotes", "ohlcv"]
        
        batch_data = {
            "metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "symbols": symbols,
                "data_types": data_types,
                "generator": "mock"
            },
            "instruments": {
                "stocks": self.generate_stock_instruments(),
                "options": self.generate_option_instruments()
            },
            "market_data": {}
        }
        
        for symbol in symbols:
            symbol_data = {}
            
            if "trades" in data_types:
                symbol_data["trades"] = self.generate_stock_trades(symbol, count=20)
            
            if "quotes" in data_types:
                symbol_data["quotes"] = self.generate_stock_quotes(symbol, count=20)
            
            if "ohlcv" in data_types:
                symbol_data["ohlcv_1m"] = self.generate_ohlcv_data(symbol, "1m", count=60)
                symbol_data["ohlcv_1h"] = self.generate_ohlcv_data(symbol, "1h", count=24)
            
            if "options" in data_types:
                symbol_data["option_trades"] = self.generate_option_trades(symbol, count=10)
            
            batch_data["market_data"][symbol] = symbol_data
        
        # Add market status
        batch_data["market_status"] = self.generate_market_status()
        
        return batch_data
    
    def save_to_file(self, data: Dict[str, Any], filename: str):
        """Save generated data to JSON file"""
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Mock data saved to {filename}")

# Convenience functions for quick data generation
def generate_sample_data(symbols: List[str] = None, output_file: str = None) -> Dict[str, Any]:
    """Generate sample market data for testing"""
    generator = MockDataGenerator(symbols)
    data = generator.generate_batch_data()
    
    if output_file:
        generator.save_to_file(data, output_file)
    
    return data

def generate_realtime_feed(symbols: List[str] = None, duration_seconds: int = 60):
    """Generate real-time mock data feed"""
    generator = MockDataGenerator(symbols)
    
    start_time = time.time()
    while time.time() - start_time < duration_seconds:
        for symbol in generator.symbols:
            # Generate some trades
            trades = generator.generate_stock_trades(symbol, count=1)
            quotes = generator.generate_stock_quotes(symbol, count=1)
            
            yield {
                "symbol": symbol,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "trades": trades,
                "quotes": quotes
            }
        
        time.sleep(1)  # 1 second interval

if __name__ == "__main__":
    # Generate sample data
    print("Generating mock market data...")
    
    symbols = ["AAPL", "GOOGL", "MSFT"]
    data = generate_sample_data(symbols, "mock_market_data.json")
    
    print(f"Generated data for {len(symbols)} symbols")
    print(f"Stock instruments: {len(data['instruments']['stocks'])}")
    print(f"Option instruments: {len(data['instruments']['options'])}")
    
    for symbol in symbols:
        symbol_data = data['market_data'][symbol]
        print(f"\n{symbol}:")
        print(f"  Trades: {len(symbol_data.get('trades', []))}")
        print(f"  Quotes: {len(symbol_data.get('quotes', []))}")
        print(f"  OHLCV 1m: {len(symbol_data.get('ohlcv_1m', []))}")
        print(f"  Option trades: {len(symbol_data.get('option_trades', []))}")
    
    print(f"\nMarket status: {data['market_status']['is_open']}")
    print("Mock data generation complete!")

