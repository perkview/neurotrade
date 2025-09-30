# trade_bot.py — corrected and hardened for MEXC Spot
import time
import math
import hmac
import hashlib
import urllib.parse
import requests
import yfinance as yf
import pandas as pd
from datetime import timedelta
from django.utils import timezone
from dashboard.models import StrategySettings, TradeLog

# ---------------- CONFIGURATION ----------------
API_KEY = "mx0vglZPIwylJNCEnD"
API_SECRET = "a45a612799654559b497adcdc34e1cca"
BASE_URL = "https://api.mexc.com"


# Tweakable defaults
DEFAULT_QUANTITY = 0.01
MILD_MULTIPLIER = 0.5
BASE_QUOTE = "USDT"
FALLBACK_SLEEP = 30
EXCHANGE_INFO_CACHE_TTL = 300

# Symbol mapping: key -> (yahoo ticker, exchange symbol)
SYMBOL_MAP = {
    "BTC": ("BTC-USD", "BTC" + BASE_QUOTE),
    "ETH": ("ETH-USD", "ETH" + BASE_QUOTE),
    "BNB": ("BNB-USD", "BNB" + BASE_QUOTE),
    "ADA": ("ADA-USD", "ADA" + BASE_QUOTE),
    "SOL": ("SOL-USD", "SOL" + BASE_QUOTE),
    "XRP": ("XRP-USD", "XRP" + BASE_QUOTE),
    "AVAX": ("AVAX-USD", "AVAX" + BASE_QUOTE),
    "DOT": ("DOT-USD", "DOT" + BASE_QUOTE),
    "LINK": ("LINK-USD", "LINK" + BASE_QUOTE),
    "LTC": ("LTC-USD", "LTC" + BASE_QUOTE),
    "DOGE": ("DOGE-USD", "DOGE" + BASE_QUOTE),
    "MANA": ("MANA-USD", "MANA" + BASE_QUOTE),
    "SAND": ("SAND-USD", "SAND" + BASE_QUOTE),
    "CAKE": ("CAKE-USD", "CAKE" + BASE_QUOTE),
    "FIL": ("FIL-USD", "FIL" + BASE_QUOTE),
}

# Exchange info cache
_exchange_info_cache = {"timestamp": 0, "data": None}

# ---------------- Utilities ----------------
def _now_ts_ms():
    return int(time.time() * 1000)

def _sign_query(params: dict, secret: str):
    """
    Build canonical URL-encoded query string (sorted keys) and sign with HMAC-SHA256.
    Excludes None/empty values to avoid signature mismatches.
    Returns (signature_hex, query_string)
    """
    items = [(k, str(v)) for k, v in params.items() if v is not None and str(v) != ""]
    ordered_pairs = sorted(items, key=lambda x: x[0])
    query_string = urllib.parse.urlencode(ordered_pairs)
    signature = hmac.new(secret.encode(), query_string.encode(), hashlib.sha256).hexdigest()
    return signature, query_string

# ---------------- Strategy Settings ----------------
def get_strategy_settings():
    try:
        settings = StrategySettings.objects.latest("created_at")
    except StrategySettings.DoesNotExist:
        return None

    return {
        "coins": settings.coins,
        "period": settings.period,
        "interval": settings.interval,
        "ema_period": settings.ema_period,
        "rsi_period": settings.rsi_period,
        "volume_ma_period": settings.volume_ma_period,
        "volume_spike_multiplier": settings.volume_spike_multiplier,
        "buy_rsi_low": settings.buy_rsi_low,
        "sell_rsi_low": settings.sell_rsi_low,
        "buy_rsi_high": settings.buy_rsi_high,
        "sell_rsi_high": settings.sell_rsi_high,
        "use_volume": settings.use_volume,
        "ema_buy_buffer": settings.ema_buffer_buy,
        "ema_sell_buffer": settings.ema_buffer_sell,
        "check_interval_seconds": settings.check_interval_seconds,
        "active": settings.active,
        "low_vol_coins": settings.low_vol_coins,
        "high_vol_coins": settings.high_vol_coins,
        "mild_rsi_buffer_low": settings.mild_rsi_buffer_low,
        "mild_rsi_buffer_high": settings.mild_rsi_buffer_high,
        "confidence_threshold": getattr(settings, "confidence_threshold", 0.6),
    }

# ---------------- Data fetch ----------------
def fetch_ohlcv_for(coin_key: str, period: str, interval: str):
    """
    Primary OHLCV: yfinance. If yfinance fails for a coin, return None.
    (For real-time execution price we use get_symbol_price, see below.)
    """
    mapping = SYMBOL_MAP.get(coin_key.upper())
    if not mapping:
        print(f"⚠️ No symbol mapping for {coin_key}; skipping")
        return None
    yahoo_ticker = mapping[0]
    try:
        df = yf.Ticker(yahoo_ticker).history(period=period, interval=interval)
        if df is None or df.empty:
            # yfinance returned nothing
            print(f"${yahoo_ticker}: possibly delisted or no price data found (period={period})")
            return None
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as exc:
        print(f"Error fetching {yahoo_ticker} for {coin_key}: {exc}")
        return None

def get_symbol_price(mexc_symbol: str):
    """
    Public endpoint: get current price from MEXC. Falls back to 0.0 on failure.
    Endpoint: GET /api/v3/ticker/price?symbol=BTCUSDT
    """
    try:
        url = f"{BASE_URL}/api/v3/ticker/price"
        resp = requests.get(url, params={"symbol": mexc_symbol}, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        price = float(data.get("price", 0.0))
        return price
    except Exception as exc:
        print(f"⚠️ Failed to fetch ticker price for {mexc_symbol}: {exc}")
        return 0.0

# ---------------- Indicators ----------------
def ema(series, period):
    return series.ewm(span=period, adjust=False, min_periods=period).mean()

def rsi(series, period):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def volume_spike(volume, period, multiplier, rsi=None, confirm_with_rsi=True):
    vol_ma = volume.rolling(window=period, min_periods=1).mean()
    spike = (volume > vol_ma * multiplier)
    if confirm_with_rsi and rsi is not None:
        spike = spike & (rsi.diff() > 0)
    return spike.astype(float)

# ---------------- Strategy ----------------
def combined_strategy(df, params, coin_name):
    """Return DataFrame with signals and confidence based on EMA/RSI/Volume rules."""
    df = df.copy()
    close = df["Close"]
    df["EMA"] = ema(close, params["ema_period"]).ffill()
    df["RSI"] = rsi(close, params["rsi_period"]).ffill()

    if params.get("use_volume", True):
        df["Vol_Spike"] = volume_spike(
            volume=df["Volume"],
            period=params["volume_ma_period"],
            multiplier=params["volume_spike_multiplier"],
            rsi=df["RSI"],
            confirm_with_rsi=params.get("confirm_volume_with_rsi", True),
        ).ffill().fillna(0.0)
    else:
        df["Vol_Spike"] = 1.0

    ema_buy_level = df["EMA"] * (1 + params["ema_buy_buffer"])
    ema_sell_level = df["EMA"] * (1 - params["ema_sell_buffer"])

    strict_buy = ((close > ema_buy_level) | (df["RSI"] <= params["buy_rsi_threshold"])) & (df["Vol_Spike"] == 1.0)
    strict_sell = ((close < ema_sell_level) | (df["RSI"] >= params["sell_rsi_threshold"]))

    mild_buffer = params.get("mild_rsi_buffer", 2.0)
    mild_buy = ((close > df["EMA"]) & (df["RSI"] < (params["buy_rsi_threshold"] + mild_buffer))) | (
        (df["RSI"] < params["buy_rsi_threshold"]) & (df["Vol_Spike"] == 1.0)
    )
    mild_sell = ((close < df["EMA"]) & (df["RSI"] > (params["sell_rsi_threshold"] - mild_buffer))) | (df["RSI"] > params["sell_rsi_threshold"])

    df["Signal"] = 0.0
    df.loc[strict_buy, "Signal"] = 1.0
    df.loc[strict_sell, "Signal"] = -1.0
    df.loc[mild_buy & (df["Signal"] == 0.0), "Signal"] = 0.5
    df.loc[mild_sell & (df["Signal"] == 0.0), "Signal"] = -0.5

    buy_thresh = params["buy_rsi_threshold"]
    df["RSI_score"] = (1.0 - (abs(df["RSI"] - buy_thresh) / 100.0)).clip(0, 1)
    df["EMA_score"] = (1.0 - (abs(close - df["EMA"]) / close)).clip(0, 1)
    df["Vol_score"] = df["Vol_Spike"].clip(0, 1)

    df["Confidence"] = (0.5 * df["RSI_score"] + 0.3 * df["EMA_score"] + 0.2 * df["Vol_score"])
    confidence_threshold = params.get("confidence_threshold", 0.6)
    df.loc[(df["Signal"] == 1.0) & (df["Confidence"] < confidence_threshold), "Signal"] = 0.5
    df.loc[df["Confidence"] < 0.1, "Signal"] = 0.0

    df["Signal_Label"] = df["Signal"].map({1.0: "BUY", 0.5: "MILD_BUY", 0.0: "HOLD", -0.5: "MILD_SELL", -1.0: "SELL"})
    return df

# ---------------- Exchange helpers ----------------
def get_exchange_info(force_refresh=False):
    now_ts = time.time()
    if not force_refresh and _exchange_info_cache["data"] and (now_ts - _exchange_info_cache["timestamp"] < EXCHANGE_INFO_CACHE_TTL):
        return _exchange_info_cache["data"]
    try:
        url = BASE_URL + "/api/v3/exchangeInfo"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        _exchange_info_cache["data"] = data
        _exchange_info_cache["timestamp"] = now_ts
        return data
    except Exception as exc:
        print(f"Warning: failed to fetch exchange info: {exc}")
        return None

def _adjust_qty_to_step(symbol, qty, exchange_info):
    if not exchange_info:
        return float(round(qty, 8))
    symbols = exchange_info.get("symbols") if isinstance(exchange_info, dict) else None
    if not symbols:
        return float(round(qty, 8))
    info = next((s for s in symbols if s.get("symbol") == symbol), None)
    if not info:
        return float(round(qty, 8))
    step, min_qty = None, None
    for f in info.get("filters", []):
        if f.get("filterType") == "LOT_SIZE":
            step = float(f.get("stepSize", 0))
            min_qty = float(f.get("minQty", 0))
            break
    if not step or step == 0:
        return float(round(qty, 8))
    decimals = max(0, int(round(-math.log10(step)))) if step < 1 else 0
    adj = (math.floor(qty / step) * step)
    if adj <= 0:
        adj = step
    if min_qty and adj < min_qty:
        adj = min_qty
    return float(round(adj, decimals))

# ---------------- Orders (fixed for MEXC Spot JSON API) ----------------
def _post_order(coin_symbol, quantity, side="BUY", ord_type="MARKET", price=None):
    """
    Place a signed order on MEXC Spot.

    Parameters:
        coin_symbol (str): Symbol like 'BTCUSDT'
        quantity (float): Amount to buy/sell
        side (str): 'BUY' or 'SELL'
        ord_type (str): 'MARKET' or 'LIMIT'
        price (float, optional): Required for LIMIT orders

    Returns:
        dict: JSON response from MEXC or {'error': '...'}
    """
    if not API_KEY or not API_SECRET:
        return {"error": "API_KEY/API_SECRET not set"}

    url = f"{BASE_URL}/api/v3/order"
    timestamp = _now_ts_ms()
    
    # Prepare params as strings with enough precision
    params = {
        "symbol": coin_symbol.upper(),
        "side": side.upper(),
        "type": ord_type.upper(),
        "quantity": f"{quantity:.8f}",
        "timestamp": timestamp,
        "recvWindow": 5000,
    }

    if ord_type.upper() == "LIMIT":
        if price is None or price <= 0:
            return {"error": "LIMIT orders require a valid price"}
        params["price"] = f"{price:.8f}"

    # Sign the query
    signature, query_string = _sign_query(params, API_SECRET)
    params["signature"] = signature

    # Explicitly set Content-Type to fix 700013 error
    headers = {
        "X-MEXC-APIKEY": API_KEY,
        "Content-Type": "application/x-www-form-urlencoded"
    }

    try:
        # Send POST as form-encoded
        resp = requests.post(url, data=params, headers=headers, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except requests.HTTPError:
        try:
            return resp.json()
        except Exception as exc:
            return {"error": f"HTTP error + failed JSON parse: {exc}"}
    except requests.Timeout:
        return {"error": "Timeout while placing order"}
    except requests.RequestException as exc:
        return {"error": f"RequestException: {exc}"}
    except Exception as exc:
        return {"error": f"Unexpected error: {exc}"}



def buy_coin(coin_symbol, quantity, ord_type="MARKET", price=None):
    return _post_order(coin_symbol, quantity, side="BUY", ord_type=ord_type, price=price)

def sell_coin(coin_symbol, quantity, ord_type="MARKET", price=None):
    return _post_order(coin_symbol, quantity, side="SELL", ord_type=ord_type, price=price)



def buy_coin(coin_symbol, quantity, ord_type="MARKET", price=None):
    return _post_order(coin_symbol, quantity, side="BUY", ord_type=ord_type, price=price)

def sell_coin(coin_symbol, quantity, ord_type="MARKET", price=None):
    return _post_order(coin_symbol, quantity, side="SELL", ord_type=ord_type, price=price)

# ---------------- Logging and position utils ----------------
def append_trade_safe(**kwargs):
    try:
        # Get the singleton TradeLog instance (or create one if none exists)
        trade_log, _ = TradeLog.objects.get_or_create(id=1)
        trade_log.append_trade(
            coin=kwargs.get("coin"),
            action=kwargs.get("action"),
            quantity=kwargs.get("quantity"),
            entry_price=kwargs.get("entry_price") or kwargs.get("order_price"),
            exit_price=kwargs.get("exit_price"),
            pnl=kwargs.get("pnl"),
            status=kwargs.get("status", "OPEN"),
            ema200=kwargs.get("ema200"),
            rsi=kwargs.get("rsi"),
            volume_spike=kwargs.get("volume_spike", False),
            order_symbol=kwargs.get("order_symbol"),
            trade_type=kwargs.get("trade_type"),
            order_type=kwargs.get("order_type"),
            order_price=kwargs.get("order_price"),
            timestamp_ms=kwargs.get("timestamp_ms"),
            signature=kwargs.get("signature"),
            next_check=kwargs.get("next_check"),
            actions=kwargs.get("actions"),
        )
    except Exception as exc:
        print(f"Warning: failed to append trade to DB: {exc}")

def has_open_position(coin):
    try:
        return TradeLog.objects.filter(coin=coin, status="OPEN").exists()
    except Exception:
        return False

def get_open_positions(coin):
    try:
        return TradeLog.objects.filter(coin=coin, status="OPEN").order_by("id")
    except Exception:
        return []

def close_open_positions(coin, exit_price, qty_sold, order_response):
    try:
        remaining = float(qty_sold)
        open_positions = get_open_positions(coin)
        for pos in open_positions:
            pos_qty = float(pos.quantity or 0)
            if pos_qty <= 0:
                append_trade_safe(
                    coin=coin,
                    action="SELL",
                    exit_price=exit_price,
                    status="CLOSED",
                    order_symbol=pos.order_symbol,
                    order_price=exit_price,
                    timestamp_ms=_now_ts_ms(),
                    actions=order_response,
                )
                continue

            if remaining >= pos_qty:
                append_trade_safe(
                    coin=coin,
                    action="SELL",
                    exit_price=exit_price,
                    quantity=pos_qty,
                    status="CLOSED",
                    order_symbol=pos.order_symbol,
                    order_price=exit_price,
                    timestamp_ms=_now_ts_ms(),
                    actions=order_response,
                )
                remaining -= pos_qty
            else:
                append_trade_safe(
                    coin=coin,
                    action="SELL",
                    exit_price=exit_price,
                    quantity=remaining,
                    status="CLOSED",
                    order_symbol=pos.order_symbol,
                    order_price=exit_price,
                    timestamp_ms=_now_ts_ms(),
                    actions=order_response,
                )
                remaining = 0

            if remaining <= 0:
                break

        # If any remaining quantity not accounted for
        if remaining > 0:
            append_trade_safe(
                coin=coin,
                action="SELL",
                exit_price=exit_price,
                quantity=remaining,
                status="CLOSED",
                order_symbol=coin + BASE_QUOTE,
                order_price=exit_price,
                timestamp_ms=_now_ts_ms(),
                actions=order_response,
            )

    except Exception as exc:
        print(f"Warning: failed to update/close open positions for {coin}: {exc}")


# ---------------- Account & Balance ----------------
def get_account_balance(asset="USDT"):
    if not API_KEY or not API_SECRET:
        print("⚠️ API_KEY/API_SECRET not set")
        return 0.0

    path = "/api/v3/account"
    url = BASE_URL + path
    timestamp = _now_ts_ms()
    params = {"timestamp": timestamp, "recvWindow": 5000}  # safe value
    signature, query_string = _sign_query(params, API_SECRET)
    query_string += f"&signature={signature}"  # append signature
    headers = {"X-MEXC-APIKEY": API_KEY}

    try:
        resp = requests.get(url, params=query_string, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        for bal in data.get("balances", []):
            if bal.get("asset", "").upper() == asset.upper():
                return float(bal.get("free", 0.0))
        return 0.0
    except requests.RequestException as e:
        print(f"⚠️ HTTP error while fetching balance for {asset}: {e}")
        return 0.0



def calculate_trade_qty(symbol, price, percent=3.0, asset="USDT"):
    """
    Calculate a safe trade quantity for MEXC Spot.
    
    Automatically adjusts for:
    - Low balances
    - Minimum order size
    - Step size increments

    Parameters:
        symbol (str): Symbol like 'BTCUSDT'
        price (float): Current price of the coin
        percent (float): % of balance to use
        asset (str): Quote asset, e.g., USDT

    Returns:
        float: Adjusted quantity to trade
    """
    usdt_balance = get_account_balance(asset)
    if usdt_balance <= 0 or price <= 0:
        return 0.0

    # Calculate intended spend amount in USDT
    spend_amount = usdt_balance * (percent / 100.0)
    qty = spend_amount / price

    # Fetch exchange info to get LOT_SIZE filter
    exchange_info = get_exchange_info()
    if exchange_info:
        symbols_info = exchange_info.get("symbols", [])
        info = next((s for s in symbols_info if s.get("symbol") == symbol), None)
        if info:
            # Get LOT_SIZE step and min qty
            step, min_qty = None, None
            for f in info.get("filters", []):
                if f.get("filterType") == "LOT_SIZE":
                    step = float(f.get("stepSize", 0))
                    min_qty = float(f.get("minQty", 0))
                    break
            if step and step > 0:
                # Round down to nearest step
                decimals = max(0, int(round(-math.log10(step)))) if step < 1 else 0
                qty = math.floor(qty / step) * step
                if qty < min_qty:
                    # If qty below min, use min_qty if balance allows
                    qty = min_qty if (min_qty * price) <= usdt_balance else 0.0
                qty = round(qty, decimals)
            else:
                qty = round(qty, 8)
        else:
            qty = round(qty, 8)
    else:
        qty = round(qty, 8)

    # Ensure we never try to buy more than we can afford
    if qty * price > usdt_balance:
        qty = math.floor(usdt_balance / price * 1e8) / 1e8  # Round down safely

    return qty




# ---------------- Signal helper (integrates combined_strategy) ----------------
def check_signal(coin):
    """
    Returns (action, confidence, price).
    For price: prefer MEXC ticker price (real-time). For indicators we still use yfinance OHLC.
    """
    params = get_strategy_settings()
    if not params:
        return "HOLD", 0.0, 0.0

    # Get OHLCV for indicators (yfinance)
    df = fetch_ohlcv_for(coin, params["period"], params["interval"])
    if df is None or df.empty:
        # No OHLCV available: still try to get real-time price from exchange
        mapping = SYMBOL_MAP.get(coin)
        if mapping:
            mexc_symbol = mapping[1]
            price = get_symbol_price(mexc_symbol)
            return "HOLD", 0.0, price
        return "HOLD", 0.0, 0.0

    df_res = combined_strategy(
        df,
        {
            "ema_period": params["ema_period"],
            "rsi_period": params["rsi_period"],
            "volume_ma_period": params["volume_ma_period"],
            "volume_spike_multiplier": params["volume_spike_multiplier"],
            "use_volume": params["use_volume"],
            "ema_buy_buffer": params["ema_buy_buffer"],
            "ema_sell_buffer": params["ema_sell_buffer"],
            "buy_rsi_threshold": params["buy_rsi_low"],
            "sell_rsi_threshold": params["sell_rsi_high"],
            "mild_rsi_buffer": params["mild_rsi_buffer_low"],
            "confidence_threshold": params["confidence_threshold"],
        },
        coin,
    )

    last = df_res.iloc[-1]
    action = last["Signal_Label"]
    # prefer live exchange price to avoid yfinance zero problems
    mapping = SYMBOL_MAP.get(coin)
    mexc_symbol = mapping[1] if mapping else None
    live_price = get_symbol_price(mexc_symbol) if mexc_symbol else 0.0
    price = float(last["Close"]) if float(last["Close"]) > 0 else live_price
    # final fallback to live_price if close==0
    if price <= 0:
        price = live_price
    confidence = float(last.get("Confidence", 0.0))
    return action, confidence, price

def _symbols_from_settings_or_default(params):
    coins_raw = params.get("coins") if params else None
    if isinstance(coins_raw, dict):
        return [k.upper() for k in coins_raw.keys()]
    if isinstance(coins_raw, str):
        return [c.strip().upper() for c in coins_raw.split(",") if c.strip()]
    if isinstance(coins_raw, list):
        return [c.strip().upper() for c in coins_raw]
    return ["BTC", "ETH", "SOL"]

# ---------------- Main Loop ----------------
def main_loop():
    print("🚀 Starting Trading Bot (MEXC 3% Balance Strategy)...\n")
    while True:
        try:
            params = get_strategy_settings()
            if not params or not params.get("active", False):
                print("⚠️ No active strategy. Sleeping...")
                time.sleep(FALLBACK_SLEEP)
                continue

            symbols = _symbols_from_settings_or_default(params)
            exchange_info = get_exchange_info()

            for coin in symbols:
                try:
                    print(f"\n🔎 Checking {coin} ...")
                    action, confidence, price = check_signal(coin)
                    if price <= 0:
                        print(f"⚠️ {coin}: invalid price ({price}) — skipping")
                        continue

                    mapping = SYMBOL_MAP.get(coin)
                    if not mapping:
                        print(f"⚠️ No symbol mapping for {coin}; skipping trading")
                        continue
                    mexc_symbol = mapping[1]

                    if action == "BUY":
                        qty = calculate_trade_qty(mexc_symbol, price, percent=3.0)
                        if qty > 0:
                            resp = buy_coin(mexc_symbol, qty, ord_type="MARKET")
                            if resp and (resp.get("orderId") or resp.get("code") == 0):
                                print(f"✅ {coin}: BUY executed @ {price} | Qty={qty} | Conf={confidence:.2f} | Resp={resp}")
                                append_trade_safe(
                                    coin=coin, action="BUY", quantity=qty, entry_price=price, status="OPEN",
                                    order_symbol=mexc_symbol, trade_type="BUY", order_type="MARKET", order_price=price,
                                    timestamp_ms=_now_ts_ms(), actions=resp,
                                    next_check=timezone.now() + timedelta(seconds=params.get("check_interval_seconds", 900))
                                )
                            else:
                                print(f"⚠️ {coin}: BUY failed | Resp={resp}")

                    elif action == "MILD_BUY":
                        qty = calculate_trade_qty(mexc_symbol, price, percent=1.5)
                        if qty > 0:
                            resp = buy_coin(mexc_symbol, qty, ord_type="MARKET")
                            if resp and (resp.get("orderId") or resp.get("code") == 0):
                                print(f"🤏 {coin}: MILD_BUY executed @ {price} | Qty={qty} | Conf={confidence:.2f} | Resp={resp}")
                                append_trade_safe(
                                    coin=coin, action="MILD_BUY", quantity=qty, entry_price=price, status="OPEN",
                                    order_symbol=mexc_symbol, trade_type="MILD_BUY", order_type="MARKET",
                                    order_price=price, timestamp_ms=_now_ts_ms(), actions=resp,
                                    next_check=timezone.now() + timedelta(seconds=params.get("check_interval_seconds", 900))
                                )
                            else:
                                print(f"⚠️ {coin}: MILD_BUY failed | Resp={resp}")

                    elif action == "SELL":
                        qty = calculate_trade_qty(mexc_symbol, price, percent=3.0)
                        if qty > 0:
                            resp = sell_coin(mexc_symbol, qty, ord_type="MARKET")
                            if resp and (resp.get("orderId") or resp.get("code") == 0):
                                close_open_positions(coin, exit_price=price, qty_sold=qty, order_response=resp)
                                print(f"✅ {coin}: SELL executed @ {price} | Qty={qty} | Conf={confidence:.2f} | Resp={resp}")
                                append_trade_safe(
                                    coin=coin, action="SELL", quantity=qty, exit_price=price, status="CLOSED",
                                    order_symbol=mexc_symbol, trade_type="SELL", order_type="MARKET",
                                    order_price=price, timestamp_ms=_now_ts_ms(), actions=resp,
                                    next_check=timezone.now() + timedelta(seconds=params.get("check_interval_seconds", 900))
                                )
                            else:
                                print(f"⚠️ {coin}: SELL failed | Resp={resp}")

                    elif action == "MILD_SELL":
                        qty = calculate_trade_qty(mexc_symbol, price, percent=1.5)
                        if qty > 0:
                            resp = sell_coin(mexc_symbol, qty, ord_type="MARKET")
                            if resp and (resp.get("orderId") or resp.get("code") == 0):
                                close_open_positions(coin, exit_price=price, qty_sold=qty, order_response=resp)
                                print(f"🤏 {coin}: MILD_SELL executed @ {price} | Qty={qty} | Conf={confidence:.2f} | Resp={resp}")
                                append_trade_safe(
                                    coin=coin, action="MILD_SELL", quantity=qty, exit_price=price, status="CLOSED",
                                    order_symbol=mexc_symbol, trade_type="MILD_SELL", order_type="MARKET",
                                    order_price=price, timestamp_ms=_now_ts_ms(), actions=resp,
                                    next_check=timezone.now() + timedelta(seconds=params.get("check_interval_seconds", 900))
                                )
                            else:
                                print(f"⚠️ {coin}: MILD_SELL failed | Resp={resp}")

                    else:
                        print(f"⏸️ {coin}: HOLD @ {price} | Conf={confidence:.2f}")

                except Exception as exc_coin:
                    print(f"⚠️ Error while processing {coin}: {exc_coin}")
                    continue

            sleep_for = int(params.get("check_interval_seconds", 900))
            print(f"\n⏳ Sleeping {sleep_for}s...\n")
            time.sleep(sleep_for)

        except Exception as e:
            print(f"❌ Error in main_loop: {e}")
            time.sleep(10)

# Run directly (for debug only)
if __name__ == "__main__":
    main_loop()
