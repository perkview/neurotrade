import time
import hmac
import hashlib
import requests
from urllib.parse import urlencode

# 🔑 Hardcoded API keys (replace with your real keys)
API_KEY = "mx0vglZPIwylJNCEnD"
API_SECRET = "a45a612799654559b497adcdc34e1cca".encode()

BASE_URL = "https://api.mexc.com/api/v3"

def sign_request(params: dict) -> dict:
    """
    Add signature using HMAC-SHA256.
    """
    query = urlencode(params)
    signature = hmac.new(API_SECRET, query.encode("utf-8"), hashlib.sha256).hexdigest()
    params["signature"] = signature
    return params

def get_account_info():
    """
    Fetch spot account balances.
    """
    url = BASE_URL + "/account"
    params = {
        "timestamp": int(time.time() * 1000),
        "recvWindow": 5000
    }
    signed = sign_request(params)
    headers = {"X-MEXC-APIKEY": API_KEY}
    
    resp = requests.get(url, headers=headers, params=signed)
    if resp.status_code == 200:
        balances = resp.json().get("balances", [])
        non_zero = [b for b in balances if float(b["free"]) > 0 or float(b["locked"]) > 0]
        for bal in non_zero:
            print(f"{bal['asset']}: Free={bal['free']}, Locked={bal['locked']}")
        return balances
    else:
        print("❌ Error fetching balances:", resp.status_code, resp.text)
        return None

def place_market_buy(symbol: str, usdt_amount: float):
    """
    Place a MARKET BUY order using quoteOrderQty (spend USDT).
    """
    url = BASE_URL + "/order"
    params = {
        "symbol": symbol,
        "side": "BUY",
        "type": "MARKET",
        "quoteOrderQty": usdt_amount,  # how much USDT to spend
        "timestamp": int(time.time() * 1000),
        "recvWindow": 5000
    }
    signed = sign_request(params)
    headers = {"X-MEXC-APIKEY": API_KEY}
    
    resp = requests.post(url, headers=headers, params=signed)
    if resp.status_code == 200:
        print("✅ Order placed:", resp.json())
        return resp.json()
    else:
        print("❌ Error placing order:", resp.status_code, resp.text)
        return None

if __name__ == "__main__":
    print("📊 Balances before order:")
    get_account_info()
    
    print("\n💰 Buying ADA with 3 USDT...")
    place_market_buy("ADAUSDT", 3)
    
    print("\n📊 Balances after order:")
    get_account_info()
