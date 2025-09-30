from django.db import models
from django.utils import timezone

# Create your models here.
class BotStatus(models.Model):
    id = models.IntegerField(primary_key=True, default=1, editable=False)
    is_running = models.BooleanField(default=False)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return "Running" if self.is_running else "Stopped"
    


class CoinPrediction(models.Model):
    log = models.TextField(default='', blank=True)  # All predictions stored as text here
    updated_at = models.DateTimeField(auto_now=True)

    def append_prediction(self, coin, signal, price, ema200=None, rsi=None, volume_spike=False, next_check=None):
        next_check = next_check or timezone.now()
        timestamp = timezone.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = (
            f"[{timestamp}] {coin} - {signal} @ {price} | "
            f"EMA200: {ema200 or 'N/A'}, RSI: {rsi or 'N/A'}, "
            f"Volume Spike: {'Yes' if volume_spike else 'No'}, "
            f"Next Check: {next_check.strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        self.log += log_entry
        self.save()

    def __str__(self):
        return f"Prediction Log (Last updated: {self.updated_at.strftime('%Y-%m-%d %H:%M:%S')})"



class StrategySettings(models.Model):
    # ---------------- Coins ----------------
    coins = models.JSONField(default=dict)
    low_vol_coins = models.JSONField(default=list, help_text='List of low volatility coins (e.g., ["BTC", "ETH"])')
    high_vol_coins = models.JSONField(default=list, help_text='List of high volatility coins (e.g., ["DOGE", "MANA"])')

    # ---------------- General Settings ----------------
    period = models.CharField(max_length=20, default="60d", help_text="Yahoo Finance period (e.g., 60d)")
    interval = models.CharField(max_length=10, default="15m", help_text="Yahoo Finance interval (e.g., 15m, 1h)")
    check_interval_seconds = models.IntegerField(default=900, help_text="Bot check interval in seconds")
    active = models.BooleanField(default=True)

    # ---------------- Indicators ----------------
    ema_period = models.IntegerField(default=50)
    rsi_period = models.IntegerField(default=14)
    volume_ma_period = models.IntegerField(default=20)
    volume_spike_multiplier = models.FloatField(default=1.0)
    use_volume = models.BooleanField(default=True)

    # ---------------- RSI Thresholds ----------------
    buy_rsi_low = models.IntegerField(default=53, help_text="Low Volatility - Buy RSI threshold")
    sell_rsi_low = models.IntegerField(default=61, help_text="Low Volatility - Sell RSI threshold")
    buy_rsi_high = models.IntegerField(default=55, help_text="High Volatility - Buy RSI threshold")
    sell_rsi_high = models.IntegerField(default=67, help_text="High Volatility - Sell RSI threshold")

    # ---------------- EMA Buffers ----------------
    ema_buffer_buy = models.FloatField(default=0.003, help_text="EMA Buy Buffer (%) e.g., 0.005 = 0.5%")
    ema_buffer_sell = models.FloatField(default=0.003, help_text="EMA Sell Buffer (%) e.g., 0.002 = 0.2%")

    # ---------------- Mild RSI Buffers ----------------
    mild_rsi_buffer_low = models.FloatField(default=8.0, help_text="Soft signal buffer for low-volatility coins")
    mild_rsi_buffer_high = models.FloatField(default=10.0, help_text="Soft signal buffer for high-volatility coins")

    # ---------------- Confidence Settings ----------------
    confidence_threshold = models.FloatField(default=0.6, help_text="Minimum confidence for strict BUY/SELL")
    ultra_low_conf_threshold = models.FloatField(default=0.3, help_text="Force HOLD below this confidence")
    confidence_rsi_weight = models.FloatField(default=0.5, help_text="Weight for RSI in confidence score")
    confidence_ema_weight = models.FloatField(default=0.3, help_text="Weight for EMA in confidence score")
    confidence_vol_weight = models.FloatField(default=0.2, help_text="Weight for Volume in confidence score")

    # ---------------- Volatility & Adaptive Tuning ----------------
    volatility_window = models.IntegerField(default=20, help_text="Rolling window for volatility calculation")
    vol_adjust_low = models.FloatField(default=0.3, help_text="Low-volatility RSI adjustment multiplier")
    vol_adjust_high = models.FloatField(default=0.5, help_text="High-volatility RSI adjustment multiplier")
    min_buy_rsi = models.IntegerField(default=25, help_text="Hard floor for Buy RSI")
    max_buy_rsi = models.IntegerField(default=60, help_text="Hard ceiling for Buy RSI")
    min_sell_rsi = models.IntegerField(default=40, help_text="Hard floor for Sell RSI")
    max_sell_rsi = models.IntegerField(default=80, help_text="Hard ceiling for Sell RSI")
    mild_rsi_buffer_min = models.FloatField(default=2.0, help_text="Minimum allowed mild RSI buffer")
    default_recent_accuracy = models.FloatField(default=0.7, help_text="Fallback accuracy if no history available")

    # ---------------- Logging ----------------
    log_next_check_minutes = models.IntegerField(default=15, help_text="Minutes until next_check in DB logs")

    # ---------------- Metadata ----------------
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"StrategySettings(id={self.pk}, active={self.active})"
    

    
class TradeStatus(models.Model):
    id = models.IntegerField(primary_key=True, default=1, editable=False)
    is_running = models.BooleanField(default=False)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return "Running" if self.is_running else "Stopped"
    


class TradeLog(models.Model):
    log = models.TextField(default='', blank=True)  # All trades stored as text
    updated_at = models.DateTimeField(auto_now=True)

    def append_trade(
        self,
        coin,
        action,
        quantity=None,
        entry_price=None,
        exit_price=None,
        pnl=None,
        status='OPEN',
        ema200=None,
        rsi=None,
        volume_spike=False,
        order_symbol=None,
        trade_type=None,
        order_type=None,
        order_price=None,
        timestamp_ms=None,
        signature=None,
        next_check=None,
        actions=None
    ):
        next_check = next_check or timezone.now()
        timestamp = timezone.now().strftime('%Y-%m-%d %H:%M:%S')

        log_entry = (
            f"[{timestamp}] {coin} - {action} | "
            f"Quantity: {quantity or 'N/A'}, Entry Price: {entry_price or 'N/A'}, "
            f"Exit Price: {exit_price or 'N/A'}, PnL: {pnl or 'N/A'}, Status: {status}, "
            f"EMA200: {ema200 or 'N/A'}, RSI: {rsi or 'N/A'}, Volume Spike: {'Yes' if volume_spike else 'No'}, "
            f"Order Symbol: {order_symbol or 'N/A'}, Trade Type: {trade_type or 'N/A'}, "
            f"Order Type: {order_type or 'N/A'}, Order Price: {order_price or 'N/A'}, "
            f"Timestamp(ms): {timestamp_ms or 'N/A'}, Signature: {signature or 'N/A'}, "
            f"Next Check: {next_check.strftime('%Y-%m-%d %H:%M:%S')}, Actions: {actions or 'N/A'}\n"
        )
        self.log += log_entry
        self.save()

    def __str__(self):
        return f"Trade Log (Last updated: {self.updated_at.strftime('%Y-%m-%d %H:%M:%S')})"
    


class TradeRiskSettings(models.Model):
    """
    Stores risk management parameters for each trade.
    These settings can be applied dynamically per trade.
    """

    # Link to trade log if needed
    trade = models.ForeignKey("TradeLog", on_delete=models.CASCADE, related_name="risk_settings", null=True, blank=True)

    # Risk Management Settings
    take_profit_percent = models.FloatField(default=2.0, help_text="Take profit % per trade (e.g., 2.0 = +2%)")
    stop_loss_percent = models.FloatField(default=1.0, help_text="Stop loss % per trade (e.g., 1.0 = -1%)")
    trailing_stop_percent = models.FloatField(default=0.5, help_text="Trailing stop % from peak (e.g., 0.5 = -0.5%)")
    max_holding_minutes = models.IntegerField(default=720, help_text="Maximum holding time in minutes (default = 12h)")
    risk_per_trade_percent = models.FloatField(default=1.0, help_text="Portfolio % to risk per trade (for position sizing)")

    # Tracking
    active = models.BooleanField(default=True)
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return (
            f"Risk Settings | TP: {self.take_profit_percent}%, "
            f"SL: {self.stop_loss_percent}%, "
            f"TS: {self.trailing_stop_percent}%, "
            f"MaxHold: {self.max_holding_minutes}m, "
            f"Risk/Trade: {self.risk_per_trade_percent}%"
        )