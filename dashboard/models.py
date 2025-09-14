from django.db import models

# Create your models here.
from django.db import models
from django.utils import timezone

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