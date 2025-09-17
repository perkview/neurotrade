from django.core.management.base import BaseCommand
from dashboard.models import BotStatus
from dashboard import main
import time

class Command(BaseCommand):
    help = "Run the trading bot continuously in the background"

    def handle(self, *args, **kwargs):
        self.stdout.write("🚀 Bot process started...")

        while True:
            bot_status, _ = BotStatus.objects.get_or_create(id=1)

            if bot_status.is_running:
                main.main_loop()
                time.sleep(15 * 60)  # wait 15 min before next run
            else:
                time.sleep(5)  # sleep shorter if bot is off
