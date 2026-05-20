"""
Telegram Notifier for DMS V4.
Sends critical alerts to a designated Telegram chat using the Telegram Bot API.
"""
import asyncio
from typing import Optional
from telegram import Bot
from config import settings


class TelegramNotifier:
    def __init__(self):
        self.enabled = settings.telegram_enabled
        self.token = settings.telegram_bot_token
        self.chat_id = settings.telegram_chat_id
        self.bot: Optional[Bot] = None

        if self.enabled and self.token and self.chat_id:
            self.bot = Bot(token=self.token)
            print("[Telegram] Bot initialized.")
        else:
            print("[Telegram] Disabled or missing credentials.")

    async def send_alert(self, message: str):
        if not self.bot or not self.enabled:
            return

        try:
            # We use a quick background task to avoid blocking
            asyncio.create_task(
                self.bot.send_message(chat_id=self.chat_id, text=f"🚨 DMS ALERT 🚨\n\n{message}")
            )
        except Exception as e:
            print(f"[Telegram] Failed to send alert: {e}")

    async def send_report(self, report: str):
        if not self.bot or not self.enabled:
            return

        try:
            asyncio.create_task(
                self.bot.send_message(chat_id=self.chat_id, text=f"📊 SESSION REPORT 📊\n\n{report}")
            )
        except Exception as e:
            print(f"[Telegram] Failed to send report: {e}")
