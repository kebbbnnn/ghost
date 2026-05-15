import os
import json
import time
import asyncio
from typing import Optional, Dict, Any
from src.core.exceptions import RateLimitError

class AccountsExhaustedError(RateLimitError):
    pass

class AccountRotator:
    def __init__(self, accounts_file: str):
        self.accounts_file = accounts_file
        self.active_accounts = []
        self.cooldowns: Dict[str, float] = {}  # email -> timestamp when cooldown ends
        self.last_mtime = 0
        self.current_index = 0
        self._lock = asyncio.Lock()

    async def _load_accounts_if_changed(self):
        if not os.path.exists(self.accounts_file):
            return

        current_mtime = os.path.getmtime(self.accounts_file)
        if current_mtime > self.last_mtime:
            try:
                with open(self.accounts_file, "r") as f:
                    data = json.load(f)
                    self.active_accounts = [acc for acc in data.get("accounts", []) if acc.get("status") == "active" and acc.get("sessionKey")]
                self.last_mtime = current_mtime
            except Exception as e:
                print(f"[ClaudeRotator] Error loading accounts: {e}")

    async def get_next_account(self) -> Dict[str, Any]:
        async with self._lock:
            await self._load_accounts_if_changed()

            if not self.active_accounts:
                raise AccountsExhaustedError("No active Claude accounts configured.")

            now = time.time()
            start_index = self.current_index
            
            # If current_index is out of bounds (e.g. accounts were removed), reset it
            if self.current_index >= len(self.active_accounts):
                self.current_index = 0
                start_index = 0
            
            while True:
                account = self.active_accounts[self.current_index]
                email = account["email"]

                self.current_index += 1
                if self.current_index >= len(self.active_accounts):
                    self.current_index = 0

                # Check cooldown
                if email in self.cooldowns:
                    if now < self.cooldowns[email]:
                        # Still in cooldown
                        if self.current_index == start_index:
                            # We looped through all accounts and none are available
                            raise AccountsExhaustedError("All Claude accounts are currently rate limited.")
                        continue
                    else:
                        # Cooldown expired, remove it
                        del self.cooldowns[email]
                
                return account

    async def mark_rate_limited(self, email: str, retry_after_seconds: Optional[int] = None):
        async with self._lock:
            # Default to 5 hours (18000 seconds) if no specific time given
            cooldown_period = retry_after_seconds if retry_after_seconds is not None else 18000
            self.cooldowns[email] = time.time() + cooldown_period
            print(f"[ClaudeRotator] Account {email} rate limited. Cooldown for {cooldown_period} seconds.")
