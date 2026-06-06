import asyncio
import json
import os
from datetime import datetime, timezone
from playwright.async_api import async_playwright

ACCOUNTS_FILE = os.path.join(os.path.dirname(__file__), "accounts.json")
try:
    from src.core.config import ACCOUNTS_FILE
except ImportError:
    pass  # Fallback to local path when run standalone


async def extract_session_key(context):
    try:
        cookies = await context.cookies()
        for cookie in cookies:
            if cookie["name"] == "sessionKey":
                return cookie["value"]
    except Exception:
        pass
    return None


async def run_login():
    print("Launching Claude login window...")
    print("NOTE: Use Claude's EMAIL login (not Google OAuth) — Google blocks automated browsers.\n")
    async with async_playwright() as p:
        # Use bundled Chromium with anti-detection flags.
        # Google OAuth will NOT work here, but Claude's email login will.
        browser = await p.chromium.launch(
            headless=False,
            args=[
                "--disable-blink-features=AutomationControlled",
            ],
        )
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        )
        page = await context.new_page()

        # Hide the webdriver property
        await page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
        """)

        try:
            await page.goto("https://claude.ai/login", timeout=120000, wait_until="domcontentloaded")
        except Exception as e:
            print(f"Warning: Initial navigation issue ({e}), but the browser is open. Please continue logging in manually.")

        print("Please log in. Waiting for sessionKey cookie...")
        print("(You can close the browser to cancel)\n")

        session_key = None
        while True:
            await asyncio.sleep(2)
            try:
                # Check if browser is still open
                if not browser.is_connected():
                    print("\nBrowser was closed. Aborting.")
                    return
                session_key = await extract_session_key(context)
                if session_key:
                    break
            except Exception:
                print("\nBrowser was closed. Aborting.")
                return

        print("\n✅ Login detected! Successfully extracted sessionKey.")
        email = input("Please enter the email associated with this account: ").strip()

        try:
            await browser.close()
        except Exception:
            pass

        # Load existing accounts
        accounts_data = {"accounts": []}
        if os.path.exists(ACCOUNTS_FILE):
            try:
                with open(ACCOUNTS_FILE, "r") as f:
                    accounts_data = json.load(f)
            except Exception:
                pass

        # Check if email already exists and update, or append new
        updated = False
        for account in accounts_data["accounts"]:
            if account["email"] == email:
                account["sessionKey"] = session_key
                account["added_at"] = datetime.now(tz=timezone.utc).isoformat()
                account["status"] = "active"
                updated = True
                print(f"Updated existing account '{email}'.")
                break

        if not updated:
            accounts_data["accounts"].append({
                "email": email,
                "sessionKey": session_key,
                "added_at": datetime.now(tz=timezone.utc).isoformat(),
                "status": "active"
            })

        with open(ACCOUNTS_FILE, "w") as f:
            json.dump(accounts_data, f, indent=2)

        print(f"✅ Successfully saved account '{email}' to {ACCOUNTS_FILE}")

def list_accounts():
    if not os.path.exists(ACCOUNTS_FILE):
        print("No accounts file found.")
        return

    with open(ACCOUNTS_FILE, "r") as f:
        data = json.load(f)

    accounts = data.get("accounts", [])
    if not accounts:
        print("No accounts saved.")
        return

    print(f"\n{'#':<4} {'Email':<40} {'Status':<10} {'Added At':<28} {'Session Key'}")
    print("-" * 120)
    for i, acc in enumerate(accounts, 1):
        key = acc.get("sessionKey", "")
        masked = key[:8] + "..." + key[-4:] if len(key) > 12 else key
        print(f"{i:<4} {acc.get('email', 'N/A'):<40} {acc.get('status', 'N/A'):<10} {acc.get('added_at', 'N/A'):<28} {masked}")
    print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Manage Claude Accounts")
    parser.add_argument("--add", action="store_true", help="Add a new Claude account by logging in")
    parser.add_argument("--list", action="store_true", help="List all saved accounts")
    args = parser.parse_args()

    if args.add:
        asyncio.run(run_login())
    elif args.list:
        list_accounts()
    else:
        parser.print_help()
