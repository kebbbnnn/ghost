import socket
import subprocess
import threading
import time
import os
import uvicorn
import webview
from src.server import app

PORT = 8100

class UvicornServerThread(threading.Thread):
    def __init__(self, app, host="127.0.0.1", port=8100):
        super().__init__()
        self.daemon = True
        self.config = uvicorn.Config(app, host=host, port=port, log_level="warning")
        self.server = uvicorn.Server(self.config)

    def run(self):
        self.server.run()

    def stop(self):
        self.server.should_exit = True


class GUIBridgeAPI:
    def __init__(self):
        self.login_window = None

    def open_claude_login(self):
        if self.login_window is not None:
            try:
                self.login_window.focus()
                return
            except Exception:
                self.login_window = None

        print("[GUIBridge] Spawning Claude Login child webview...")
        self.login_window = webview.create_window(
            title="Claude Login Helper",
            url="https://claude.ai/login",
            width=550,
            height=700,
            resizable=True
        )
        
        t = threading.Thread(target=self._poll_claude_cookies, daemon=True)
        t.start()

    def _poll_claude_cookies(self):
        session_key = None
        while self.login_window is not None:
            time.sleep(1.5)
            try:
                cookies = self.login_window.get_cookies()
                for cookie in cookies:
                    cookie_name = getattr(cookie, 'name', None) or (cookie.get('name') if isinstance(cookie, dict) else None)
                    cookie_val = getattr(cookie, 'value', None) or (cookie.get('value') if isinstance(cookie, dict) else None)
                    cookie_domain = getattr(cookie, 'domain', None) or (cookie.get('domain') if isinstance(cookie, dict) else None)
                    
                    if cookie_name == "sessionKey" and cookie_domain and "claude.ai" in cookie_domain:
                        session_key = cookie_val
                        break
                
                if session_key:
                    print(f"[GUIBridge] Captured sessionKey cookie! Value: {session_key[:8]}...")
                    self._save_claude_account(session_key)
                    self.login_window.destroy()
                    self.login_window = None
                    break
            except Exception:
                break

    def _save_claude_account(self, session_key: str):
        import httpx
        try:
            email = "claude-session@ghosttunnel.local"
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Cookie": f"sessionKey={session_key}"
            }
            try:
                r = httpx.get("https://claude.ai/api/me", headers=headers, timeout=10.0)
                if r.status_code == 200:
                    email = r.json().get("email_address", email)
            except Exception as e:
                print(f"[GUIBridge] Failed to fetch email address from Claude API: {e}")

            r = httpx.post(f"http://127.0.0.1:{PORT}/api/accounts/add", json={
                "email": email,
                "sessionKey": session_key
            }, timeout=5.0)
            if r.status_code == 200:
                print(f"[GUIBridge] Successfully registered account '{email}' on server.")
            else:
                print(f"[GUIBridge] Failed to register account on server: {r.text}")
        except Exception as e:
            print(f"[GUIBridge] Error saving Claude account: {e}")


def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0


def free_port(port):
    try:
        pids = subprocess.check_output(["lsof", "-ti", f":{port}"]).decode().strip().split()
        for pid in pids:
            if pid:
                print(f"[Port Recovery] Killing process {pid} occupying port {port}...")
                subprocess.run(["kill", "-9", pid], check=True)
        time.sleep(1.0)
    except Exception as e:
        print(f"[Port Recovery] Failed to free port {port}: {e}")


def main():
    print("==================================================")
    print("GhostTunnel Desktop Application Suite Bootstrap")
    print("==================================================")

    # Check port conflicts
    if is_port_in_use(PORT):
        print(f"[Warning] Port {PORT} is already in use!")
        free_port(PORT)
        if is_port_in_use(PORT):
            print(f"[Error] Port {PORT} remains occupied. Aborting.")
            return

    # Start FastAPI server thread
    server_thread = UvicornServerThread(app, host="127.0.0.1", port=PORT)
    server_thread.start()
    print(f"FastAPI Server started programmatically on http://127.0.0.1:{PORT}")

    # Wait for server to start listening
    print("[GUI Setup] Waiting for FastAPI server to start listening...")
    for _ in range(50):  # wait up to 5 seconds
        if is_port_in_use(PORT):
            print("[GUI Setup] FastAPI server is ready.")
            break
        time.sleep(0.1)
    else:
        print("[GUI Setup] Warning: FastAPI server did not start in time.")

    # Initialize PyWebView main window
    api = GUIBridgeAPI()
    main_window = webview.create_window(
        title="GhostTunnel All-in-One Suite",
        url=f"http://127.0.0.1:{PORT}/dashboard/",
        width=1200,
        height=800,
        resizable=True,
        js_api=api
    )

    # Lifecycle teardown on close
    def on_closed():
        print("Main window closed. Terminating FastAPI server...")
        server_thread.stop()

    main_window.events.closed += on_closed

    # Start Webview main loop
    webview.start()

if __name__ == "__main__":
    main()
