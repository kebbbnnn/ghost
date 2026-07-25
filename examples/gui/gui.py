import sys
import os

# Add project root to sys.path so that we can import src.server when running from examples/gui/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import socket
import subprocess
import threading
import time
import webview

PORT = 8100

class UvicornServerThread(threading.Thread):
    def __init__(self, app, host="127.0.0.1", port=8100):
        super().__init__()
        self.daemon = True
        import uvicorn
        self.config = uvicorn.Config(app, host=host, port=port, log_level="warning")
        self.server = uvicorn.Server(self.config)

    def run(self):
        self.server.run()

    def stop(self):
        self.server.should_exit = True


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

    import argparse
    parser = argparse.ArgumentParser(description="GhostTunnel Desktop Application Suite Bootstrap")
    parser.add_argument(
        "--remote", "-r",
        action="store_true",
        help="Connect to the remote server (https://ghosttunnel-lk2x.onrender.com)"
    )
    parser.add_argument(
        "--url", "-u",
        type=str,
        default=None,
        help="Connect to a custom remote or local server URL"
    )
    args = parser.parse_args()

    remote_server_url = "https://ghosttunnel-lk2x.onrender.com"
    target_url = None

    if args.remote:
        target_url = remote_server_url
    elif args.url:
        target_url = args.url
    elif sys.stdin.isatty():
        print("\nChoose connection mode:")
        print("1. Local Server (127.0.0.1)")
        print("2. Remote Server (ghosttunnel-lk2x.onrender.com)")
        try:
            choice = input("Enter choice [1/2] (default 1): ").strip()
            if choice == "2":
                target_url = remote_server_url
        except (KeyboardInterrupt, EOFError):
            print("\nDefaulting to Local Server.")

    if target_url:
        # Standardize target_url path to point to /dashboard/
        from urllib.parse import urlparse
        parsed = urlparse(target_url)
        if not parsed.path or parsed.path == "/":
            target_url = f"{target_url.rstrip('/')}/dashboard/"
        elif parsed.path.rstrip("/") == "/dashboard" and not target_url.endswith("/"):
            target_url = f"{target_url}/"

        print(f"[GUI Setup] Connecting to remote server at: {target_url}")

        main_window = webview.create_window(
            title="GhostTunnel All-in-One Suite (Remote)",
            url=target_url,
            width=1200,
            height=800,
            resizable=True,
            text_select=True
        )
        webview.start()
        return

    # Check port conflicts
    if is_port_in_use(PORT):
        print(f"[Warning] Port {PORT} is already in use!")
        free_port(PORT)
        if is_port_in_use(PORT):
            print(f"[Error] Port {PORT} remains occupied. Aborting.")
            return

    # Start FastAPI server thread
    from src.server import app
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
    main_window = webview.create_window(
        title="GhostTunnel All-in-One Suite",
        url=f"http://127.0.0.1:{PORT}/dashboard/",
        width=1200,
        height=800,
        resizable=True,
        text_select=True
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
