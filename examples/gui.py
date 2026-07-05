import sys
import os

# Add project root to sys.path so that we can import src.server when running from examples/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import socket
import subprocess
import threading
import time
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
    main_window = webview.create_window(
        title="GhostTunnel All-in-One Suite",
        url=f"http://127.0.0.1:{PORT}/dashboard/",
        width=1200,
        height=800,
        resizable=True
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
