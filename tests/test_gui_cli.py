import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# Ensure project root is in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from examples.gui import gui

class TestGuiCLI(unittest.TestCase):
    @patch("examples.gui.gui.webview")
    def test_remote_argument(self, mock_webview):
        # Test that --remote maps to the correct remote URL and doesn't run the server
        with patch("sys.argv", ["gui.py", "--remote"]):
            with patch("sys.stdin.isatty", return_value=False):
                with patch("examples.gui.gui.UvicornServerThread") as mock_server_thread:
                    gui.main()
                    
                    # Assert local server thread was NOT instantiated or started
                    mock_server_thread.assert_not_called()
                    
                    # Assert webview window was created with remote URL
                    mock_webview.create_window.assert_called_once_with(
                        title="GhostTunnel All-in-One Suite (Remote)",
                        url="https://ghosttunnel-lk2x.onrender.com/dashboard/",
                        width=1200,
                        height=800,
                        resizable=True,
                        text_select=True
                    )
                    mock_webview.start.assert_called_once()

    @patch("examples.gui.gui.webview")
    def test_custom_url_argument(self, mock_webview):
        # Test that --url maps to the correct custom URL and doesn't run the server
        with patch("sys.argv", ["gui.py", "--url", "https://custom-server.com/dashboard"]):
            with patch("sys.stdin.isatty", return_value=False):
                with patch("examples.gui.gui.UvicornServerThread") as mock_server_thread:
                    gui.main()
                    
                    # Assert local server thread was NOT instantiated or started
                    mock_server_thread.assert_not_called()
                    
                    # Assert webview window was created with standardized custom URL (ends with slash)
                    mock_webview.create_window.assert_called_once_with(
                        title="GhostTunnel All-in-One Suite (Remote)",
                        url="https://custom-server.com/dashboard/",
                        width=1200,
                        height=800,
                        resizable=True,
                        text_select=True
                    )

    @patch("examples.gui.gui.webview")
    @patch("examples.gui.gui.is_port_in_use")
    def test_local_default(self, mock_port_in_use, mock_webview):
        # Test that default execution runs the local server and webview
        with patch("sys.argv", ["gui.py"]):
            with patch("sys.stdin.isatty", return_value=False):
                mock_port_in_use.return_value = False
                
                with patch("examples.gui.gui.UvicornServerThread") as mock_server_thread:
                    mock_server_instance = MagicMock()
                    mock_server_thread.return_value = mock_server_instance
                    
                    # Since is_port_in_use will be called in a loop waiting for server,
                    # we need it to return True after the server is started.
                    mock_port_in_use.side_effect = [False, True]
                    
                    gui.main()
                    
                    # Assert local server thread was started
                    mock_server_thread.assert_called_once()
                    mock_server_instance.start.assert_called_once()
                    
                    # Assert webview window was created with local URL
                    mock_webview.create_window.assert_called_once_with(
                        title="GhostTunnel All-in-One Suite",
                        url="http://127.0.0.1:8100/dashboard/",
                        width=1200,
                        height=800,
                        resizable=True,
                        text_select=True
                    )

