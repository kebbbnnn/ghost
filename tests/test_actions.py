"""Unit tests for the action schema, compiler, and validator."""

import unittest
from unittest.mock import patch, MagicMock

from agent.actions import (
    parse_action,
    validate_action,
    compile_action,
    ClickAction,
    DoubleClickAction,
    RightClickAction,
    TypeAction,
    HotkeyAction,
    ScrollAction,
    WaitAction,
)


class TestParseAction(unittest.TestCase):
    """Test parsing JSON dicts into Pydantic action models."""

    def test_parse_click(self):
        action = parse_action({"action": "click", "x": 100, "y": 200})
        self.assertIsInstance(action, ClickAction)
        self.assertEqual(action.x, 100)
        self.assertEqual(action.y, 200)

    def test_parse_double_click(self):
        action = parse_action({"action": "double_click", "x": 50, "y": 75})
        self.assertIsInstance(action, DoubleClickAction)

    def test_parse_right_click(self):
        action = parse_action({"action": "right_click", "x": 300, "y": 400})
        self.assertIsInstance(action, RightClickAction)

    def test_parse_type(self):
        action = parse_action({"action": "type", "text": "Hello World"})
        self.assertIsInstance(action, TypeAction)
        self.assertEqual(action.text, "Hello World")

    def test_parse_hotkey(self):
        action = parse_action({"action": "hotkey", "keys": ["command", "s"]})
        self.assertIsInstance(action, HotkeyAction)
        self.assertEqual(action.keys, ["command", "s"])

    def test_parse_scroll(self):
        action = parse_action({
            "action": "scroll",
            "x": 400,
            "y": 300,
            "direction": "down",
            "amount": 5,
        })
        self.assertIsInstance(action, ScrollAction)
        self.assertEqual(action.direction, "down")
        self.assertEqual(action.amount, 5)

    def test_parse_wait(self):
        action = parse_action({"action": "wait", "seconds": 2.5})
        self.assertIsInstance(action, WaitAction)
        self.assertEqual(action.seconds, 2.5)

    def test_parse_wait_default_seconds(self):
        action = parse_action({"action": "wait"})
        self.assertIsInstance(action, WaitAction)
        self.assertEqual(action.seconds, 1.0)

    def test_parse_invalid_action_type(self):
        with self.assertRaises(Exception):
            parse_action({"action": "explode", "x": 100})

    def test_parse_missing_required_field(self):
        with self.assertRaises(Exception):
            parse_action({"action": "click", "x": 100})  # missing y

    def test_parse_empty_dict(self):
        with self.assertRaises(Exception):
            parse_action({})

    def test_parse_nonsense(self):
        with self.assertRaises(Exception):
            parse_action({"foo": "bar"})


class TestValidateAction(unittest.TestCase):
    """Test action validation against safety policies."""

    def test_valid_click_in_bounds(self):
        action = ClickAction(action="click", x=100, y=200)
        valid, reason = validate_action(action, window_width=800, window_height=600)
        self.assertTrue(valid)
        self.assertEqual(reason, "")

    def test_click_out_of_bounds_x(self):
        action = ClickAction(action="click", x=900, y=200)
        valid, reason = validate_action(action, window_width=800, window_height=600)
        self.assertFalse(valid)
        self.assertIn("x", reason.lower())

    def test_click_out_of_bounds_y(self):
        action = ClickAction(action="click", x=100, y=700)
        valid, reason = validate_action(action, window_width=800, window_height=600)
        self.assertFalse(valid)
        self.assertIn("y", reason.lower())

    def test_click_negative_coordinate(self):
        action = ClickAction(action="click", x=-10, y=200)
        valid, reason = validate_action(action, window_width=800, window_height=600)
        self.assertFalse(valid)

    def test_hotkey_allowed(self):
        action = HotkeyAction(action="hotkey", keys=["command", "s"])
        valid, reason = validate_action(action, window_width=800, window_height=600)
        self.assertTrue(valid)

    def test_hotkey_blacklisted_cmd_q(self):
        action = HotkeyAction(action="hotkey", keys=["command", "q"])
        valid, reason = validate_action(action, window_width=800, window_height=600)
        self.assertFalse(valid)
        self.assertIn("blacklist", reason.lower())

    def test_hotkey_blacklisted_cmd_shift_delete(self):
        action = HotkeyAction(action="hotkey", keys=["command", "shift", "delete"])
        valid, reason = validate_action(action, window_width=800, window_height=600)
        self.assertFalse(valid)

    def test_type_valid(self):
        action = TypeAction(action="type", text="Hello")
        valid, reason = validate_action(action, window_width=800, window_height=600)
        self.assertTrue(valid)

    def test_type_too_long(self):
        action = TypeAction(action="type", text="x" * 501)
        valid, reason = validate_action(action, window_width=800, window_height=600)
        self.assertFalse(valid)
        self.assertIn("length", reason.lower())

    def test_wait_valid(self):
        action = WaitAction(action="wait", seconds=3.0)
        valid, reason = validate_action(action, window_width=800, window_height=600)
        self.assertTrue(valid)

    def test_wait_too_long(self):
        action = WaitAction(action="wait", seconds=15.0)
        valid, reason = validate_action(action, window_width=800, window_height=600)
        self.assertFalse(valid)
        self.assertIn("seconds", reason.lower())

    def test_scroll_in_bounds(self):
        action = ScrollAction(action="scroll", x=400, y=300, direction="down", amount=3)
        valid, reason = validate_action(action, window_width=800, window_height=600)
        self.assertTrue(valid)


class TestCompileAction(unittest.TestCase):
    """Test action compilation to pyautogui callables."""

    @patch("agent.actions.pyautogui")
    def test_compile_click(self, mock_pyautogui):
        action = ClickAction(action="click", x=100, y=200)
        fn = compile_action(action, window_x=50, window_y=30)
        fn()
        mock_pyautogui.click.assert_called_once_with(150, 230)

    @patch("agent.actions.pyautogui")
    def test_compile_double_click(self, mock_pyautogui):
        action = DoubleClickAction(action="double_click", x=100, y=200)
        fn = compile_action(action, window_x=0, window_y=0)
        fn()
        mock_pyautogui.doubleClick.assert_called_once_with(100, 200)

    @patch("agent.actions.pyautogui")
    def test_compile_type(self, mock_pyautogui):
        action = TypeAction(action="type", text="Hello")
        fn = compile_action(action, window_x=0, window_y=0)
        fn()
        mock_pyautogui.write.assert_called_once_with("Hello", interval=0.02)

    @patch("agent.actions.pyautogui")
    def test_compile_hotkey(self, mock_pyautogui):
        action = HotkeyAction(action="hotkey", keys=["command", "c"])
        fn = compile_action(action, window_x=0, window_y=0)
        fn()
        mock_pyautogui.hotkey.assert_called_once_with("command", "c")

    @patch("agent.actions.pyautogui")
    def test_compile_scroll_down(self, mock_pyautogui):
        action = ScrollAction(action="scroll", x=400, y=300, direction="down", amount=5)
        fn = compile_action(action, window_x=10, window_y=20)
        fn()
        mock_pyautogui.scroll.assert_called_once_with(-5, 410, 320)

    @patch("agent.actions.pyautogui")
    def test_compile_scroll_up(self, mock_pyautogui):
        action = ScrollAction(action="scroll", x=400, y=300, direction="up", amount=3)
        fn = compile_action(action, window_x=0, window_y=0)
        fn()
        mock_pyautogui.scroll.assert_called_once_with(3, 400, 300)

    @patch("agent.actions.time")
    def test_compile_wait(self, mock_time):
        action = WaitAction(action="wait", seconds=2.0)
        fn = compile_action(action, window_x=0, window_y=0)
        fn()
        mock_time.sleep.assert_called_once_with(2.0)

    @patch("agent.actions.pyautogui")
    def test_compile_action_propagates_exception(self, mock_pyautogui):
        mock_pyautogui.click.side_effect = RuntimeError("PyAutoGUI device access error")
        action = ClickAction(action="click", x=100, y=200)
        fn = compile_action(action, window_x=0, window_y=0)
        with self.assertRaises(RuntimeError):
            fn()


if __name__ == "__main__":
    unittest.main()
