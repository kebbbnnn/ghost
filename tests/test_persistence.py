import unittest
import os
import json
from fastapi.testclient import TestClient
from src.server import app, THREADS_FILE

class TestThreadsPersistence(unittest.TestCase):
    def setUp(self):
        # Backup threads file if it exists
        self.backup_path = THREADS_FILE + ".bak"
        if os.path.exists(THREADS_FILE):
            os.rename(THREADS_FILE, self.backup_path)
        self.client = TestClient(app)

    def tearDown(self):
        # Restore backup if it exists
        if os.path.exists(THREADS_FILE):
            os.remove(THREADS_FILE)
        if os.path.exists(self.backup_path):
            os.rename(self.backup_path, THREADS_FILE)

    def test_persistence_flow(self):
        # 1. Fetch initially - should return empty list or file content if not exists
        response = self.client.get("/api/threads")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), [])

        # 2. Save new threads
        dummy_threads = [
            {
                "id": "thread_12345",
                "model": "auto",
                "title": "Test Title",
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there"}
                ],
                "created": 123456789
            }
        ]
        
        response = self.client.post("/api/threads", json=dummy_threads)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "success"})

        # 3. Verify file exists on disk
        self.assertTrue(os.path.exists(THREADS_FILE))
        with open(THREADS_FILE, "r") as f:
            data = json.load(f)
        self.assertEqual(data, dummy_threads)

        # 4. Fetch again - should return the dummy threads
        response = self.client.get("/api/threads")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), dummy_threads)

    def test_persistence_utf8_and_validation(self):
        # Test UTF-8 emojis and multi-byte unicode
        utf8_threads = [
            {
                "id": "thread_utf8",
                "title": "Quantum Physics 🚀 & 東京 weather ☀️",
                "messages": [
                    {"role": "user", "content": "Bonjour le monde! こんにちは 🚀"}
                ]
            }
        ]
        res = self.client.post("/api/threads", json=utf8_threads)
        self.assertEqual(res.status_code, 200)

        # Read directly using utf-8
        with open(THREADS_FILE, "r", encoding="utf-8") as f:
            saved = json.load(f)
        self.assertEqual(saved[0]["title"], "Quantum Physics 🚀 & 東京 weather ☀️")

        # Test rejecting non-list/dict payloads
        res_invalid = self.client.post("/api/threads", json="just a string")
        self.assertEqual(res_invalid.status_code, 400)

        res_invalid_num = self.client.post("/api/threads", json=12345)
        self.assertEqual(res_invalid_num.status_code, 400)

    def test_root_redirect(self):
        # Root GET / should redirect to /dashboard/
        res = self.client.get("/", follow_redirects=False)
        self.assertIn(res.status_code, (302, 307))
        self.assertEqual(res.headers.get("location"), "/dashboard/")

if __name__ == "__main__":
    unittest.main()
