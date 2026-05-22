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

if __name__ == "__main__":
    unittest.main()
