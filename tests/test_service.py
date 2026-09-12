import unittest
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent.parent / "service" / "model_network.joblib"


@unittest.skipUnless(MODEL_PATH.exists(), "model_network.joblib not built yet")
class ServiceTest(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient

        import service.app as app_module

        self.app_module = app_module
        self.client = TestClient(app_module.app)

    def tearDown(self):
        self.app_module.API_KEY = ""
        self.app_module.MODE = "both"

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["model_loaded"])

    def test_analyze_network_with_no_fields_allows_by_default(self):
        response = self.client.post("/analyze/network", json={})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["verdict"], "allow")
        self.assertIsNone(body["reasoning"])
        self.assertGreater(body["safety_score"], 0.9)

    def test_analyze_network_rejects_wrong_api_key_when_configured(self):
        self.app_module.API_KEY = "secret-token"
        response = self.client.post("/analyze/network", json={}, headers={"Authorization": "Bearer wrong"})
        self.assertEqual(response.status_code, 401)

    def test_analyze_network_accepts_correct_api_key(self):
        self.app_module.API_KEY = "secret-token"
        response = self.client.post("/analyze/network", json={}, headers={"Authorization": "Bearer secret-token"})
        self.assertEqual(response.status_code, 200)

    def test_network_only_mode_blocks_file_endpoint(self):
        self.app_module.MODE = "network"
        response = self.client.post("/analyze/file", content=b"hello")
        self.assertEqual(response.status_code, 404)

    def test_analyze_file_never_crashes_regardless_of_clamd_availability(self):
        response = self.client.post("/analyze/file", content=b"hello world")
        self.assertEqual(response.status_code, 200)
        self.assertIn(response.json()["verdict"], ("allow", "block", "uncertain"))

    def test_analyze_file_rejects_empty_body(self):
        response = self.client.post("/analyze/file", content=b"")
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
