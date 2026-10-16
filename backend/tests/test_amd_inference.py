import os
import sys
import json
import time
import unittest
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import threading

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.amd_inference import AMDInferenceEngine, StructuredReviewAnalysis


class MockLLMHandler(BaseHTTPRequestHandler):
    response_code = 200
    response_body = None
    delay = 0.0

    def log_message(self, *args):
        pass

    def do_POST(self):
        if MockLLMHandler.delay > 0:
            time.sleep(MockLLMHandler.delay)
        if MockLLMHandler.response_code != 200:
            self.send_response(MockLLMHandler.response_code)
            self.end_headers()
            return

        body = MockLLMHandler.response_body
        if body is None:
            body = json.dumps({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({
                            "sentiment": "negative",
                            "confidence": 0.95,
                            "complaint_category": "battery",
                            "product_issue": "Battery drain under load",
                            "recommended_action": "Audit supplier cell batch"
                        })
                    }
                }],
                "usage": {"prompt_tokens": 80, "completion_tokens": 30}
            })
        encoded = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


class TestAMDInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), MockLLMHandler)
        cls.port = cls.server.server_address[1]
        cls.mock_url = f"http://127.0.0.1:{cls.port}/v1"
        cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def setUp(self):
        MockLLMHandler.response_code = 200
        MockLLMHandler.response_body = None
        MockLLMHandler.delay = 0.0
        self.engine = AMDInferenceEngine()
        self.engine.reset_circuit_breaker()
        os.environ.pop("LLM_BASE_URL", None)
        os.environ.pop("LLM_API_KEY", None)
        os.environ.pop("RADEON_API_KEY", None)
        os.environ.pop("LLM_TIMEOUT", None)

    def tearDown(self):
        os.environ.pop("LLM_BASE_URL", None)
        os.environ.pop("LLM_API_KEY", None)
        os.environ.pop("RADEON_API_KEY", None)
        os.environ.pop("LLM_TIMEOUT", None)

    def test_missing_api_credentials_uses_cpu_fallback(self):
        # When no credentials are in environment, infer_batch uses CPU heuristics
        self.assertFalse(self.engine.token_factory_configured)
        res = self.engine.infer_batch([{"text": "The battery dies after 30 minutes.", "rating": 1}])
        self.assertEqual(len(res), 1)
        item = res[0]
        self.assertEqual(item["sentiment"], "negative")
        self.assertEqual(item["complaint_category"], "battery")
        self.assertIn("CPU", item["processing_engine"])

    def test_successful_mocked_api_response(self):
        os.environ["LLM_BASE_URL"] = self.mock_url
        os.environ["LLM_API_KEY"] = "mock-token-for-test"
        self.assertTrue(self.engine.token_factory_configured)

        res = self.engine.infer_batch([{"text": "Battery life is terrible and heats up quickly.", "rating": 1}])
        self.assertEqual(len(res), 1)
        item = res[0]
        self.assertEqual(item["sentiment"], "negative")
        self.assertEqual(item["complaint_category"], "battery")
        self.assertEqual(item["confidence"], 0.95)
        self.assertIn("AMD Token Factory", item["processing_engine"])

    def test_malformed_model_output_falls_back_to_cpu(self):
        os.environ["LLM_BASE_URL"] = self.mock_url
        os.environ["LLM_API_KEY"] = "mock-token-for-test"
        # Mock returns non-JSON text
        MockLLMHandler.response_body = json.dumps({
            "choices": [{"message": {"role": "assistant", "content": "I am unable to output JSON."}}]
        })

        res = self.engine.infer_batch([{"text": "Fan noise is loud and cooling is weak.", "rating": 2}])
        self.assertEqual(len(res), 1)
        item = res[0]
        # Should gracefully fall back to CPU heuristic without raising an exception
        self.assertEqual(item["sentiment"], "negative")
        self.assertEqual(item["complaint_category"], "cooling")
        self.assertIn("CPU", item["processing_engine"])

    def test_api_status_error_and_circuit_breaker(self):
        os.environ["LLM_BASE_URL"] = self.mock_url
        os.environ["LLM_API_KEY"] = "mock-token-for-test"
        MockLLMHandler.response_code = 500

        # Run 3 consecutive failures to trigger circuit breaker
        for i in range(3):
            res = self.engine.infer_batch([{"text": f"Broke instantly {i}", "rating": 1}])
            self.assertEqual(len(res), 1)
            self.assertIn("CPU", res[0]["processing_engine"])

        st = self.engine.status()
        self.assertTrue(st["circuit_paused"])
        self.assertFalse(st["token_factory_active"])

    def test_api_timeout_falls_back_to_cpu(self):
        # Target an unreachable port with short timeout
        os.environ["LLM_BASE_URL"] = "http://127.0.0.1:9/v1"
        os.environ["LLM_API_KEY"] = "mock-token-for-test"
        os.environ["LLM_TIMEOUT"] = "0.5"

        res = self.engine.infer_batch([{"text": "The delivery was delayed by a week.", "rating": 2}])
        self.assertEqual(len(res), 1)
        item = res[0]
        self.assertEqual(item["complaint_category"], "delivery")
        self.assertIn("CPU", item["processing_engine"])

    def test_rocm_active_remains_false_when_token_factory_configured(self):
        # Strict requirement: local rocm_active remains False even if Token Factory is connected
        os.environ["LLM_BASE_URL"] = self.mock_url
        os.environ["LLM_API_KEY"] = "mock-token-for-test"
        st = self.engine.status()
        self.assertTrue(st["token_factory_active"])
        self.assertFalse(st["rocm_active"])  # Never falsely claim local ROCm


if __name__ == "__main__":
    unittest.main()
