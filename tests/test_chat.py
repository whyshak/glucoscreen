"""
tests/test_chat.py — Test suite for Dia Chatbot and /api/chat endpoint.
"""

import unittest
from app import create_app
from app.chatbot_service import (
    FALLBACK_INTENT,
    INTENTS,
    QUICK_PROMPTS,
    WELCOME_MESSAGE,
    generate_mock_chat_response,
    score_intent,
)


class TestDiaChatbot(unittest.TestCase):
    """Unit and integration tests for the Dia chatbot UI & API."""

    def setUp(self):
        self.app = create_app()
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def test_chat_page_renders_successfully(self):
        """GET /chat renders the dedicated Dia chat interface."""
        response = self.client.get("/chat")
        self.assertEqual(response.status_code, 200)
        content = response.data.decode("utf-8")
        self.assertIn("Dia · Diabetes Assistant", content)
        self.assertIn("dia-fullpage-panel", content)
        self.assertIn("dia-chat-input-container", content)
        self.assertIn("Take Screening", content)

    def test_api_chat_empty_payload_returns_400(self):
        """POST /api/chat with non-JSON or empty message returns 400."""
        # Non-JSON
        resp1 = self.client.post("/api/chat", data="not json")
        self.assertEqual(resp1.status_code, 400)

        # Empty JSON
        resp2 = self.client.post("/api/chat", json={})
        self.assertEqual(resp2.status_code, 400)
        data2 = resp2.get_json()
        self.assertIn("error", data2)

        # Whitespace-only message
        resp3 = self.client.post("/api/chat", json={"message": "   "})
        self.assertEqual(resp3.status_code, 400)

    def test_api_chat_overlength_message_returns_400(self):
        """POST /api/chat with message > 1000 characters returns 400."""
        long_message = "a" * 1001
        response = self.client.post("/api/chat", json={"message": long_message})
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn("under 1,000 characters", data["error"])

    def test_api_chat_valid_greeting_inquiry(self):
        """POST /api/chat with 'hello' returns greeting and suggestion chips."""
        response = self.client.post("/api/chat", json={"message": "Hello there"})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("text", data)
        self.assertIn("suggestions", data)
        self.assertIn("Dia", data["text"])
        self.assertFalse(data["rag_ready"])
        self.assertIsInstance(data["suggestions"], list)
        self.assertGreater(len(data["suggestions"]), 0)

    def test_api_chat_symptoms_inquiry(self):
        """POST /api/chat with symptoms question returns medical guidance."""
        response = self.client.post(
            "/api/chat",
            json={"message": "What are the common symptoms of diabetes?"}
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("Common symptoms of diabetes", data["text"])
        self.assertIn("thirst", data["text"].lower())
        self.assertFalse(data["rag_ready"])

    def test_api_chat_diet_inquiry(self):
        """POST /api/chat with diet question returns diabetes plate recommendations."""
        response = self.client.post(
            "/api/chat",
            json={"message": "What food should I eat for a healthy diet?"}
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("diabetes-friendly plate", data["text"].lower())

    def test_api_chat_fallback_on_unrecognized_query(self):
        """POST /api/chat with unrecognized query returns friendly educational fallback."""
        response = self.client.post(
            "/api/chat",
            json={"message": "xyz987quantumphysicsfluctuations"}
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("I'm not quite sure I caught that", data["text"])
        self.assertGreaterEqual(len(data["suggestions"]), 1)

    def test_score_intent_direct_unit(self):
        """Verify intent scoring logic and fallback."""
        # Exact match
        matched_diet = score_intent("Tell me about healthy diet and meals")
        self.assertEqual(matched_diet["id"], "diet")

        # Exercise
        matched_ex = score_intent("How much workout and exercise should I do?")
        self.assertEqual(matched_ex["id"], "exercise")

        # Hypoglycemia
        matched_hypo = score_intent("What do I do for low blood sugar or hypoglycemia?")
        self.assertEqual(matched_hypo["id"], "hypoglycemia")

        # Unknown
        matched_fb = score_intent("abcdef 123456")
        self.assertEqual(matched_fb["id"], "fallback")


if __name__ == "__main__":
    unittest.main()
