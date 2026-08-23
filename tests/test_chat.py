"""
tests/test_chat.py — Test suite for Dia Live RAG Chatbot and /api/chat endpoint.
"""

import unittest
from app import create_app
from app.rag_service import (
    DEFAULT_SUGGESTIONS,
    QUICK_PROMPTS,
    SYSTEM_PROMPT,
    WELCOME_MESSAGE,
    RAGChatbot,
    rag_bot,
)


class TestDiaLiveRAGChatbot(unittest.TestCase):
    """Unit and integration tests for the live RAG chatbot backend and /api/chat route."""

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

    def test_api_chat_live_rag_inquiry_symptoms(self):
        """POST /api/chat with symptoms query retrieves context and returns answer."""
        response = self.client.post(
            "/api/chat",
            json={"message": "What are the common symptoms of diabetes?"}
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("text", data)
        self.assertIn("suggestions", data)
        self.assertIn("sources", data)
        self.assertTrue(data.get("rag_ready", False))
        self.assertIsInstance(data["suggestions"], list)
        self.assertGreater(len(data["suggestions"]), 0)

    def test_api_chat_live_rag_inquiry_diet(self):
        """POST /api/chat with diet query returns dietary guidance and suggestions."""
        response = self.client.post(
            "/api/chat",
            json={"message": "What should I eat for breakfast with diabetes?"}
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("text", data)
        self.assertTrue(len(data["text"]) > 0)
        self.assertTrue(data.get("rag_ready", False))

    def test_rag_chatbot_direct_retrieval(self):
        """Test RAGChatbot retrieval directly."""
        bot = RAGChatbot()
        bot.initialize()
        self.assertIsNotNone(bot._retriever)

        # Retrieve documents
        docs = bot._retriever.invoke("blood sugar levels")
        self.assertGreater(len(docs), 0)
        self.assertTrue(any("blood" in d.page_content.lower() or "glucose" in d.page_content.lower() or "diabetes" in d.page_content.lower() for d in docs))

    def test_rag_chatbot_ask_returns_structured_dict(self):
        """Test RAGChatbot ask method output format."""
        result = rag_bot.ask("What is prediabetes?")
        self.assertIn("text", result)
        self.assertIn("suggestions", result)
        self.assertIn("sources", result)
        self.assertTrue(result["rag_ready"])
        self.assertIsInstance(result["suggestions"], list)


if __name__ == "__main__":
    unittest.main()
