"""
test_rag.py — Automated verification test for FAISS RAG chatbot system.
"""

import os
import sys

# Ensure root directory is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_rag_engine():
    from app.rag import DiabetesRAGEngine

    print("--> Initializing DiabetesRAGEngine...")
    engine = DiabetesRAGEngine()
    
    assert len(engine.passages) > 0, "Passages should not be empty!"
    print(f"--> Loaded {len(engine.passages)} passages.")

    test_queries = [
        "What are the main risk factors for diabetes?",
        "How does high blood pressure affect diabetes risk?",
        "What foods lower blood sugar?",
        "How accurate is the GlucoScreen SVM model?"
    ]

    for q in test_queries:
        print(f"\n[QUERY]: {q}")
        res = engine.answer_question(q)
        assert "answer" in res
        assert "sources" in res
        print(f"[SOURCES]: {res['sources']}")
        print(f"[ANSWER PREVIEW]: {res['answer'][:150].encode('ascii', 'ignore').decode('ascii')}...")
        print(f"[SUGGESTIONS]: {res['suggested_questions']}")


    print("\n[SUCCESS] FAISS RAG Engine Verification Test Passed Successfully!")

if __name__ == '__main__':
    test_rag_engine()
