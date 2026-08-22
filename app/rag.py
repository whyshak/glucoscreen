"""
rag.py — RAG (Retrieval-Augmented Generation) Engine using FAISS for Dr. GlucoBot.
"""

import os
import re
import numpy as np

# Try importing faiss and sentence_transformers
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False


class FallbackEmbedder:
    """Lightweight TF-IDF + Dense projection embedder if sentence-transformers is unavailable."""
    def __init__(self, passages):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        
        self.vectorizer = TfidfVectorizer(stop_words='english')
        tfidf = self.vectorizer.fit_transform(passages)
        
        n_components = min(32, max(2, tfidf.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        vecs = self.svd.fit_transform(tfidf)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        self.embeddings = (vecs / norms).astype(np.float32)

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        tfidf = self.vectorizer.transform(texts)
        vecs = self.svd.transform(tfidf)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        return (vecs / norms).astype(np.float32)


class DiabetesRAGEngine:
    """FAISS-powered Retrieval Augmented Generation engine for diabetes query answering."""

    def __init__(self, kb_path=None):
        if kb_path is None:
            kb_path = os.path.join(os.path.dirname(__file__), 'data', 'knowledge_base.txt')
        self.kb_path = kb_path
        self.passages = []
        self.passage_titles = []
        self.index = None
        self.embedder = None
        self.is_st_model = False
        
        self._initialize_index()

    def _initialize_index(self):
        """Loads knowledge base, creates passages, and builds FAISS index."""
        if not os.path.exists(self.kb_path):
            print(f"[RAG] Warning: Knowledge base path not found at {self.kb_path}")
            return

        with open(self.kb_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Split knowledge base by sections and headers
        raw_sections = re.split(r'\n(?=## |\n\n)', content)
        for sec in raw_sections:
            sec_clean = sec.strip()
            if len(sec_clean) > 30:
                lines = sec_clean.split('\n')
                title = lines[0].replace('#', '').strip()
                self.passages.append(sec_clean)
                self.passage_titles.append(title)

        if not self.passages:
            print("[RAG] Warning: No passages parsed from knowledge base.")
            return

        print(f"[RAG] Loaded {len(self.passages)} passages from knowledge base.")

        # Initialize Embedder
        if ST_AVAILABLE:
            try:
                print("[RAG] Initializing SentenceTransformer model 'all-MiniLM-L6-v2'...")
                self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
                self.is_st_model = True
            except Exception as e:
                print(f"[RAG] SentenceTransformer loading failed: {e}. Falling back to TF-IDF embedder.")
                self.embedder = FallbackEmbedder(self.passages)
                self.is_st_model = False
        else:
            print("[RAG] SentenceTransformers not installed. Using FallbackEmbedder.")
            self.embedder = FallbackEmbedder(self.passages)
            self.is_st_model = False

        # Encode passages
        if self.is_st_model:
            embeddings = self.embedder.encode(self.passages, convert_to_numpy=True)
            # Normalize vectors for Cosine similarity
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1e-10
            embeddings = (embeddings / norms).astype(np.float32)
        else:
            embeddings = self.embedder.embeddings

        # Build FAISS index
        dimension = embeddings.shape[1]
        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for normalized cosine similarity
            self.index.add(embeddings)
            print(f"[RAG] Built FAISS IndexFlatIP with {self.index.ntotal} vectors of dimension {dimension}.")
        else:
            print("[RAG] FAISS not available. Using numpy matrix multiplication fallback.")
            self.embeddings_matrix = embeddings

    def search(self, query, top_k=3):
        """Searches top_k matching passages for given user query using FAISS."""
        if not self.passages:
            return []

        if self.is_st_model:
            q_emb = self.embedder.encode([query], convert_to_numpy=True)
            norm = np.linalg.norm(q_emb)
            if norm > 0:
                q_emb = q_emb / norm
            q_emb = q_emb.astype(np.float32)
        else:
            q_emb = self.embedder.encode([query])

        results = []
        if FAISS_AVAILABLE and self.index is not None:
            scores, indices = self.index.search(q_emb, top_k)
            for idx, score in zip(indices[0], scores[0]):
                if 0 <= idx < len(self.passages):
                    results.append({
                        "passage": self.passages[idx],
                        "title": self.passage_titles[idx],
                        "score": float(score)
                    })
        else:
            # Fallback inner product search
            scores = np.dot(self.embeddings_matrix, q_emb.T).flatten()
            top_indices = np.argsort(scores)[::-1][:top_k]
            for idx in top_indices:
                results.append({
                    "passage": self.passages[idx],
                    "title": self.passage_titles[idx],
                    "score": float(scores[idx])
                })

        return results

    def answer_question(self, query, user_session=None):
        """Retrieves passages and generates a structured medical answer."""
        retrieved = self.search(query, top_k=3)
        
        # Build context summary
        context_text = "\n\n".join([r["passage"] for r in retrieved])
        sources = [r["title"] for r in retrieved]

        # Contextualize with user session screening results if present
        user_context_note = ""
        if user_session and "prediction_score" in user_session:
            score = user_session.get("prediction_score")
            level = user_session.get("risk_level", "Unknown")
            user_context_note = f"\n*(Note: Your current screening risk score is **{score}% - {level} Risk**)*\n"

        # Generate answer from retrieved context
        answer = self._synthesize_answer(query, context_text, user_context_note, retrieved)

        # Dynamic suggested follow-up questions
        suggestions = self._generate_suggestions(query, user_session)

        return {
            "answer": answer,
            "sources": sources,
            "suggested_questions": suggestions
        }

    def _synthesize_answer(self, query, context, user_note, retrieved):
        """Formulates an articulate response based on user query and retrieved clinical context."""
        query_lower = query.lower()

        # Custom tailored answers for common query themes
        if any(w in query_lower for w in ["hi", "hello", "hey", "who are you", "what can you do"]):
            return (
                f"Hello! 👋 I am **Dr. GlucoBot**, your AI Medical Assistant.\n\n"
                f"I can help answer your questions about **diabetes risk factors**, **symptoms**, **diet & lifestyle recommendations**, "
                f"and explain how our **GlucoScreen Machine Learning model** calculates diabetes risk profiles.\n\n"
                f"How can I assist your health journey today?"
            )

        if not context:
            return (
                "I apologize, but I couldn't find specific clinical data matching your query in my knowledge base. "
                "For comprehensive advice, please consult your primary healthcare provider."
            )

        # Build clean formatted answer incorporating retrieved facts
        lines = []
        if user_note:
            lines.append(user_note)

        # Extract major bullet points or paragraphs from context matching query terms
        relevant_blocks = []
        for r in retrieved:
            block = r["passage"]
            # remove header line if repeating
            block_lines = [l for l in block.split('\n') if not l.startswith('#')]
            clean_block = "\n".join(block_lines[:6]).strip()
            if clean_block:
                relevant_blocks.append(clean_block)

        main_body = "\n\n".join(relevant_blocks[:2])

        # Synthesize clear markdown response
        response = (
            f"{main_body}\n\n"
            f"--- \n"
            f"💡 *Medical Advisory: Dr. GlucoBot provides educational guidance based on clinical standards (ADA/CDC). "
            f"Always consult a licensed medical physician for diagnostic blood testing and clinical decisions.*"
        )
        return response

    def _generate_suggestions(self, query, user_session):
        """Generates dynamic follow-up prompt pills."""
        query_lower = query.lower()
        
        if "risk" in query_lower or "score" in query_lower:
            return [
                "What foods lower diabetes risk?",
                "What is the difference between A1C and Fasting Glucose?",
                "How much exercise is recommended per week?"
            ]
        elif "diet" in query_lower or "food" in query_lower or "eat" in query_lower:
            return [
                "What are low glycemic index foods?",
                "How does physical activity lower blood sugar?",
                "What are early diabetes symptoms?"
            ]
        elif "symptom" in query_lower or "sign" in query_lower:
            return [
                "Can prediabetes be reversed?",
                "What are standard A1C diagnostic thresholds?",
                "How does high blood pressure increase risk?"
            ]
        else:
            return [
                "What are the main risk factors for diabetes?",
                "How does GlucoScreen predict diabetes risk?",
                "What lifestyle changes reduce diabetes risk most?"
            ]
