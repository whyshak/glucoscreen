"""
rag_service.py — Live Retrieval-Augmented Generation (RAG) Service for Dia.

Encapsulates the embedding model, vector database (Pinecone or local FAISS index
built from scraped NIDDK medical articles), semantic retriever, and LLM chain
for Dia, the diabetes health assistant.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── Prompt Templates ─────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are Dia, a Diabetes Assistant for question-answering tasks.\n"
    "Answer only questions related to diabetes — its causes, symptoms, types,\n"
    "prevention, management, diagnosis, treatment, and related complications.\n"
    "If a question is not related to diabetes, politely say that you can only\n"
    "help with diabetes-related questions.\n"
    "Use the following pieces of retrieved context, sourced from NIDDK, to\n"
    "answer the question.\n"
    "If the answer is not present in the context, say that you don't know\n"
    "rather than guessing.\n"
    "Use 3 sentences maximum and keep the answer concise.\n\n"
    "{context}"
)

QUICK_PROMPTS: List[str] = [
    "What are the symptoms of diabetes?",
    "What should I eat for breakfast?",
    "How much should I exercise?",
    "What is a normal blood sugar level?",
    "How can I prevent diabetes?",
]

WELCOME_MESSAGE: Dict[str, Any] = {
    "role": "assistant",
    "text": (
        "Hello! 👋 I'm Dia, your live RAG-powered diabetes assistant. "
        "I can answer questions about diabetes risk, symptoms, diet plans, "
        "exercise, blood sugar levels, and clinical guidance backed by NIDDK medical research.\n\n"
        "What would you like to know today?"
    ),
    "suggestions": [
        "What are the symptoms of diabetes?",
        "What should I eat for breakfast?",
        "How much should I exercise?",
        "What is a normal blood sugar level?",
    ],
}

DEFAULT_SUGGESTIONS: List[str] = [
    "What are the symptoms of diabetes?",
    "What should I eat for breakfast?",
    "What is a normal blood sugar level?",
    "How can I prevent diabetes?",
]


class RAGChatbot:
    """
    RAG-based Chatbot engine for Dia using LangChain, HuggingFace embeddings,
    vector store (Pinecone or FAISS fallback), and LLM (Gemini or OpenAI).
    """

    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        data_path: Optional[str] = None,
        faiss_index_path: Optional[str] = None,
        pinecone_index_name: Optional[str] = None,
    ):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.data_path = data_path or os.path.join(base_dir, "data", "niddk_diabetes.json")
        self.faiss_index_path = faiss_index_path or os.path.join(base_dir, "data", "faiss_index")
        self.pinecone_index_name = pinecone_index_name or os.getenv("PINECONE_INDEX_NAME", "medibot")
        self.embedding_model_name = embedding_model_name

        self._embeddings = None
        self._vectorstore = None
        self._retriever = None
        self._rag_chain = None
        self._llm = None
        self._initialized = False

    def _get_embeddings(self):
        """Lazy load HuggingFace embedding model."""
        if self._embeddings is None:
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
                self._embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
            except Exception as e:
                logger.warning(f"Error loading langchain_huggingface, falling back: {e}")
                from langchain_community.embeddings import HuggingFaceEmbeddings
                self._embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
        return self._embeddings

    def _init_vectorstore(self):
        """Initialize or load the vector store (Pinecone if configured, else local FAISS)."""
        pinecone_key = os.getenv("PINECONE_API_KEY")
        embeddings = self._get_embeddings()

        if pinecone_key:
            try:
                from langchain_pinecone import PineconeVectorStore
                logger.info(f"Connecting to Pinecone index: {self.pinecone_index_name}")
                self._vectorstore = PineconeVectorStore.from_existing_index(
                    index_name=self.pinecone_index_name,
                    embedding=embeddings,
                )
                self._retriever = self._vectorstore.as_retriever(
                    search_type="similarity", search_kwargs={"k": 3}
                )
                logger.info("Successfully connected to Pinecone vector store.")
                return
            except Exception as exc:
                logger.warning(f"Could not connect to Pinecone index, falling back to FAISS: {exc}")

        # Local FAISS fallback
        from langchain_community.vectorstores import FAISS

        if os.path.exists(self.faiss_index_path) and os.path.exists(
            os.path.join(self.faiss_index_path, "index.faiss")
        ):
            logger.info(f"Loading persisted FAISS index from {self.faiss_index_path}")
            self._vectorstore = FAISS.load_local(
                self.faiss_index_path,
                embeddings,
                allow_dangerous_deserialization=True,
            )
        elif os.path.exists(self.data_path):
            logger.info(f"Building FAISS vector index from {self.data_path}")
            from langchain_core.documents import Document
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            with open(self.data_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)

            documents = []
            for entry in raw_data:
                content = entry.get("content", "")
                metadata = {
                    "source": entry.get("url", ""),
                    "title": entry.get("title", ""),
                }
                documents.append(Document(page_content=content, metadata=metadata))

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
            chunks = text_splitter.split_documents(documents)
            self._vectorstore = FAISS.from_documents(chunks, embeddings)
            os.makedirs(self.faiss_index_path, exist_ok=True)
            self._vectorstore.save_local(self.faiss_index_path)
            logger.info(f"Persisted new FAISS index to {self.faiss_index_path}")
        else:
            raise FileNotFoundError(
                f"Neither Pinecone nor local data found at {self.data_path}"
            )

        self._retriever = self._vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        )

    def _init_llm(self):
        """Initialize the LLM backend (Google Gemini or OpenAI)."""
        google_api_key = os.getenv("GOOGLE_GENAI_API_KEY") or os.getenv("GEMINI_API_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")

        if google_api_key:
            from langchain_google_genai import ChatGoogleGenerativeAI
            model_name = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash-lite")
            logger.info(f"Initializing ChatGoogleGenerativeAI with model: {model_name}")
            self._llm = ChatGoogleGenerativeAI(
                model=model_name,
                api_key=google_api_key,
                temperature=0.2,
            )
        elif openai_api_key:
            from langchain_openai import ChatOpenAI
            model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            logger.info(f"Initializing ChatOpenAI with model: {model_name}")
            self._llm = ChatOpenAI(
                model=model_name,
                api_key=openai_api_key,
                temperature=0.2,
            )
        else:
            self._llm = None
            logger.warning("No LLM API key (GOOGLE_GENAI_API_KEY or OPENAI_API_KEY) found in environment.")

    def initialize(self):
        """Initialize all pipeline components."""
        if not self._initialized:
            self._init_vectorstore()
            self._init_llm()
            self._build_chain()
            self._initialized = True

    def _build_chain(self):
        """Build the LangChain retrieval QA chain."""
        if self._llm is None or self._retriever is None:
            self._rag_chain = None
            return

        from langchain.chains import create_retrieval_chain
        from langchain_classic.chains.combine_documents import create_stuff_documents_chain
        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(self._llm, prompt)
        self._rag_chain = create_retrieval_chain(self._retriever, question_answer_chain)

    def _generate_suggestions(self, question: str, answer: str) -> List[str]:
        """Generate contextual follow-up prompt chips based on topic."""
        q_lower = question.lower()
        if any(w in q_lower for w in ["symptom", "sign", "feel", "thirst", "pee", "urinate"]):
            return [
                "How is diabetes diagnosed?",
                "What is a normal blood sugar level?",
                "What causes Type 2 diabetes?",
            ]
        elif any(w in q_lower for w in ["diet", "food", "eat", "meal", "breakfast", "sugar", "fruit"]):
            return [
                "Can I eat fruit with diabetes?",
                "Best exercises for diabetes",
                "How does water help blood sugar?",
            ]
        elif any(w in q_lower for w in ["exercise", "workout", "walk", "activity", "gym"]):
            return [
                "What are normal blood sugar targets?",
                "Foods that don't spike blood sugar",
                "How does stress affect diabetes?",
            ]
        elif any(w in q_lower for w in ["blood sugar", "glucose", "a1c", "hba1c", "level"]):
            return [
                "What causes blood sugar spikes?",
                "What should I eat for dinner?",
                "What are the symptoms of low blood sugar?",
            ]
        return DEFAULT_SUGGESTIONS

    def ask(self, question: str, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Process a user question through the live RAG retrieval and generation pipeline.

        Args:
            question: User inquiry string.
            history: Optional list of past messages.

        Returns:
            dict containing:
                "text": Final synthesized response string.
                "suggestions": Follow-up question chips.
                "sources": List of source URLs/documents.
                "rag_ready": Boolean status flag.
        """
        self.initialize()

        # Step 1: Retrieve context chunks
        retrieved_docs = []
        if self._retriever:
            try:
                retrieved_docs = self._retriever.invoke(question)
            except Exception as exc:
                logger.error(f"Error during retrieval: {exc}")

        sources = []
        for doc in retrieved_docs:
            src = doc.metadata.get("source") or doc.metadata.get("title")
            if src and src not in sources:
                sources.append(src)

        # Step 2: Generate response via LLM chain if available
        if self._rag_chain:
            try:
                result = self._rag_chain.invoke({"input": question})
                answer = result.get("answer", "")
                suggestions = self._generate_suggestions(question, answer)
                return {
                    "text": answer,
                    "suggestions": suggestions,
                    "sources": sources,
                    "rag_ready": True,
                }
            except Exception as exc:
                logger.error(f"Error during RAG chain execution: {exc}")
                # Fallback to retrieved context summary
                context_preview = (
                    "\n\n".join([f"• {d.page_content.strip()}" for d in retrieved_docs[:2]])
                    if retrieved_docs
                    else "No relevant medical context found."
                )
                return {
                    "text": (
                        "I retrieved relevant clinical data from NIDDK, but encountered an issue "
                        "synthesizing the final response with the language model.\n\n"
                        f"**Relevant medical context:**\n{context_preview}\n\n"
                        "_Please consult your healthcare provider for formal medical guidance._"
                    ),
                    "suggestions": DEFAULT_SUGGESTIONS,
                    "sources": sources,
                    "rag_ready": True,
                }

        # Step 3: Fallback if LLM API keys are not yet configured in .env
        if retrieved_docs:
            top_content = retrieved_docs[0].page_content.strip()
            return {
                "text": (
                    f"{top_content}\n\n"
                    "_Note: Live RAG vector retrieval active. To enable dynamic LLM generation, "
                    "configure `GOOGLE_GENAI_API_KEY` or `OPENAI_API_KEY` in your `.env` file._"
                ),
                "suggestions": self._generate_suggestions(question, top_content),
                "sources": sources,
                "rag_ready": True,
            }

        return {
            "text": (
                "I couldn't find specific information regarding your question in the diabetes medical database. "
                "Please consult a doctor or healthcare professional for personalized medical guidance."
            ),
            "suggestions": DEFAULT_SUGGESTIONS,
            "sources": [],
            "rag_ready": True,
        }


# Global singleton instance for easy route integration
rag_bot = RAGChatbot()
