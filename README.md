# GlucoScreen — Diabetes Risk Assessment & Health Assistant

A clean, modular Flask application initialized with the Application Factory pattern, Flask-SQLAlchemy, machine learning inference (`scikit-learn`, `pandas`, `numpy`, `joblib`, `shap`), batch processing, model evaluation, and an intelligent RAG-powered chatbot (Dia) backed by LangChain, Google Gemini / OpenAI, and Pinecone / FAISS.

---

## Project Structure

```text
glucoscreen/
├── app/
│   ├── __init__.py           # App factory & SQLAlchemy initialization
│   ├── routes.py             # Main blueprint, inference, batch & chat endpoints
│   ├── models.py             # SQLAlchemy database models
│   ├── config.py             # Environment & Flask configuration
│   ├── batch_service.py      # Batch screening processor & CSV/Excel handler
│   ├── evaluation_service.py # Model evaluation & benchmark metric service
│   ├── rag_service.py        # Dia AI RAG pipeline (LangChain, LLMs, Vector Stores)
│   ├── static/               # Static assets (CSS, JS, icons)
│   └── templates/            # Jinja2 HTML templates (screening, results, batch, etc.)
├── data/
│   ├── niddk_diabetes.json   # NIDDK health knowledge base for RAG
│   └── faiss_index/          # Local FAISS vector index (auto-built fallback)
├── models/
│   ├── best_svm_model.pkl    # Trained SVM diabetes classification model
│   ├── standard_scaler.pkl   # Fitted StandardScaler for numerical features
│   └── shap_explainer.pkl    # Pre-computed SHAP Tree/Kernel Explainer
├── tests/
│   ├── test_batch.py         # Batch processing unit tests
│   ├── test_chat.py          # Dia RAG chat & health assistant tests
│   └── test_debug_evaluate.py# Evaluation endpoint & metrics tests
├── .env                      # Local environment variables (git-ignored)
├── .env.example              # Template for environment variables
├── .flaskenv                 # Flask CLI configuration
├── .gitignore                # Git ignore settings
├── Dockerfile                # Production container specification
├── requirements.txt          # Python package dependencies
├── run.py                    # Application entry point script
└── README.md                 # Project documentation
```

---

## Setup & Installation

### 1. Create Virtual Environment

```bash
# Navigate to project directory
cd glucoscreen

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

### 2. Install Dependencies

Install the curated dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## Environment Configuration

Copy `.env.example` to create your local `.env` file:

```bash
cp .env.example .env
```

### Environment Variables Reference

| Variable | Required | Default / Example | Description |
| :--- | :--- | :--- | :--- |
| **`SECRET_KEY`** | Yes | `your-secret-key-here` | Secret key used for cryptographic session signing. |
| **`DATABASE_URL`** | No | `sqlite:///app.sqlite3` | SQLAlchemy database connection URI. |
| **`FLASK_APP`** | No | `run.py` | Flask application entry point. |
| **`FLASK_ENV`** | No | `development` | Flask runtime environment (`development` / `production`). |
| **`GOOGLE_GENAI_API_KEY`** | Optional* | `your_google_genai_api_key` | Google Gemini API key for Dia AI health chatbot. *(Recommended)* |
| **`GOOGLE_MODEL`** | No | `gemini-2.5-flash-lite` | Gemini model variant to use. |
| **`OPENAI_API_KEY`** | Optional* | `your_openai_api_key` | OpenAI API key (used if `GOOGLE_GENAI_API_KEY` is not provided). |
| **`OPENAI_MODEL`** | No | `gpt-4o-mini` | OpenAI model variant to use. |
| **`PINECONE_API_KEY`** | Optional | `your_pinecone_api_key` | Pinecone vector DB key. If omitted, the app automatically falls back to local FAISS. |
| **`PINECONE_INDEX_NAME`** | No | `medibot` | Pinecone index name for knowledge embeddings. |
| **`PORT`** | No | `8080` | Port used by Docker / Cloud Run deployments (defaults to `8080` in container). |

> **\*Note on Dia RAG Chatbot:** Provide either `GOOGLE_GENAI_API_KEY` or `OPENAI_API_KEY` to enable AI conversational answers. If neither key is present, the app gracefully falls back to structured FAQ matching.

---

## Running the Application

### Development Mode
```bash
flask run
# OR
python run.py
```
The development server will start at `http://127.0.0.1:5000/`.

### Production Deployment (Gunicorn)
```bash
gunicorn "app:create_app()" --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:8080
```

### Docker Container
```bash
# Build Docker image
docker build -t glucoscreen .

# Run Docker container with environment file
docker run -p 8080:8080 --env-file .env glucoscreen
```

---

## Running Tests

Run the full automated test suite:

```bash
python -m unittest discover tests
```
