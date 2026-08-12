# Flask Application Scaffold with ML Support

A clean, modular Flask application initialized with the Application Factory pattern, Flask-SQLAlchemy, environment configuration, and Machine Learning deployment dependencies (`scikit-learn`, `pandas`, `numpy`, `joblib`, `gunicorn`).

## Project Structure

```text
myapp/
├── app/
│   ├── __init__.py      # App factory & SQLAlchemy initialization
│   ├── routes.py        # Main blueprint and application routes
│   ├── models.py        # SQLAlchemy database models
│   ├── config.py        # Environment configuration
│   ├── static/          # Static assets (CSS, JS, images)
│   └── templates/       # Jinja2 HTML templates
│       └── index.html   # Main index template
├── tests/
│   └── __init__.py      # Test package initialization
├── .env                 # Local environment variables (git-ignored)
├── .env.example         # Template for environment variables
├── .gitignore           # Git ignore settings
├── .flaskenv            # Flask CLI configuration
├── requirements.txt     # Pinned Python package dependencies (Flask, ML libraries, WSGI server)
├── run.py               # Entry point script
└── README.md            # Project documentation
```

## Setup & Installation

### 1. Create Virtual Environment

Create and activate a Python virtual environment:

```bash
# Navigate to project directory
cd myapp

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

### 2. Install Dependencies

Install pinned Python dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Installed ML packages include:
- `scikit-learn`: Model loading, inference, preprocessing pipelines.
- `pandas` & `numpy`: Data structures, array operations, and feature manipulation.
- `joblib`: Model serialization / deserialization.
- `gunicorn`: Production WSGI HTTP server for ML inference deployments.

### 3. Environment Setup

Copy `.env.example` to `.env` if not present:

```bash
cp .env.example .env
```

Customize your `.env` variables if necessary:
- `SECRET_KEY`: Application secret key for session signing.
- `DATABASE_URL`: SQLAlchemy database URI (defaults to SQLite: `sqlite:///app.sqlite3`).

### 4. Run the Application

#### Development Mode
```bash
flask run
# OR
python run.py
```

#### Production Deployment (Gunicorn WSGI Server)
```bash
gunicorn "app:create_app()" -w 4 -b 0.0.0.0:5000
```

The application will be accessible at `http://127.0.0.1:5000/`.
