# Flask Application Scaffold

A clean, modular Flask application initialized with the Application Factory pattern, Flask-SQLAlchemy, and environment configuration.

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
├── requirements.txt     # Pinned Python package dependencies
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

### 3. Environment Setup

Copy `.env.example` to `.env` if not present:

```bash
cp .env.example .env
```

Customize your `.env` variables if necessary:
- `SECRET_KEY`: Application secret key for session signing.
- `DATABASE_URL`: SQLAlchemy database URI (defaults to SQLite: `sqlite:///app.sqlite3`).

### 4. Run the Application

You can run the application using either the Flask CLI or direct Python execution:

#### Using Flask CLI
```bash
flask run
```

#### Using Python Entry Point
```bash
python run.py
```

The application will be accessible at `http://127.0.0.1:5000/`.
