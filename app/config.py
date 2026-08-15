import os
from dotenv import load_dotenv

# Base directory of the application
basedir = os.path.abspath(os.path.dirname(__file__))
project_root = os.path.dirname(basedir)

# Load environment variables from .env in project root
load_dotenv(os.path.join(project_root, '.env'))


class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'default-dev-secret-key'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or f"sqlite:///{os.path.join(project_root, 'app.sqlite3')}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # ── ML Model paths ─────────────────────────────────────────────────────
    # Place your trained model.pkl and scaler.pkl inside the models/ directory
    # at the project root before uncommenting the TODO blocks in routes.py.
    MODEL_PATH  = os.environ.get('MODEL_PATH')  or os.path.join(project_root, 'models', 'model.pkl')
    SCALER_PATH = os.environ.get('SCALER_PATH') or os.path.join(project_root, 'models', 'scaler.pkl')
