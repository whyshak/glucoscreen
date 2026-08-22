import os
import sys
import subprocess

# Auto-switch to virtual environment python if available
_venv_python = os.path.abspath(os.path.join(os.path.dirname(__file__), 'venv', 'Scripts', 'python.exe'))
if os.path.exists(_venv_python) and os.path.abspath(sys.executable) != _venv_python:
    sys.exit(subprocess.call([_venv_python] + sys.argv[1:]))

from app import create_app

app = create_app()

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)


