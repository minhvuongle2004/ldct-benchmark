"""
Run the EDR-REDNet Visualizer Web App.
Usage: python run_app.py
"""
import os
import sys
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
WEBAPP_DIR = os.path.join(ROOT, "webapp")
sys.path.insert(0, ROOT)

if __name__ == "__main__":
    print("=" * 55)
    print("  EDR-REDNet Visualizer")
    print("  Open browser at: http://localhost:8000")
    print("=" * 55)
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "api:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload",
    ], cwd=WEBAPP_DIR)
