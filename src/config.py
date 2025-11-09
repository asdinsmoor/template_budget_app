from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")