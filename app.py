from pathlib import Path
import sys

APP_DIR = Path(__file__).resolve().parent / "app"
sys.path.insert(0, str(APP_DIR))

from ui import demo

demo.launch()