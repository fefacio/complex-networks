from pathlib import Path

# DIRECTORIES

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" 
RAW_DATA_DIR = DATA_DIR / "raw"
GRAPHML_DATA_DIR = DATA_DIR / "graphml"
CPP_DIR = PROJECT_ROOT / "libs"

RESULTS_DIR = PROJECT_ROOT / "results"