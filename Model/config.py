from pathlib import Path

ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT.parent
DATA_FILE = PACKAGE / "Data" / "network_inputs_366.npz"
FORECAST_FILE = PACKAGE / "Data" / "forecast2026_inputs.npz"
OUTPUT_DIR = PACKAGE / "results"
RECIPE_DIR = ROOT / "training_recipe"
LINEAR_DIR = ROOT / "weights" / "linear"
FORECAST_ORIGIN = "2026-09"
FINAL_MODEL_SHA256 = (
    "05d21a64041969aa95632bfa8746a4d30603ee35786c76bd726ab83c598824d3"
)
