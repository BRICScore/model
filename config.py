from pathlib import Path

IDENTIFIER_DIR_PATH = Path("./data/identifier/")
IDENTIFIER_DIR_PATH.parent.mkdir(parents=True, exist_ok=True)
IDENTIFIER_FEATURES_PATH = IDENTIFIER_DIR_PATH / "features"
IDENTIFIER_FEATURES_PATH.mkdir(parents=True, exist_ok=True)
IDENTIFIER_RESULTS_PATH = IDENTIFIER_DIR_PATH / "clean"
IDENTIFIER_RESULTS_PATH.mkdir(parents=True, exist_ok=True)
UI_DATA_SOURCE_DIR_PATH = Path("./data/raw/")
UI_DATA_SOURCE_DIR_PATH.mkdir(parents=True, exist_ok=True)
