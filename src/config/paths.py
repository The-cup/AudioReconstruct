from pathlib import Path


PROJECT_ROOT = Path("D:\\projects\\python\\AudioReconstruct")
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
LOW_FREQ_DATA_DIR = RAW_DATA_DIR / "low_freq"
SELECTED_EMBEDDED_DIR = DATA_DIR / "selected"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DATASET_DIR = DATA_DIR / "dataset"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"
LOGS_DIR = ARTIFACTS_DIR / "logs"
REPORTS_DIR = ARTIFACTS_DIR / "reports"

