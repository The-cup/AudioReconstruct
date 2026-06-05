from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DATASET_DIR = DATA_DIR / "dataset"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"
LOGS_DIR = ARTIFACTS_DIR / "logs"
REPORTS_DIR = ARTIFACTS_DIR / "reports"

