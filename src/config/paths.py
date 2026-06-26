from pathlib import Path

PROJECT_ROOT: Path | None = Path("D:\\projects\\python\\AudioReconstruct")
DATA_DIR: Path | None = None
RAW_DATA_DIR: Path | None = None
LOW_FREQ_DATA_DIR: Path | None = None
SELECTED_EMBEDDED_DIR: Path | None = None
PROCESSED_DATA_DIR: Path | None = None

ARTIFACTS_DIR: Path | None = None
CHECKPOINTS_DIR: Path | None = None
LOGS_DIR: Path | None = None
REPORTS_DIR: Path | None = None

def build_dir_path(project_path: Path):
    global DATA_DIR, RAW_DATA_DIR, LOW_FREQ_DATA_DIR, SELECTED_EMBEDDED_DIR, PROCESSED_DATA_DIR, \
        ARTIFACTS_DIR, CHECKPOINTS_DIR, LOGS_DIR, REPORTS_DIR
    DATA_DIR = project_path / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    LOW_FREQ_DATA_DIR = DATA_DIR / "low_freq"
    SELECTED_EMBEDDED_DIR = DATA_DIR / "selected"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"

    ARTIFACTS_DIR = DATA_DIR / "artifacts"
    CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"
    LOGS_DIR = ARTIFACTS_DIR / "logs"
    REPORTS_DIR = ARTIFACTS_DIR / "reports"

