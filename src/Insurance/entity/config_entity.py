from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_directory_file: Path

@dataclass
class DataValidationConfig:
    root_dir: Path
    source_data_location: Path
    status_file: str
    all_schema: dict