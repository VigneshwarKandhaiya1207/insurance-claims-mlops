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

@dataclass
class DataTransformationConfig:
    root_dir: Path
    source_data_location: Path

@dataclass
class ModelTrainerConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    model_name: Path
    numerical_features: list
    one_hot_encoding: list
    label_encoding: list
    ordinal_encoding: list
    columns_to_be_dropped: list
    target_column: list

@dataclass
class ModelEvaluationConfig:
    root_dir: Path
    test_data_path: Path
    model_name: Path
    metrics_file: str
    target_column: list
    columns_to_be_dropped: list