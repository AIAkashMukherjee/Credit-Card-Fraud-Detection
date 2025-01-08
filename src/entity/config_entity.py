from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    local_data_file: Path
    train_file_path:Path
    test_file_path:Path

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    preprocessor_obj: Path
    train_file_path:Path
    test_file_path:Path
    save_train_path: Path
    save_test_path: Path


@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    train_model_path: Path
    training_data_path: Path
    testing_data_path: Path
    
@dataclass(frozen=True)
class EvaluationConfig:
    root_dir: Path
    model_path:Path
    test_data:Path
    mlflow_uri: str       