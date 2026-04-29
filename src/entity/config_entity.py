from dataclasses import dataclass

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: str
    kaggle_dataset: str
    unzip_dir: str

@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: str
    status_file: str
    required_files: list

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: str
    data_path: str
    image_size: int
    batch_size: int
    num_workers: int

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: str
    trained_model_path: str
    epochs: int
    learning_rate: float
    early_stopping_patience: int
    num_classes: int
    backbone_lr_factor: float
    label_smoothing: float
    weight_decay: float
    dropout: float

@dataclass(frozen=True)
class CascadeModelConfig:
    root_dir: str
    stage1_model_dir: str
    stage1_model_path: str
    stage2_model_dir: str
    stage2_model_path: str
    epochs: int
    learning_rate: float
    early_stopping_patience: int
    backbone_lr_factor: float
    label_smoothing: float
    weight_decay: float
    dropout: float

@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: str
    metrics_path: str
