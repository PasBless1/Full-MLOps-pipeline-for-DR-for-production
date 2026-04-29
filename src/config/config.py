from src.utils.common import read_yaml, create_directories
from src.entity.config_entity import (
    DataIngestionConfig, DataValidationConfig, DataTransformationConfig,
    ModelTrainerConfig, CascadeModelConfig, ModelEvaluationConfig,
)

CONFIG_PATH = "config/config.yaml"
PARAMS_PATH = "params.yaml"

class ConfigurationManager:
    def __init__(self, config_path=CONFIG_PATH, params_path=PARAMS_PATH):
        self.config = read_yaml(config_path)
        self.params = read_yaml(params_path)
        create_directories([
            self.config["data_ingestion"]["root_dir"],
            self.config["data_validation"]["root_dir"],
            self.config["data_transformation"]["root_dir"],
            self.config["model_trainer"]["root_dir"],
            self.config["cascade_model"]["root_dir"],
            self.config["model_evaluation"]["root_dir"],
            "models/cascade", "reports/metrics",
        ])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        c = self.config["data_ingestion"]
        return DataIngestionConfig(**c)

    def get_data_validation_config(self) -> DataValidationConfig:
        c = self.config["data_validation"]
        return DataValidationConfig(**c)

    def get_data_transformation_config(self) -> DataTransformationConfig:
        c = self.config["data_transformation"]
        return DataTransformationConfig(
            root_dir    = c["root_dir"],
            data_path   = c["data_path"],
            image_size  = self.params["IMAGE_SIZE"],
            batch_size  = self.params["BATCH_SIZE"],
            num_workers = c["num_workers"],
        )

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        c = self.config["model_trainer"]
        p = self.params
        return ModelTrainerConfig(
            root_dir                 = c["root_dir"],
            trained_model_path       = c["trained_model_path"],
            epochs                   = p["EPOCHS"],
            learning_rate            = p["LEARNING_RATE"],
            early_stopping_patience  = p["EARLY_STOPPING_PATIENCE"],
            num_classes              = p["NUM_CLASSES"],
            backbone_lr_factor       = p["BACKBONE_LR_FACTOR"],
            label_smoothing          = p["LABEL_SMOOTHING"],
            weight_decay             = p["WEIGHT_DECAY"],
            dropout                  = p["DROPOUT"],
        )

    def get_cascade_config(self) -> CascadeModelConfig:
        c = self.config["cascade_model"]
        p = self.params
        return CascadeModelConfig(
            root_dir                 = c["root_dir"],
            stage1_model_dir         = c["stage1_model_dir"],
            stage1_model_path        = c["stage1_model_path"],
            stage2_model_dir         = c["stage2_model_dir"],
            stage2_model_path        = c["stage2_model_path"],
            epochs                   = p["EPOCHS"],
            learning_rate            = p["LEARNING_RATE"],
            early_stopping_patience  = p["EARLY_STOPPING_PATIENCE"],
            backbone_lr_factor       = p["BACKBONE_LR_FACTOR"],
            label_smoothing          = p["LABEL_SMOOTHING"],
            weight_decay             = p["WEIGHT_DECAY"],
            dropout                  = p["DROPOUT"],
        )

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        c = self.config["model_evaluation"]
        return ModelEvaluationConfig(**c)
