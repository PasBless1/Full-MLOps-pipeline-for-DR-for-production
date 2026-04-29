from src.config.config import ConfigurationManager
from src.data.ingestion import DataIngestion
from src.data.validation import DataValidation
from src.data.transformation import DataTransformation
from src.models.cascade_trainer import CascadeTrainer
from src.utils.logger import logger


class CascadeTrainingPipeline:
    def run(self):
        manager = ConfigurationManager()

        logger.info(">>> Stage 1: Data Ingestion")
        DataIngestion(manager.get_data_ingestion_config()).run()

        logger.info(">>> Stage 2: Data Validation")
        ok = DataValidation(manager.get_data_validation_config()).validate()
        if not ok:
            raise Exception("Data validation failed.")

        logger.info(">>> Stage 3: Data Transformation")
        dt = DataTransformation(manager.get_data_transformation_config())
        train_loader, val_loader, _ = dt.get_data_loaders()

        logger.info(">>> Stage 4: Cascade Training (Stage1Net → Stage2Net)")
        CascadeTrainer(manager.get_cascade_config()).train(train_loader, val_loader)
        logger.info("Cascade pipeline complete.")


if __name__ == "__main__":
    CascadeTrainingPipeline().run()
