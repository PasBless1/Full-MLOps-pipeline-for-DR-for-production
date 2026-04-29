from src.config.config import ConfigurationManager
from src.data.transformation import DataTransformation
from src.models.evaluate import ModelEvaluator
from src.utils.logger import logger


class EvaluationPipeline:
    def run(self):
        manager = ConfigurationManager()
        logger.info(">>> Stage 1: Load Test Data")
        dt = DataTransformation(manager.get_data_transformation_config())
        _, _, test_loader = dt.get_data_loaders()

        logger.info(">>> Stage 2: Evaluate Models")
        evaluator = ModelEvaluator(
            manager.get_model_evaluation_config(),
            manager.get_model_trainer_config(),
            manager.get_cascade_config(),
        )
        metrics = evaluator.evaluate(test_loader)

        print("\n📊 FINAL RESULTS")
        print("=" * 45)
        for name, m in metrics.items():
            print(f"  {name:<12} Accuracy: {m['accuracy']}  Kappa: {m['quadratic_kappa']}")
        print("=" * 45)
        return metrics


if __name__ == "__main__":
    EvaluationPipeline().run()
