from src.Insurance.config.configurations import ConfigurationManager
from src.Insurance.components.model_evaluation import ModelEvaluation
from logger import logger


STAGE_NAME = "Model evaluation stage"
class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass

    def initiate_model_evaluation(self):
        config=ConfigurationManager()
        model_evaluation_config=config.get_model_evaluation_config()
        model_evaluation=ModelEvaluation(config=model_evaluation_config)
        model_evaluation.log_into_mlflow()