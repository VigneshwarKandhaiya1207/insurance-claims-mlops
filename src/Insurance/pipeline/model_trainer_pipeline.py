from src.Insurance.entity.config_entity import ModelTrainerConfig
from src.Insurance.config.configurations import ConfigurationManager
from src.Insurance.components.model_trainer import ModelTrainer
from logger import logger

STAGE_NAME="Model Trainer Stage"

class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def initate_model_training(self):
        config=ConfigurationManager()
        model_trainer_config=config.get_model_trainer_config()
        model_trainer=ModelTrainer(config=model_trainer_config)
        model_trainer.train()
