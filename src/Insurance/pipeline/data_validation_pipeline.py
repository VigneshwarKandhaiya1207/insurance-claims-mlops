from src.Insurance.entity.config_entity import DataValidationConfig
from src.Insurance.config.configurations import ConfigurationManager
from src.Insurance.components.data_validation import DataValidation
from logger import logger

STAGE_NAME="Data Validation Stage"

class DataValidationTrainingPipeline:
    def __init__(self):
        pass

    def initiate_data_validaion(self):
        config=ConfigurationManager()
        data_ingestion_config=config.get_data_validation_config()
        data_ingestion=DataValidation(config=data_ingestion_config)
        data_ingestion.validate_all_schemas()
