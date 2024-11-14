from src.Insurance.entity.config_entity import DataIngestionConfig
from src.Insurance.config.configurations import ConfigurationManager
from src.Insurance.components.data_ingestion import DataIngestion
from logger import logger

STAGE_NAME="Data Ingestion Stage"

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def initiate_data_ingestion(self):
        config=ConfigurationManager()
        data_ingestion_config=config.get_dataingestion_config()
        data_ingestion=DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()