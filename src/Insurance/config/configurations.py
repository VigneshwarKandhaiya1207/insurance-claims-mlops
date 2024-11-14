from src.Insurance.constants import *
from src.Insurance.utils.utils import create_directories,read_yaml
from src.Insurance.entity.config_entity import DataIngestionConfig,DataValidationConfig
from logger import logger


class ConfigurationManager:
    def __init__(self,config_file_path=CONFIG_FILE_PATH,
                 schema_file_path=SCHEMA_FILE_PATH):
        self.config=read_yaml(config_file_path)
        self.schema=read_yaml(schema_file_path)

        create_directories([self.config.artifact_root])

    def get_dataingestion_config(self)-> DataIngestionConfig:
        config=self.config.data_ingestion
        create_directories([config.root_dir])

        data_ingestion_config=DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_directory_file=config.local_directory_file
        )

        return data_ingestion_config
    
    def get_data_validation_config(self)-> DataValidationConfig:
        config=self.config.data_validation
        schema=self.schema.COLUMNS


        create_directories([config.root_dir])

        data_validation_config=DataValidationConfig(
            root_dir=config.root_dir,
            source_data_location=config.source_data_location,
            status_file=config.status_file,
            all_schema=schema
        )

        return data_validation_config

