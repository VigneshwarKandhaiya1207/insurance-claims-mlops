import os
from src.Insurance.entity.config_entity import DataIngestionConfig
from logger import logger
from urllib import request

class DataIngestion:
    def __init__(self,config:DataIngestionConfig):
        self.config=config

    def download_file(self):
        logger.info("Trying to download the data from {}".format(self.config.source_URL))
        try:
            if not os.path.exists(self.config.local_directory_file):
                filename,headers=request.urlretrieve(
                    url=self.config.source_URL,
                    filename=self.config.local_directory_file
                )
                logger.info("Data downloaded successfully to the path {}".format(self.config.local_directory_file))
            else:
                logger.info("Data already exists!!")
        except Exception as e:
            raise e
            
