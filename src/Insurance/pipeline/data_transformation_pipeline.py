from logger import logger
from src.Insurance.config.configurations import ConfigurationManager
from src.Insurance.components.data_transformation import DataTransformation

STAGE_NAME= "Data Transformation Stage"

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def initiate_data_transformation(self):
        config=ConfigurationManager()
        data_transformation_config=config.get_data_transformation_config()
        data_transformation=DataTransformation(config=data_transformation_config)
        data_transformation.train_test_splitting()
