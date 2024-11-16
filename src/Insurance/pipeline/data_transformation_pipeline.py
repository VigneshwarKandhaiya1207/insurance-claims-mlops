from logger import logger
from src.Insurance.config.configurations import ConfigurationManager
from src.Insurance.components.data_transformation import DataTransformation

STAGE_NAME= "Data Transformation Stage"

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def initiate_data_transformation(self):
        try:
            with open("artifacts/datavalidation/status.txt", "r") as status_check:
                status=status_check.read().split(":")[-1].strip()
                if status=="True":
                    config=ConfigurationManager()
                    data_transformation_config=config.get_data_transformation_config()
                    data_transformation=DataTransformation(config=data_transformation_config)
                    data_transformation.train_test_splitting()
                else:
                    logger.info("Schema Validation Failed.")
                    raise Exception("Your data scheme is not valid")
        except Exception as e:
            print(e)
