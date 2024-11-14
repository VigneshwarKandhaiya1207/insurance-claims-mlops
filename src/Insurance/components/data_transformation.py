import os
import pandas as pd
from logger import logger
from src.Insurance.entity.config_entity import DataTransformationConfig
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.compose import ColumnTransformer

class DataTransformation:
    def __init__(self,config:DataTransformationConfig):
        self.config=config
    
    def train_test_splitting(self):
        data=pd.read_csv(self.config.source_data_location)

        logger.info("Initiating Train test split")
        train,test=train_test_split(data,test_size=0.2,random_state=42)
        logger.info("Writting train and test data to CSV file.")
        train.to_csv(os.path.join(self.config.root_dir,"train.csv"),index = False)
        test.to_csv(os.path.join(self.config.root_dir,"test.csv"),index = False)

        logger.info("Train test split completed.")
        logger.info(train.shape)
        logger.info(test.shape)


