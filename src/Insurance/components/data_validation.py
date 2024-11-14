import pandas as pd
import os
from logger import logger
from src.Insurance.entity.config_entity import DataValidationConfig


class DataValidation:
    def __init__(self,config:DataValidationConfig):
        self.config=config

    def validate_all_schemas(self)-> bool:
        try:
            validation_status=None

            data=pd.read_csv(self.config.source_data_location)
            all_cols=data.columns.to_list()
            all_schema=self.config.all_schema.keys()

            for col in all_cols:
                if col not in all_schema:
                    validation_status=False
                else:
                    validation_status=True
            
            with open(self.config.status_file,"w") as status_file_appender:
                status_file_appender.write("Validation Status : {}".format(validation_status))
            
            
            return validation_status
        except Exception as e:
            logger.exception(e)
            raise e
