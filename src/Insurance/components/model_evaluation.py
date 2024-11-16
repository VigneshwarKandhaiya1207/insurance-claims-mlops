import pandas as pd
import numpy as np
import os
import mlflow
import joblib
from pathlib import Path
from dotenv import load_dotenv
from logger import logger
from src.Insurance.entity.config_entity import ModelEvaluationConfig
from src.Insurance.utils.utils import save_json
from sklearn.metrics import f1_score,precision_score,recall_score,classification_report,ConfusionMatrixDisplay


load_dotenv()


class ModelEvaluation:
    def __init__(self,config=ModelEvaluationConfig):
        self.config=config

    def get_model_evaluation(self,actual,pred):


        test_f1_score=f1_score(actual,pred)
        test_precision_score=precision_score(actual,pred)
        test_recall_score=recall_score(actual,pred)


        print("Train Classification Report:\n", classification_report(actual,pred))
        print("Validation Classification Report:\n", classification_report(actual,pred))

        return (
            test_f1_score,
            test_precision_score,
            test_recall_score
        )
    
    def log_into_mlflow(self):

        test_data = pd.read_csv(self.config.test_data_path)
        test_data=test_data.drop(self.config.columns_to_be_dropped,axis=1)
        model = joblib.load(self.config.model_name)

        x_test = test_data.drop(self.config.target_column, axis=1)
        y_test = test_data[self.config.target_column]

        with mlflow.start_run():

            predicted_qualities = model.predict(x_test)

            print(x_test.info())

            (test_f1_score, test_precision_score, test_recall_score) = self.get_model_evaluation(y_test, predicted_qualities)
            
            # Saving metrics as local
            scores = {
                    "test_f1_score": test_f1_score,
                    "test_precision_score": test_precision_score,
                    "test_recall_score": test_recall_score}
            save_json(path=Path(self.config.metrics_file), data=scores)

            mlflow.log_metric("test_f1_score", test_f1_score)
            mlflow.log_metric("test_precision_score", test_precision_score)
            mlflow.log_metric("test_recall_score", test_recall_score)


