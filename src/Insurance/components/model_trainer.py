import pandas as pd
import numpy as np
import os
import joblib
import mlflow
from logger import logger
from src.Insurance.entity.config_entity import ModelTrainerConfig
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer,SimpleImputer
from sklearn.preprocessing import RobustScaler,PowerTransformer,StandardScaler,OneHotEncoder,FunctionTransformer,LabelEncoder,OrdinalEncoder
from scipy.stats import skew
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split,GridSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN,SMOTETomek
from sklearn.pipeline import Pipeline as SklearnPipeline
from imblearn.pipeline import Pipeline as ImblearnPipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,VotingClassifier,StackingClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn.metrics import classification_report,f1_score,precision_score,recall_score,ConfusionMatrixDisplay


os.environ["MLFLOW_TRACKING_URI"]="https://dagshub.com/vigneshwar_kandhaiya/insurance-claims-mlops.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"]="vigneshwar_kandhaiya"
os.environ["MLFLOW_TRACKING_PASSWORD"]="5e52108ecd9bdea4007d036d3ccefc4af246697e"

class ModelTrainer:
    def __init__(self,config=ModelTrainerConfig):
        self.config=config


    def map_yes_no(self,x):
        return np.where(x == 'Yes', 1, 0)

    def evaluate_models(self,X_train,X_valid,y_train,y_valid,model_name,model_instance,param_grid=None,smote_type="SMOTE"):
        numerical_pipeline=SklearnPipeline(steps=[
            ("knnimpute",KNNImputer(n_neighbors=5)),
            ("scaler",StandardScaler())
        ])

        categorical_onehot_pipeline=SklearnPipeline(steps=[
            ("impute_na",SimpleImputer(strategy="most_frequent")),
            ("ohe_encoding",OneHotEncoder())
        ])

        categorical_label_pipeline=SklearnPipeline(steps=[
            ("impute_na",SimpleImputer(strategy="most_frequent")),
            ("label_encoding",FunctionTransformer(self.map_yes_no, validate=False))
        ])

        categorical_ordinal_pipeline=SklearnPipeline(steps=[
            ("impute_na",SimpleImputer(strategy="most_frequent")),
            ("ordinal_encoding",OrdinalEncoder())
        ])

        preprocessor=ColumnTransformer(transformers=[
            ("numerical_features",numerical_pipeline,self.config.numerical_features),
            ("cat_onehot_encoding",categorical_onehot_pipeline,self.config.one_hot_encoding),
            ("cat_label_encoding",categorical_label_pipeline,self.config.label_encoding),
            ("cat_ordinal_encoding",categorical_ordinal_pipeline,self.config.ordinal_encoding)
        ],
        remainder="passthrough")


        if smote_type == 'SMOTEENN': 
            smote = SMOTEENN() 
        else:
            smote = SMOTETomek()
        # smote = SMOTE()

        pipeline = ImblearnPipeline(
            steps=[
                ('preprocessor',preprocessor),
                ("smote",smote),
                ('{}'.format(model_name),model_instance)
            ]
        )
        with mlflow.start_run():
            if param_grid:
                grid_search=GridSearchCV(estimator=pipeline,
                                        param_grid=param_grid,
                                        cv=5,
                                        n_jobs=-1,
                                        scoring='f1',
                                        verbose=2)
                logger.info("Traing with hyperparameter Tuning for {}".format(model_name))
                grid_search.fit(X_train,y_train)
                best_model = grid_search.best_estimator_
                logger.info(f"Best hyperparameters for {model_name}: {grid_search.best_params_}")
                y_train_pred = best_model.predict(X_train)
                y_valid_pred = best_model.predict(X_valid)
                best_params = grid_search.best_params_
                for param, value in best_params.items():
                    mlflow.log_param(param, value)


            else:
                # No hyperparameter tuning, use the default model_instance
                logger.info(f"Training with {model_name}")
                pipeline.fit(X_train, y_train)
                y_train_pred = pipeline.predict(X_train)
                y_valid_pred = pipeline.predict(X_valid)
                best_params = None

            train_f1_score=f1_score(y_train,y_train_pred)
            train_precision_score=precision_score(y_train,y_train_pred)
            train_recall_score=recall_score(y_train,y_train_pred)

            test_f1_score=f1_score(y_valid,y_valid_pred)
            test_precision_score=precision_score(y_valid,y_valid_pred)
            test_recall_score=recall_score(y_valid,y_valid_pred)

            mlflow.log_metric("train_f1_score", train_f1_score)
            mlflow.log_metric("test_f1_score", test_f1_score)
            mlflow.log_metric("train_precision_score", train_precision_score)
            mlflow.log_metric("test_precision_score", test_precision_score)
            mlflow.log_metric("train_recall_score", train_recall_score)
            mlflow.log_metric("test_recall_score", test_recall_score)


            print("Train Classification Report:\n", classification_report(y_train,y_train_pred))
            print("Validation Classification Report:\n", classification_report(y_valid,y_valid_pred))

        return {'best_params': best_params,
                'train_f1_score': train_f1_score,
                'train_precision_score' : train_precision_score,
                'train_recall_score':train_recall_score,
                'test_f1_score':test_f1_score,
                'test_precision_score':test_precision_score,
                'test_recall_score': test_recall_score

                }



    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)
        
        train_data=train_data.drop(self.config.columns_to_be_dropped,axis=1)
        train,valid=train_test_split(train_data,test_size=0.2,random_state=42)

        X_train=train.drop(self.config.target_column,axis=1)
        X_valid=valid.drop(self.config.target_column,axis=1)
        y_train=train[self.config.target_column].values.ravel()
        y_valid=valid[self.config.target_column].values.ravel()


        param_grids = {
            'logistic_regression': {
                'logistic_regression__C': [0.01, 0.1, 1, 10],
                'logistic_regression__penalty': ['l2'],
                'logistic_regression__solver': ['lbfgs', 'saga'],
                'logistic_regression__max_iter': [10000, 20000]
            },
            'random_Forest': {
                'random_Forest__n_estimators': [100, 200],
                'random_Forest__max_depth': [None, 10, 20],
                'random_Forest__min_samples_split': [2, 5],
                'random_Forest__min_samples_leaf': [1, 2],
                'random_Forest__max_features': ['auto', 'sqrt']
            },
            'gradient_Boosting': {
                'gradient_Boosting__n_estimators': [100, 200],
                'gradient_Boosting__learning_rate': [0.01, 0.1, 0.2],
                'gradient_Boosting__max_depth': [3, 5, 7]
            },
            'ada_Boost': {
                'ada_Boost__n_estimators': [50, 100],
                'ada_Boost__learning_rate': [0.5, 1.0, 1.5]
            }
        }
        model_performance_report={}
        models_to_evaluate={
            'logistic_regression': LogisticRegression(),
            'random_Forest': RandomForestClassifier(),
            'gradient_Boosting': GradientBoostingClassifier(),
            'ada_Boost' : AdaBoostClassifier()
        }


        for model_name,model_instance in models_to_evaluate.items():
            param_grid = param_grids.get(model_name, None)
            model_performance_report[model_name]=self.evaluate_models(X_train,X_valid,y_train,y_valid,model_name,model_instance,param_grid=param_grid,smote_type="SMOTETomek")

        best_model_name = max(model_performance_report, key=lambda x: model_performance_report[x]['test_f1_score'])

        best_model_metrics = model_performance_report[best_model_name]
        best_model_params = best_model_metrics['best_params']
        final_scores=pd.DataFrame(model_performance_report)
        print(final_scores)
        print(final_scores.T)

        joblib.dump(best_model_name, os.path.join(self.config.root_dir, self.config.model_name))

