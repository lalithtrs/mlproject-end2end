# Data Processing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys, os

# Modelling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRFRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV

from src.exception import CustomException
from src.utils import save_object, evaluate_model
from src.logger import logging
from dataclasses import dataclass


@dataclass
class ModelTrainerConfig:
    trsined_model_file_path = os.path.join('artifacts', "model.pkl")

class ModelTrain:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()


    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info("Spliting train and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]

            )

            models = {
                "Random Forest" : RandomForestRegressor(),
                "Linear Regression" : LinearRegression(),
                "Ridge": Ridge(),
                "Lasso" : Lasso(),
                "DecisionTreeRegressor" : DecisionTreeRegressor(),
                "AdaBoostRegressor": AdaBoostRegressor(),
                "SVR" : SVR()
            }

            model_report:dict=evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,\
                                            models=models)
            
            # To get the best model
            best_model_score = max(sorted(model_report.values()))

            # To get best model name from dict
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException(f"No best models are found")
            
            logging.info(f"Best found model on both training and testing dataset")


            save_object(
                file_path=self.model_trainer_config.trsined_model_file_path,
                obj=best_model
            )

            prediction = best_model.predict(X_test)

            r2 = r2_score(y_test, prediction)

            return r2
    
        except Exception as e:
            raise CustomException(e, sys)

