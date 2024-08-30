import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys, os
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
from sklearn.impute import SimpleImputer
from src.exception import CustomException
from src.utils import save_object, evaluate_model
from src.logger import logging
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', "model.pkl")

class ModelTrain:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def preprocess_data(self, X, y):
        logging.info("Starting data preprocessing")
        
        # Check for infinite values
        inf_mask = np.isinf(X)
        if np.any(inf_mask):
            logging.warning(f"Found {np.sum(inf_mask)} infinite values in features. Replacing with NaN.")
            X[inf_mask] = np.nan

        # Check for NaN values
        nan_mask = np.isnan(X)
        if np.any(nan_mask):
            logging.warning(f"Found {np.sum(nan_mask)} NaN values in features.")
            
            # Impute NaN values
            imputer = SimpleImputer(strategy='mean')
            X = imputer.fit_transform(X)
            logging.info("Imputed NaN values with mean strategy")

        # Check for NaN or infinite values in target variable
        invalid_y = np.isnan(y) | np.isinf(y)
        if np.any(invalid_y):
            logging.warning(f"Found {np.sum(invalid_y)} invalid values in target variable. Removing corresponding samples.")
            X = X[~invalid_y]
            y = y[~invalid_y]

        logging.info("Data preprocessing completed")
        return X, y

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info("Splitting train and test input data")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            # Preprocess the data
            X_train, y_train = self.preprocess_data(X_train, y_train)
            X_test, y_test = self.preprocess_data(X_test, y_test)

            models = {
                "Random Forest": RandomForestRegressor(),
                "Linear Regression": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "Decision Tree": DecisionTreeRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "SVR": SVR()
            }
            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
                },
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting": {
                    'learning_rate': [.1, .01, .05, .001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Linear Regression": {},
                "Ridge": {'alpha': [0.1, 1.0, 10.0]},
                "Lasso": {'alpha': [0.1, 1.0, 10.0]},
                "AdaBoost": {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1.0]},
                "SVR": {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']}
            }

            model_report = evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                          models=models, params=params)
            
            # Filter out None values and log any models that failed
            valid_model_report = {name: score for name, score in model_report.items() if score is not None}
            
            if not valid_model_report:
                raise CustomException("No models produced valid scores.")

            best_model_score = max(valid_model_report.values())
            best_model_name = max(valid_model_report, key=valid_model_report.get)
            best_model = models[best_model_name]

            logging.info(f"Best found model: {best_model_name} with score: {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            prediction = best_model.predict(X_test)
            r2 = r2_score(y_test, prediction)
            return r2

        except Exception as e:
            logging.error(f"An error occurred during model training: {str(e)}")
            raise CustomException(e, sys)