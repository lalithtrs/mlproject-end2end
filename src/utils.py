import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException
import dill
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}

        for model_name, model in models.items():
            try:
                logging.info(f"Evaluating model: {model_name}")
                
                para = params[model_name]

                grid = GridSearchCV(estimator=model, param_grid=para, cv=3)
                grid.fit(X_train, y_train)

                model.set_params(**grid.best_params_)
                model.fit(X_train, y_train)

                # Make predictions
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                # Compute R2 scores
                train_r2_score = r2_score(y_train, y_train_pred)
                test_r2_score = r2_score(y_test, y_test_pred)

                # Store the test R2 score in the report
                report[model_name] = test_r2_score

                logging.info(f"Model {model_name} evaluation completed. Test R2 Score: {test_r2_score}")
                logging.info(f"Best parameters for {model_name}: {grid.best_params_}")

            except Exception as model_error:
                logging.error(f"An error occurred while evaluating {model_name}: {str(model_error)}")
                report[model_name] = None

        return report
    
    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e,sys) 