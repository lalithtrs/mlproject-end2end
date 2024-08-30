import pandas as pd
import sys, os
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline():
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Construct the path to the model file
        self.model_path = os.path.join(current_dir, '..', 'components', 'artifacts', 'model.pkl')
        self.preprocessor_path = os.path.join(current_dir, '..', 'components', 'artifacts', 'preprocessor.pkl')

        # Ensure the path is absolute and normalized
        self.model_path = os.path.abspath(self.model_path)
        self.preprocessor_path = os.path.abspath(self.preprocessor_path)

    def predict(self, features):
        try:
            model_path = self.model_path
            preprocessor_path = self.preprocessor_path
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            # Convert features to a DataFrame with column names
            data_df = pd.DataFrame(features, columns=["gender", "race/ethnicity", "parental level of education", "lunch", "test preparation course", "math score", "reading score", "writing score"])

            data_scaled = preprocessor.transform(data_df)
            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            raise CustomException(e, sys)

class CustomData():
    def __init__(self,
                 gender: str,
                 race_ethnicity: str,
                 parental_level_of_education,
                 lunch: str,
                 test_preparation_course: str,
                 math_score: int,
                 reading_score: int,
                 writing_score: int
                 ):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.math_score = math_score
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_df(self):
        try:
            custom_data = {
                "gender": [self.gender],
                "race": [self.race_ethnicity],
                "parent_edu": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_prep": [self.test_preparation_course],
                "math score": [self.math_score],
                "reading score": [self.reading_score],
                "writing score": [self.writing_score]
            }

            return pd.DataFrame(custom_data)

        except Exception as e:
            raise CustomException(e, sys)