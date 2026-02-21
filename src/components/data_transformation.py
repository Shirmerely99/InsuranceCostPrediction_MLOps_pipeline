import os 
import sys
from src.logger import logging
from src.exception import CustomException

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from dataclasses import dataclass

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    transformer_path = os.path.join('artifacts',"transformer.pkl")

class DataTransformation:
    def __init__ (self):
        self.data_transformation = DataTransformationConfig()

    def get_data_transformer(self):
        try:
            num_columns = ['age', 'bmi', 'children', 'obesity', 'smoker_obese']
            cat_columns = ['sex', 'smoker', 'region']
            
            num_pipeline = Pipeline([
                ('scaler', StandardScaler())
                ])
            
            cat_pipeline = Pipeline([
                ('ohe', OneHotEncoder(drop='first', handle_unknown='ignore'))
                ])
            
            transformer = ColumnTransformer([
                ('num', num_pipeline, num_columns),
                ('cat', cat_pipeline, cat_columns)
                ])

            return transformer
        
        except Exception as e:
            raise CustomException(e,sys)
        

    def initiate_data_transformation(self, train_data, test_data):
        logging.info("Data Tranformation initiated")
        try:
            train_data = pd.read_csv(train_data)
            test_data = pd.read_csv(test_data)
            logging.info("training and testing dataset read as dataframe")

            target_column = "charges"
            num_column = ['age', 'bmi', 'children', 'obesity', 'smoker_obese']
            
            X_train = train_data.drop(columns=[target_column])
            y_train = train_data[target_column]

            X_test = test_data.drop(columns=[target_column])
            y_test = test_data[target_column]

            transformer_obj = self.get_data_transformer()
            X_train_arr = transformer_obj.fit_transform(X_train)
            X_test_arr = transformer_obj.transform(X_test)
            logging.info("Applied the transforming object on train and test dataframe.")
            
            train_arr = np.c_[X_train_arr, np.array(y_train)]
            test_arr = np.c_[X_test_arr, np.array(y_test)]

            save_object(file_path= self.data_transformation.transformer_path,
                        obj= transformer_obj)
            logging.info("Saved the transforming object.")
            
            logging.info("Data Transformation completed")
            
            return (train_arr,
                    test_arr,
                    self.data_transformation.transformer_path)
        
        except Exception as e:
            raise CustomException(e,sys)