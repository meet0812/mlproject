import sys 
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import customException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class dataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = dataTransformationConfig()
    
    def get_data_transformer_object(self):
        try:
            numerical_column = ["writing_score","reading_score"]
            categorial_column = ["gender","race_ethnicity","parental_level_of_education","lunch","test_preparation_course"]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scalar",StandardScaler())
                ]        
            )
            cat_pipeline = Pipeline(
                    steps=[
                        ("imputer",SimpleImputer(strategy="most_frequent")),
                        ("one_hot_encoder",OneHotEncoder()),
                        ("scalar",StandardScaler(with_mean=False))
                    ]
                )
            
            # logging.info("Numerical column encoding completed")
            # logging.info("Categorial column encoding completed")
            logging.info(f"Categorial column: {categorial_column}")
            logging.info(f"Numerical column: {numerical_column}")

            preprocessor = ColumnTransformer(
                [
                    ("Numerical_pipeline",num_pipeline,numerical_column),
                    ("cat_pipeline",cat_pipeline,categorial_column)
                ]
            )

            return preprocessor
        except Exception as e:
            raise customException(e,sys)
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaning preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "math_score"
            numerical_column_name = ["writing_score","reading_score"]

            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_name = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_name = test_df[target_column_name]

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe")

            input_feature_train_array = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_array = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_array,np.array(target_feature_train_name)
            ]

            test_arr = np.c_[input_feature_test_array,np.array(target_feature_test_name)]

            logging.info(f"Saved preprocessing object")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return(train_arr,test_arr,self.data_transformation_config.preprocessor_obj_file_path)


        except Exception as e:
            raise customException(e,sys)
            