import os 
import sys
from src.exception import customException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split

from dataclasses import dataclass

from src.Components.data_transformation import DataTransformation
from src.Components.data_transformation import dataTransformationConfig
from src.Components.model_traniner import ModelTrainer
from src.Components.model_traniner import ModelTrainingConfig


@dataclass
class DataIngestionConfig:
    train_data_path : str = os.path.join('artifact',"train.csv")
    test_data_path : str = os.path.join('artifact',"test.csv")
    raw_data_path : str = os.path.join('artifact',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Enter the data ingestion method or component")
        try:
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train test split initialized")
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index = False , header = True)
            test_set.to_csv(self.ingestion_config.test_data_path,index = False , header = True)

            logging.info("Ingestion of the data is completed")
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise customException(e,sys)
        
if __name__ == "__main__":

    obj = DataIngestion()
    train_data,test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    #train_array,test_array = data_transformation.initiate_data_transformation(train_data,test_data)
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    modelTrainer = ModelTrainer()
    print(modelTrainer.initiate_model_trainier(train_arr,test_arr))