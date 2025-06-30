import os 
import sys
from dataclasses import dataclass

from catboost import CatBoostClassifier
from sklearn.ensemble import(AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import customException
from src.logger import logging
from src.utils import evalute_models

from src.utils import save_object

@dataclass
class ModelTrainingConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainingConfig()

    def initiate_model_trainier(self,train_array,test_array):
        try:
            logging.info("Spliting training and test input data")

            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Random Forest":RandomForestRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "Gradiant Boosting":GradientBoostingRegressor(),
                "Linear Regression":LinearRegression(),
                "K-Neighbour Classifier":KNeighborsRegressor(),
                "XCBoosting classfier":XGBRegressor(),
                "Cat Boosting":CatBoostClassifier(),
                "AdaBoosting classfier":AdaBoostRegressor()
            }
            model_report:dict=evalute_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models)

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            if best_model_score<0.6:
                raise customException("No best model found")
            
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            r2score = r2_score(y_test,predicted)
            return r2score
        except Exception as e:
            raise customException(e,sys)
