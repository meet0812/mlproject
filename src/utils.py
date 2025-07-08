import os 
import sys 

import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import customException

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
    
    except Exception as e:
        raise customException(e,sys)

def evalute_models(X_train, y_train, X_test, y_test, models,param):
    try:
        report = {}
        fitted_models = {}

        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = models[model_name]
            para = param[model_name]

            gs = GridSearchCV(model, para, cv=3, n_jobs=6, verbose=2)
            gs.fit(X_train, y_train)

            best_model = gs.best_estimator_
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            
            report[model_name] = test_model_score
            fitted_models[model_name] = best_model 

        return report,fitted_models
    except Exception as e:
        raise customException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return dill.load(file_obj)
    
    except Exception as e:
        raise customException(e,sys)
