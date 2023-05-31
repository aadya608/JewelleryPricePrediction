import os
import sys
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

from src.logger import logging
from src.exception import CustomException

def save_obj(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)
        logging.info("Object saved successfully")

    except Exception as e:
        logging.error("Error while saving object: %s", e)
        raise CustomException("Error while saving object: %s", e)
    
    def evaluate_model(X_train,y_train,X_test,y_test,models):
        try:
            report={}
            for i in models:
                model=list(models.values())[i]
                model.fit(X_train,y_train)

                y_pred=model.predict(X_test)
                test_model_score = r2_score(y_test,y_test_pred)

                report[list(models.keys())[i]] =  test_model_score

            return report
        except Exception as e:
            logging.info("Error while evaluating model: %s", e)
            raise CustomException (e,sys)