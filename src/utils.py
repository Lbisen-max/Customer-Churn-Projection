import os
import pandas as pd
import sys
import numpy as np
import dill

from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report, precision_score, recall_score
from sklearn.model_selection import train_test_split

from exception import CustomException
from src.logger import logging

def save_object(file_path,obj):
    try:
        dir_path =os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)


def evaluate_model(X_train,y_train,X_test,y_test,models,param): 

    try:
        # Make predictions and evaluate each model using confusion matrices
        results = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            print(f"Training model: {i}")
            model.fit(X_train, y_train)
            print(f"Evaluating model: {i}")
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            print(f"Confusion matrix:\n{cm}")
            report = classification_report(y_test, y_pred)
            print(f"Classification report:\n{report}")
            f1 = f1_score(y_test, y_pred, average='weighted')
            results[list(models.keys())[i]] = f1

        return results
        
    except Exception as e:
        raise CustomException(e,sys)

    
def load_object(file_path):
    try:
         with open(file_path,"rb") as file_obj:
              return dill.load(file_obj)
         
    except Exception as e:
        raise CustomException(e,sys)
                    

                  
                  
            
                 
                    
   
