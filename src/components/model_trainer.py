import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import os
import sys

from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report, precision_score, recall_score
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from dataclasses import dataclass



from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from logger import logging
from src.utils import save_object,evaluate_model
import os

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")


class ModelTrainer:

    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split traning and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = { "logistic regressionn": LogisticRegression(),
                      "KNN classifier" : KNeighborsClassifier(),
                      "Decesion tree classifier" : DecisionTreeClassifier(),
                      "Random classifier" : RandomForestClassifier(),
                      "Ada boost classifier" : AdaBoostClassifier(),
                      "gradient boosting classifier" : GradientBoostingClassifier(),
                      "XGBClassifier" : XGBClassifier()}
            
            params={
                "logistic regressionn":{"penalty":['l1', 'l2'],
                 "C" : [0.001, 0.01, 0.1, 1, 10, 100],
                 "solver" :  ['liblinear', 'saga']

                 },
                 "KNeighborsClassifier":{
                     "n_neighbors":[3, 5, 7, 9, 11],
                     "weights" : ['uniform', 'distance'],
                     "p" :  [1, 2]
                 },
                 "DecisionTreeClassifier" : {
                     "criterion" : ['gini', 'entropy'],
                     "max_depth" : [None, 5, 10, 15, 20],
                     "min_samples_split" :  [2, 5, 10],
                     "min_samples_leaf" : [1, 2, 4]
                 },
                 "RandomForestClassifier" : {
                     "n_estimators" : [100, 200, 300],
                     "criterion" : ['gini', 'entropy'],
                     "max_depth" : [None, 5, 10, 15, 20],
                     "min_samples_split": [2, 5, 10],
                     "min_samples_leaf" : [1, 2, 4]
                 },
                 "AdaBoostClassifier" : {
                     "n_estimators" :  [50, 100, 200, 300],
                     "learning_rate" : [0.01, 0.1, 1],
                     "algorithm" : ['SAMME', 'SAMME.R']
                 },
                 "GradientBoostingClassifier" : {
                     "loss" : ['deviance', 'exponential'],
                     "learning_rate" : [0.01, 0.1, 1],
                     "n_estimators" : [50, 100, 200, 300],
                     "max_depth" : [3, 5, 7],
                     "min_samples_split" : [2, 5, 10],
                     "min_samples_leaf" : [1, 2, 4]
                 }

            }
            

            model_report : dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=params)
            
            logging.info("Model trained and evaluated")

            best_model_score = max(sorted(model_report.values()))
            
            # to get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            if best_model_score<85:
                print("no best model")
                
            logging.info(f"Found model on traning dataset")

            

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model

            )
    
        except Exception as e:
            raise CustomException(e,sys)

    



