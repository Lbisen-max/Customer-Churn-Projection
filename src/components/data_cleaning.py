import sys
from dataclasses import dataclass
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object
from sklearn.model_selection import train_test_split
from src.components.data_transformation import DataTransformation
from sklearn.utils import resample

from exception import CustomException
from src.logger import logging



@dataclass
class DataCleaningConfig:
    train_data_path_clean:str=os.path.join("artifacts","train_cleaned.csv")
    test_data_path_clean:str=os.path.join("artifacts","test_cleaned.csv")

class DataCleaning:
    def __init__(self):
        self.cleaning_config=DataCleaningConfig()

    def initiate_data_cleaning(self):

        logging.info("Entered the data cleaning method or component")

        try:
            df1=pd.read_csv("artifacts/train.csv")
            df2=pd.read_csv("artifacts/test.csv")

            logging.info('Read the dataset as dataframe')

            df1['TotalCharges']=df1['TotalCharges'].replace(" ",np.nan)
            df1=df1[df1['TotalCharges'].notnull()]
            df1 = df1.reset_index()[df1.columns]
            df1['TotalCharges'] = df1['TotalCharges'].astype(float)

            df1['MultipleLines'] = df1['MultipleLines'].str.replace('No phone service','No')
            df1['OnlineSecurity'] = df1['OnlineSecurity'].str.replace('No internet service', 'No')
            df1['OnlineBackup'] = df1['OnlineBackup'].str.replace('No internet service', 'No')
            df1['DeviceProtection'] = df1['DeviceProtection'].str.replace('No internet service', 'No')
            df1['TechSupport'] = df1['TechSupport'].str.replace('No internet service', 'No')
            df1['StreamingTV'] = df1['StreamingTV'].str.replace('No internet service', 'No')
            df1['StreamingMovies'] = df1['StreamingMovies'].str.replace('No internet service', 'No')
            df1=df1.drop(['customerID'],axis=1)
        

            

            df2['TotalCharges']=df2['TotalCharges'].replace(" ",np.nan)
            df2=df2[df2['TotalCharges'].notnull()]
            df2 = df2.reset_index()[df2.columns]
            df2['TotalCharges'] = df2['TotalCharges'].astype(float)

            df2['MultipleLines'] = df2['MultipleLines'].str.replace('No phone service','No')
            df2['OnlineSecurity'] = df2['OnlineSecurity'].str.replace('No internet service', 'No')
            df2['OnlineBackup'] = df2['OnlineBackup'].str.replace('No internet service', 'No')
            df2['DeviceProtection'] = df2['DeviceProtection'].str.replace('No internet service', 'No')
            df2['TechSupport'] = df2['TechSupport'].str.replace('No internet service', 'No')
            df2['StreamingTV'] = df2['StreamingTV'].str.replace('No internet service', 'No')
            df2['StreamingMovies'] = df2['StreamingMovies'].str.replace('No internet service', 'No')
            df2=df2.drop(['customerID'],axis=1)

            # Upsampling the data

            minority = df1[df1.Churn=="Yes"]
            majority = df1[df1.Churn=="No"]
            minority_upsample = resample(minority, replace=True, n_samples=majority.shape[0])
            df1 = pd.concat([minority_upsample, majority], axis=0)

            logging.info('train and test data is cleaned and train data is resampled') 

            os.makedirs(os.path.dirname(self.cleaning_config.train_data_path_clean),exist_ok=True)
            logging.info('directory created')

            df1.to_csv(self.cleaning_config.train_data_path_clean,index=False,header=True)
            logging.info('df1 saved')
            df2.to_csv(self.cleaning_config.test_data_path_clean,index=False,header=True)
            logging.info('df2 saved')

            logging.info("Train test data cleaned")
            return df1,df2
            logging.info("returned df1 and df2")
    
        except Exception as e:
            raise CustomException(e,sys)