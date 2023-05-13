import pandas as pd
import numpy as np
import os
import sys

from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder,OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.utils import resample


from exception import CustomException
from src.logger import logging

@dataclass
class DataTransformationConfig:
    Preprocessor_obj_file_path = os.path.join('artifcat',"Preprocessor.pkl")


class DataTransformation:


    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()


    def replace_data(self, column):
        column = column.str.replace('No internet service', 'No')
        column = column.str.replace('No phone service', 'No')
        return column
    
    def upsample_minority_class(self, data, minority_class_label, majority_class_label):
        minority = data[data['Churn'] == minority_class_label]
        majority = data[data['Churn'] == majority_class_label]

        minority_upsampled = resample(minority, replace=True, n_samples=majority.shape[0])
        data = pd.concat([minority_upsampled, majority], axis=0)
        return data
    
    def preprocess_data(self, data):
        data['TotalCharges'] = data['TotalCharges'].replace(" ", np.nan)
        data = data[data['TotalCharges'].notnull()]
        data = data.reset_index()[data.columns]
        data['TotalCharges'] = data['TotalCharges'].astype(float)
        data['MultipleLines'] = self.replace_data(data['MultipleLines'])
        data['OnlineSecurity'] = self.replace_data(data['OnlineSecurity'])
        data['OnlineBackup'] = self.replace_data(data['OnlineBackup'])
        data['DeviceProtection'] = self.replace_data(data['DeviceProtection'])
        data['TechSupport'] = self.replace_data(data['TechSupport'])
        data['StreamingTV'] = self.replace_data(data['StreamingTV'])
        data['StreamingMovies'] = self.replace_data(data['StreamingMovies'])

        return data
    
    def get_data_transformer_obj(self):

        try:
            numeric_columns = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
            categorical_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
       'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
       'PaperlessBilling', 'PaymentMethod']
            
            num_pipeline = Pipeline(
                steps=[
                    ("StandardScaler",StandardScaler())
                    ])
            

            cat_pipeline = Pipeline(
                steps=["Encoding",OrdinalEncoder()]
            )
            
            logging.info("Pipeline created and completed")


            preprocessor= ColumnTransformer([
                ("Numerical_pipeline",num_pipeline,numeric_columns),
                ("Categorical_pipeline",cat_pipeline,categorical_columns)

            ])

            return preprocessor
        
        
        except Exception as e:
            raise CustomException(e,sys)
        

        
    def get_data_transformer_label_object(self):

        try:
            cat_col = "Churn"
            label_encoder = LabelEncoder()

            return label_encoder
        
        except Exception as e:
            raise CustomException(e,sys)
        

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            # train_df = self.replace_data(train_df)
            # train_df = self.upsample_minority_class(train_df,'Yes', 'No')
            preprocessing_object=self.get_data_transformer_obj()
            preprocessing_label_object=self.get_data_transformer_label_object()

            logging.info(f"Applying preprocessing object on traing dataframe and testing dataframe.")


            target_column = "Churn"
            numerical_columns = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
            cat_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
             'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
           'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
            'PaperlessBilling', 'PaymentMethod']
            

            input_feature_train_df = train_df.drop(columns=[target_column],axis=1)
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns=[target_column],axis=1)
            target_feature_test_df = test_df[target_column]

            logging.info(f"Applying preprocessing object on traing dataframe and testing dataframe.")

            input_feature_train_arr=preprocessing_object.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_object.fit_transform(input_feature_test_df)

            input_target_feature_train_arr=preprocessing_label_object.fit_transform(target_feature_train_df)
            input_target_feature_test_arr=preprocessing_label_object.fit_transform(target_feature_test_df)


            train_arr = np.c_[
                input_feature_train_arr,np.array(input_target_feature_train_arr)
            ]

            test_arr = np.c_[
                input_feature_test_arr,np.array(input_target_feature_test_arr)
            ]

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path=self.Data_transformation_config.preprocessor_ob_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,test_arr,self.Data_transformation_config.preprocessor_ob_file_path,
            )
        
        except Exception as e:
            raise CustomException(e,sys)

        

           

           

        

        


            
            

        



        






 
 
 
 
    
          

