import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_obj
@dataclass 
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self, config):
        self.data_transformation_config = config
           
           
    def get_data_transformation_object(self):
        try:
            logging.info("Initiating data transformation")
            categorical_cols = ['cut', 'color', 'clarity']
            numerical_cols = ['carat', 'depth', 'table', 'x', 'y', 'z']
            
            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']
            ## Numerical Pipeline
            num_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())

                ]

            )

            # Categorigal Pipeline
            cat_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('ordinalencoder',OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                ('scaler',StandardScaler())
                ]

            )

            preprocessor=ColumnTransformer([
            ('num_pipeline',num_pipeline,numerical_cols),
            ('cat_pipeline',cat_pipeline,categorical_cols)
            ])
            
            logging.info("Data transformation completed")
            return preprocessor #getting the pickle file
        
        except Exception as e:
            logging.error(e)
            raise CustomException(e)
        
    def initiate_data_transformation(self,train_data_path, test_data_path):

        try:
            train_df=pd.read_csv(train_data_path)
            test_df=pd.read_csv(test_data_path)
            logging.info("read train and test data completed")
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            preprocessing_obj = self.get_data_transformation_object()

            target_column='price'
            drop_column=[target_column,'id']
         #dividing the data into input features and target variable
         #for train data
            input_feature_train_df=train_df.drop(columns=drop_column, axis=1)
            target_train_df=train_df[target_column]

            #for test data
            input_feature_test_df=test_df.drop(columns=drop_column, axis=1)
            target_test_df=train_df[target_column]

            #Data transformation
            input_feature_train_df_transformed_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_df_transformed_arr=preprocessing_obj.transform(input_feature_test_df)
       
            logging.info("Applying preprocessing object on train and test data completed")

            train_arr = np.concatenate((input_feature_train_df_transformed_arr, np.array(target_train_df.iloc[:input_feature_train_df_transformed_arr.shape[0]]).reshape(-1, 1)), axis=1)
            test_arr = np.concatenate((input_feature_test_df_transformed_arr, np.array(target_test_df.iloc[:input_feature_test_df_transformed_arr.shape[0]]).reshape(-1, 1)), axis=1)


            

            
            save_obj(
                file_path=self.data_transformation_config.preprocessor_obj_file_path, 
                obj=preprocessing_obj
            )
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            logging.error(e)
            raise CustomException(e, sys)