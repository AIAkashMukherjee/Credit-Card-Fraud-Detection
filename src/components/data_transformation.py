from src.entity.config_entity import DataTransformationConfig
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer
from src.exceptions.expection import CustomException
from src.logger.custom_logging import logger
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
import sys
from src.utils.utlis import save_obj


class DataTransformation:
    def __init__(self,config:DataTransformationConfig):
        self.config=config

    def create_preprocessor(self):
        try:
            logger.info('Creating data transformation pipeline')

            col_to_transform=['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']

            
            preprocessor = ColumnTransformer(
            transformers=[
            ('num', RobustScaler(), col_to_transform)
            ])

            return preprocessor

        except Exception as e:
            logger.error(f"Error in creating data transformation pipeline: {str(e)}")
            raise CustomException(e, sys)
        
    def transform_data(self):
        train_path=self.config.train_file_path
        test_path=self.config.test_file_path
        
        try:
            train_data=pd.read_csv(train_path)
            test_data=pd.read_csv(test_path)

            target_column = 'Class'
            drop_columns = [target_column]

            preprocessor=self.create_preprocessor()
            input_feature_train_data = train_data.drop(columns=drop_columns)
            target_feature_train_data = train_data[target_column]
            input_feature_test_data = test_data.drop(columns=drop_columns)
            target_feature_test_data = test_data[target_column]

            input_train_arr=preprocessor.fit_transform(input_feature_train_data)
            input_test_arr=preprocessor.transform(input_feature_test_data)

            # Apply SMOTE for class balancing
            smote = SMOTE(random_state=42)
            input_train_resampled, target_train_resampled = smote.fit_resample(input_train_arr, target_feature_train_data)

            train_array = np.c_[input_train_resampled, target_train_resampled.values.reshape(-1, 1)]
            test_array = np.c_[input_test_arr, target_feature_test_data.values.reshape(-1, 1)]


            save_obj(file_path=self.config.preprocessor_obj,obj=preprocessor)

            train_df = pd.DataFrame(train_array)
            test_df = pd.DataFrame(test_array)

            train_df.to_csv(self.config.save_train_path, index=False,header=True)
            test_df.to_csv(self.config.save_test_path, index=False,header=True)


        except Exception as e:  
            raise CustomException(e, sys)        