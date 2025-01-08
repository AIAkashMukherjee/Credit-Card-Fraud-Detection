from src.entity.config_entity import DataIngestionConfig
import sys
import os
import pandas as pd
from src.exceptions.expection import CustomException
from src.logger.custom_logging import logger
from sklearn.model_selection import train_test_split

class DataIngestion:
    def __init__(self,config:DataIngestionConfig):
        self.config=config

    def initate_data_ingestion(self):
        logger.info('Data Ingestion Started')
        try:
            df=pd.read_csv('data/creditcard.csv')


            train_set,test_set=train_test_split(df,test_size=.2,random_state=42)

            train_set.to_csv(self.config.train_file_path,index=False,header=True)

            test_set.to_csv(self.config.test_file_path,index=False,header=True)

            logger.info('Data Ingestion finished')

            return (
                self.config.train_file_path,
                self.config.test_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)  