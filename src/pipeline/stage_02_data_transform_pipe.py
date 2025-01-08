import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.exceptions.expection import CustomException
from src.config.configuration import ConfigManager
from src.components.data_transformation import DataTransformation
from src.logger.custom_logging import logger

STAGE_NAME = "Prepare Base Model stage"


class DataTransformPipe:
    def __init__(self):
        pass

    def main(self):
        try:
            config=ConfigManager()
            data_transformation_config=config.get_data_transformation_config()
            data_transformation=DataTransformation(data_transformation_config)
            data_transformation.create_preprocessor()
            data_transformation.transform_data()
        except Exception as e:
            raise CustomException(e,sys) 

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataTransformPipe()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e 