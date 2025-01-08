import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.exceptions.expection import CustomException
from src.config.configuration import ConfigManager
from src.components.model_trainer import ModelTrainer
from src.logger.custom_logging import logger

STAGE_NAME = "Model Trainer stage"


class ModelTrainPipe:
    def __init__(self):
        pass

    def main(self):
        try:
            config=ConfigManager()
            model_trainer_config=config.get_model_trainer_config()
            model_trainer=ModelTrainer(model_trainer_config)
            model_trainer.train()

        except Exception as e:
            raise CustomException(e,sys)

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainPipe()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e 