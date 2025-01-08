import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.config.configuration import ConfigManager
from src.components.model_evalutaion import ModelEvaluation
from src.logger.custom_logging import logger
from src.exceptions.expection import CustomException

STAGE_NAME = "Model Evaluation stage"


class ModelEvalPipe:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigManager()
            eval_config = config.eval_config()
            evaluation = ModelEvaluation(eval_config)
            evaluation.evaluation()

        except Exception as e:
            raise e

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelEvalPipe()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e 