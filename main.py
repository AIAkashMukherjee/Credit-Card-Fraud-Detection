from src.logger.custom_logging import logger
from src.pipeline.stage_01_train_pipe import DataIngestionPipe
from src.pipeline.stage_02_data_transform_pipe import DataTransformPipe
from src.pipeline.stage_03_model_train_pipe import ModelTrainPipe
# from src.pipeline.stage_04_model_eval import ModelEvalPipe
from src.exceptions.expection import CustomException
import sys


def run_stage(stage_name, pipeline_class):
    try:
        logger.info(f">>>>>> stage {stage_name} started <<<<<<")
        pipeline = pipeline_class()
        pipeline.main()
        logger.info(f">>>>>> stage {stage_name} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise CustomException(e,sys)
    
if __name__ == "__main__":
    stages = [
        ("Data Ingestion stage", DataIngestionPipe),
        ("Data Transformation stage", DataTransformPipe),
        ("Model Trainer stage", ModelTrainPipe),
        # ('Model Evaluation stage',ModelEvalPipe)

    ]

    for stage_name, pipeline_class in stages:
        run_stage(stage_name, pipeline_class)  