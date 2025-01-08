from src.constants import *
from src.utils.utlis import *
from src.entity.config_entity import DataIngestionConfig,DataTransformationConfig,ModelTrainerConfig,EvaluationConfig


class ConfigManager:
    def __init__(self,config_file=CONFIG_FILE_PATH,
                 params_filepath = PARAMS_FILE_PATH):
        self.config=read_yaml(config_file)
        self.params=read_yaml(params_filepath)

    def get_data_ingestion_config(self)->DataIngestionConfig:
        config=self.config.data_ingestion
        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            local_data_file= config.local_data_file,
            train_file_path=config.train_file_path,
            test_file_path=config.test_file_path

        )    
        return data_ingestion_config    
    
    def get_data_transformation_config(self)->DataTransformationConfig:
        config=self.config.data_transformation
        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            preprocessor_obj=config.preprocessor_obj,
            train_file_path=config.train_file_path,
            test_file_path=config.test_file_path,
            save_train_path=config.save_train_path,
            save_test_path=config.save_test_path


        )    
        return data_transformation_config 
    

    def get_model_trainer_config(self)-> ModelTrainerConfig:
        config=self.config.model_trainer
        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            train_model_path=config.train_model_path,
            training_data_path=config.training_data_path,
            testing_data_path=config.testing_data_path
    

        )    
        return model_trainer_config
    
    def eval_config(self):
        config = self.config.model_evaluation
        create_directories([config.root_dir])
        eval_confg=EvaluationConfig(
            root_dir=config.root_dir,
            model_path="artifacts/model_trainer/model.pkl",
            test_data="artifacts/data_transformation/final_test.csv",
            mlflow_uri=os.getenv('MLFLOW_TRACKING_URI'),
           
        )    
        return eval_confg