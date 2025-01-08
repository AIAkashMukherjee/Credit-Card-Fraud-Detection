from src.entity.config_entity import ModelTrainerConfig
from src.logger.custom_logging import logger
import sys,joblib
import pandas as pd
from sklearn.base import BaseEstimator
import keras
from keras import backend as K
from keras.models import Sequential 
from keras.layers import Activation
from keras.layers import Dense
from keras import Input
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from sklearn.metrics import roc_auc_score
from src.utils.utlis import *
from src.exceptions.expection import CustomException


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    @staticmethod
    def save_model(path: Path, model: BaseEstimator):
        """Save the trained model to the specified path."""
        try:
            joblib.dump(model, path)
            print(f"Model saved at {path}")
        except Exception as e:
            print(f"Error saving model: {e}")
            logger.error(f"Error saving model: {e}")
            raise CustomException(e, sys)      


    
    def train(self):
        train_data=pd.read_csv(self.config.training_data_path)
        test_data=pd.read_csv(self.config.testing_data_path)
        try:
            X_train = train_data.iloc[:, :-1].values
            y_train = train_data.iloc[:, -1].values
            X_test = test_data.iloc[:, :-1].values
            y_test = test_data.iloc[:, -1].values   

            print("X_train shape:", X_train.shape)
            print("y_train shape:", y_train.shape)

            n_inputs=X_train.shape[1]
            model=Sequential([  
            Input(shape=(n_inputs,)),
            Dense(64,activation='relu'),
            Dense(2,activation='softmax')
            ])

            model.compile(Adam(learning_rate=0.0001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
            model.fit(X_train, y_train, validation_split=0.2, batch_size=25, epochs=20, shuffle=True)
            
            
            y_pred = model.predict(X_test)[:, 1]
            
            test_model_score = roc_auc_score(y_test, y_pred)

            print(f"Test model score: {test_model_score*100}")

            # Save the best model
            self.save_model(path=self.config.train_model_path,model=model)
        except Exception as e:
            logger.error(f'Error occurred: {e}')
            raise CustomException(e, sys)