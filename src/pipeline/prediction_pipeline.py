import joblib 
from pathlib import Path


class PredicitonPipe:
    def __init__(self):
        self.model=joblib.load(Path('artifacts/model_trainer/model.pkl'))
        self.preprocessor=joblib.load(Path('artifacts/data_transformation/preprocessor.pkl'))


    def predict(self,X):
        X_processed = self.preprocessor.transform(X) 
        prediction=self.model.predict(X_processed)

        return prediction 