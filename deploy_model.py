import mlflow
import pickle
import pandas as pd
from src import *
import joblib
# Load local model

local_path = "model/st126438-a3-model.pkl"
predictor = joblib.load(local_path)
# Wrap model for MLflow
class CarPriceWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, predictor):
        self.predictor = predictor

    def predict(self, context, model_input: pd.DataFrame):
        return self.predictor.predict(model_input)

# Example input for MLflow schema
sample_df = pd.DataFrame([{
    'brand': 20,
    'km_driven': 772,
    'fuel': '0',
    'seller_type': '1',
    'mileage': 25.4,
    'engine': 1200.0,
    'max_power': 84.0,
    'year': 21,
    'seats': 2.0,
    'transmission': '1',
    'owner': 1,
}])

for col in ['fuel', 'seller_type', 'transmission']:
	if col in sample_df:
		sample_df[col] = sample_df[col].astype(int)
          

mlflow.set_tracking_uri("https://mlflow.ml.brain.cs.ait.ac.th/") 
mlflow.set_experiment(experiment_name="st126438-A3")

with mlflow.start_run(run_name="logistic_regression_deploy") as run:
    mlflow.pyfunc.log_model(
        name="model",
        python_model=CarPriceWrapper(predictor),
        input_example=sample_df
    )
    # Construct proper model URI to register
    model_uri = f"runs:/{run.info.run_id}/model"

# Register as a new version
registered_model = mlflow.register_model(
    model_uri=model_uri,
    name="st126438-A3-model"
)

print("Model deployed to MLflow!")