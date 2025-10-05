from dash import Dash, html, dcc, Input, Output, State
import pandas as pd
import numpy as np
import mlflow.pyfunc
import os

# Load model from MLflow
mlflow.set_tracking_uri("http://mlflow.ml.brain.cs.ait.ac.th/")
os.environ["MLFLOW_TRACKING_USERNAME"] = "admin"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "password"


model = mlflow.pyfunc.load_model("models:/st126438-A3-model/latest")

# Feature names and default values (from sample_df)
feature_names = [
    'brand', 'km_driven', 'fuel', 'seller_type', 'mileage', 'engine',
    'max_power', 'year', 'seats', 'transmission', 'owner'
]

# Sample input defaults
sample = pd.DataFrame([{
    'brand': 'BMW',
    'km_driven': 772,
    'fuel': 'Diesel',
    'seller_type': 'Trustmark Dealer',
    'mileage': 240.4,
    'engine': 12.0,
    'max_power': 8.0,
    'year': 1999,
    'seats': 4.0,
    'transmission': 'Manual',
    'owner': 1,
}])

app = Dash(__name__)

app.layout = html.Div([
    html.H2("Car Classification (MLflow Model)"),
    html.Div([
        html.Div([
            html.Label(f"{feature}:"),
            dcc.Input(id=feature, value=default_values[feature], type='text')
        ]) for feature in feature_names
    ]),
    html.Br(),
    html.Button('Predict', id='predict-btn', n_clicks=0),
    html.Div(id='output')
])

@app.callback(
    Output('output', 'children'),
    Input('predict-btn', 'n_clicks'),
    [State(feature, 'value') for feature in feature_names]
)
def predict_car(n_clicks, *args):
    if n_clicks == 0:
        return ""

    try:
        # Create dict and DataFrame from input values
        input_data = {feature: val for feature, val in zip(feature_names, args)}

        # Try converting numeric fields
        for field in ['brand', 'km_driven', 'mileage', 'engine', 'max_power', 'year', 'seats', 'owner']:
            input_data[field] = float(input_data[field])

        df = pd.DataFrame([input_data])

        # Predict
        prediction = model.predict(df)[0]
        return f"✅ Predicted Class: {prediction}"
    except Exception as e:
        return f"⚠️ Prediction failed: {str(e)}"

if __name__ == '__main__':
    app.run_server(debug=True, host="0.0.0.0")
