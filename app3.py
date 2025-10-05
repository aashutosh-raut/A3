import dash
from dash import html, dcc, Input, Output, State
import pandas as pd
import mlflow.pyfunc
import mlflow
import os

# ─── MLflow setup ───────────────────────────────────────────
os.environ["MLFLOW_TRACKING_USERNAME"] = "admin"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "password"

mlflow_tracking_url = os.environ.get(
    "MLFLOW_TRACKING_URL",
    "https://mlflow.ml.brain.cs.ait.ac.th/"
)
mlflow.set_tracking_uri(mlflow_tracking_url)
mlflow.set_experiment("st126438-a3")

model_name = "st126438-A3-model"

try:
    model = mlflow.pyfunc.load_model(f"models:/{model_name}/latest")
except Exception as e:
    print(" Failed to load model:", e)
    model = None

# ─── Default sample values ──────────────────────────────────
sample = {
    'brand': 'BMW', 'km_driven': 772, 'fuel': 'Diesel',
    'seller_type': 'Trustmark Dealer', 'mileage': 240.4,
    'engine': 12.0, 'max_power': 8.0, 'year': 1999,
    'seats': 4.0, 'transmission': 'Manual', 'owner': 1,
}

# ─── Initialize Dash ────────────────────────────────────────
app = dash.Dash(__name__, title="Car Price Predictor")
server = app.server   # ← Gunicorn entry point

# ─── Layout: clean minimalist design ────────────────────────
app.layout = html.Div(
    style={
        "fontFamily": "Arial",
        "maxWidth": "600px",
        "margin": "auto",
        "padding": "30px",
        "backgroundColor": "#f9f9f9",
        "borderRadius": "12px",
        "boxShadow": "0 0 10px rgba(0,0,0,0.1)"
    },
    children=[
        html.H2(" Car Price Class Prediction", style={"textAlign": "center"}),

        html.Label("Car Brand Name:", style={"fontWeight": "bold"}),
        dcc.Dropdown(
            id="brand",
            options=[{"label": name, "value": i} for i, name in enumerate([
                "Ambassador","Ashok","Audi","BMW","Chevrolet","Daewoo","Datsun","Fiat","Force",
                "Ford","Honda","Hyundai","Isuzu","Jaguar","Jeep","Kia","Land","Lexus","MG",
                "Mahindra","Maruti","Mercedes-Benz","Mitsubishi","Nissan","Opel","Peugeot",
                "Renault","Skoda","Tata","Toyota","Volkswagen","Volvo"
            ])],
            value=sample['brand'],
            style={"width": "300px", "margin-bottom": "10px"}
        ),

        html.Label("Km Driven:",  style={"fontWeight": "bold"}), 
        dcc.Input(id='km_driven', type='number', value=sample['km_driven'], style={"width": "200px", "margin-bottom": "10px"}),
        html.Br(),

        html.Label("Fuel Type:", style={"fontWeight": "bold"}),
        dcc.Dropdown(
            id="fuel_type",
            options=[
                {"label": "Petrol", "value": 0},
                {"label": "Diesel", "value": 1},
            ],
            value=sample['fuel'],
            style={"width": "200px", "margin-bottom": "10px"}
        ),

        html.Label("Seller Type:", style={"fontWeight": "bold"}),
        dcc.Dropdown(
            id="seller_type",
            options=[
                {"label": "Dealer", "value": 0},
                {"label": "Individual", "value": 1},
                {"label": "Trustmark Dealer", "value": 2},
            ],
            value=sample['seller_type'],
            style={"width": "200px", "margin-bottom": "10px"}
        ),


        html.Label("Transmission Type:", style={"fontWeight": "bold"}),
        dcc.Dropdown(
            id="transmission_type",
            options=[
                {"label": "Automatic", "value": 0},
                {"label": "Manual", "value": 1},
            ],
            value=sample['transmission'],
            style={"width": "200px", "margin-bottom": "10px"}
        ),

        html.Label("Year:",  style={"fontWeight": "bold"}), dcc.Input(id='year', type='number', value=sample['year']),
        html.Br(),

        html.Label("Mileage (km/l):",  style={"fontWeight": "bold"}), dcc.Input(id='mileage', type='number', value=sample['mileage']),
        html.Br(),

        html.Label("Engine (cc):",  style={"fontWeight": "bold"}), dcc.Input(id='engine', type='number', value=sample['engine']),
        html.Br(),

        html.Label("Owner Type (1=First, 2=Second, 3=Third, 4=Fourth, 5=More):", style={"fontWeight": "bold"}),
        dcc.Input(id='owner_type', type='number', value=sample['owner'], style={"margin-bottom": "10px"}),
        html.Br(),

        html.Label("Seats:", style={"fontWeight": "bold"}), dcc.Input(id='seats', type='number', value=sample['seats']),
        html.Br(),

        html.Label("Max Power (HP):", style={"fontWeight": "bold"}), dcc.Input(id='max_power', type='number', value=sample['max_power']),
        html.Br(),

        html.Button("Predict", id='predict-btn', n_clicks=0,
                    style={"width": "100%", "marginTop": "15px", "background": "#0074D9",
                           "color": "white", "fontWeight": "bold", "border": "none", "padding": "10px"}),

        html.H3("Prediction:", style={"marginTop": "20px"}),
        dcc.Loading(children=html.Div(id='prediction-output'), type="circle"),
    ]
)

# ─── Callback ───────────────────────────────────────────────
@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-btn', 'n_clicks'),
    State('brand', 'value'),
    State('km_driven', 'value'),
    State('fuel_type', 'value'),
    State('seller_type', 'value'),
    State('transmission_type', 'value'),
    State('year', 'value'),
    State('mileage', 'value'),
    State('engine', 'value'),
    State('owner_type', 'value'),
    State('seats', 'value'),
    State('max_power', 'value')
)
def predict(n_clicks, brand, km_driven, fuel_type, seller_type, transmission_type, year, mileage, engine, owner_type, seats, max_power):
    if n_clicks == 0:
        return "No prediction yet."
    if model is None:
        return "Model failed to load."

    input_df = pd.DataFrame([{
        'brand': brand,
        'km_driven': km_driven,
        'fuel_type': fuel_type,
        'seller_type': seller_type,
        'transmission_type': transmission_type,
        'year': year,
        'mileage': mileage,
        'engine': engine,
        'owner_type': owner_type,
        'seats': seats,
        'max_power': max_power
    }])

    try:
        prediction = model.predict(input_df)
        return f"Predicted Price Class: {prediction[0]}"
    except Exception as e:
        return f"Prediction failed: {e}"

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)




# import dash
# from dash import html, dcc, Input, Output, State
# import pandas as pd
# import mlflow.pyfunc
# import os

# # MLflow credentials
# os.environ["MLFLOW_TRACKING_USERNAME"] = "admin"
# os.environ["MLFLOW_TRACKING_PASSWORD"] = "password"

# mlflow_tracking_url = os.environ.get(
#     "MLFLOW_TRACKING_URL",
#     "https://mlflow.ml.brain.cs.ait.ac.th/"
# )
# import mlflow
# mlflow.set_tracking_uri(mlflow_tracking_url)
# mlflow.set_experiment("st126438-a3")

# model_name = "st126438-A3-model"

# # Load the MLflow model once
# try:
#     model = mlflow.pyfunc.load_model(f"models:/{model_name}/latest")
# except Exception as e:
#     print("Failed to load model:", e)
#     model = None

# # Sample input defaults
# sample = pd.DataFrame([{
#     'brand': 'BMW',
#     'km_driven': 772,
#     'fuel': 'Diesel',
#     'seller_type': 'Trustmark Dealer',
#     'mileage': 240.4,
#     'engine': 12.0,
#     'max_power': 8.0,
#     'year': 1999,
#     'seats': 4.0,
#     'transmission': 'Manual',
#     'owner': 1,
# }])

# # Initialize Dash app
# app = dash.Dash(__name__)

# app.layout = html.Div([
#     html.H1("Car Price Class Prediction"),

#     html.Label("Car Brand Name:", style={"fontWeight": "bold"}),
#     dcc.Dropdown(
#         id="brand",
#         options=[{"label": name, "value": i} for i, name in enumerate([
#             "Ambassador","Ashok","Audi","BMW","Chevrolet","Daewoo","Datsun","Fiat","Force",
#             "Ford","Honda","Hyundai","Isuzu","Jaguar","Jeep","Kia","Land","Lexus","MG",
#             "Mahindra","Maruti","Mercedes-Benz","Mitsubishi","Nissan","Opel","Peugeot",
#             "Renault","Skoda","Tata","Toyota","Volkswagen","Volvo"
#         ])],
#         value=sample['brand'],
#         style={"width": "300px", "margin-bottom": "10px"}
#     ),

#     html.Label("Km Driven:", style={"fontWeight": "bold"}),
#     dcc.Input(id='km_driven', type='number', value=sample['km_driven'].iloc[0], style={"margin-bottom": "10px"}),

#     html.Label("Fuel Type:", style={"fontWeight": "bold"}),
#     dcc.Dropdown(
#         id="fuel_type",
#         options=[
#             {"label": "Petrol", "value": 0},
#             {"label": "Diesel", "value": 1},
#             {"label": "CNG", "value": 2}
#         ],
#         value=sample['fuel'],
#         style={"width": "200px", "margin-bottom": "10px"}
#     ),

#     html.Label("Seller Type:", style={"fontWeight": "bold"}),
#     dcc.Dropdown(
#         id="seller_type",
#         options=[
#             {"label": "Dealer", "value": 0},
#             {"label": "Individual", "value": 1},
#             {"label": "Trustmark Dealer", "value": 2},
#         ],
#         value=sample['seller_type'],
#         style={"width": "200px", "margin-bottom": "10px"}
#     ),

#     html.Label("Transmission Type:", style={"fontWeight": "bold"}),
#     dcc.Dropdown(
#         id="transmission_type",
#         options=[
#             {"label": "Automatic", "value": 0},
#             {"label": "Manual", "value": 1},
#         ],
#         value=sample['transmission'],
#         style={"width": "200px", "margin-bottom": "10px"}
#     ),

#     html.Label("Year:", style={"fontWeight": "bold"}),
#     dcc.Input(id='year', type='number', value=sample['year'].iloc[0], style={"margin-bottom": "10px"}),

#     html.Label("Mileage (km/l):", style={"fontWeight": "bold"}),
#     dcc.Input(id='mileage', type='number', value=sample['mileage'].iloc[0], style={"margin-bottom": "10px"}),

#     html.Label("Engine (cc):", style={"fontWeight": "bold"}),
#     dcc.Input(id='engine', type='number', value=sample['engine'].iloc[0], style={"margin-bottom": "10px"}),

#     html.Label("Owner Type (1=First, 2=Second, 3=Third, 4=Fourth, 5=More):", style={"fontWeight": "bold"}),
#     dcc.Input(id='owner_type', type='number', value=sample['owner'].iloc[0], style={"margin-bottom": "10px"}),

#     html.Label("Seats:", style={"fontWeight": "bold"}),
#     dcc.Input(id='seats', type='number', value=sample['seats'].iloc[0], style={"margin-bottom": "10px"}),

#     html.Label("Max Power (HP):", style={"fontWeight": "bold"}),
#     dcc.Input(id='max_power', type='number', value=sample['max_power'].iloc[0], style={"margin-bottom": "10px"}),

#     html.Button("Predict", id='predict-btn', n_clicks=0, style={"margin-top": "10px"}),

#     html.H2("Prediction:"),
#     dcc.Loading(
#         id="loading-prediction",
#         type="circle",
#         children=html.Div(id='prediction-output')
#     ),
# ])

# @app.callback(
#     Output('prediction-output', 'children'),
#     Input('predict-btn', 'n_clicks'),
#     State('brand', 'value'),
#     State('km_driven', 'value'),
#     State('fuel_type', 'value'),
#     State('seller_type', 'value'),
#     State('transmission_type', 'value'),
#     State('year', 'value'),
#     State('mileage', 'value'),
#     State('engine', 'value'),
#     State('owner_type', 'value'),
#     State('seats', 'value'),
#     State('max_power', 'value'),
# )
# def predict(n_clicks, brand, km_driven, fuel_type, seller_type, transmission_type, year, mileage, engine, owner_type, seats, max_power):
#     if n_clicks == 0:
#         return "No prediction yet."
#     if model is None:
#         return "Model failed to load."

#     input_df = pd.DataFrame([{
#         'brand': brand,
#         'km_driven': km_driven,
#         'fuel_type': fuel_type,
#         'seller_type': seller_type,
#         'transmission_type': transmission_type,
#         'year': year,
#         'mileage': mileage,
#         'engine': engine,
#         'owner_type': owner_type,
#         'seats': seats,
#         'max_power': max_power
#     }])

#     try:
#         prediction = model.predict(input_df)
#         return f"Predicted Price Class: {prediction[0]}"
#     except Exception as e:
#         return f"Prediction failed: {e}"

# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=8050)

