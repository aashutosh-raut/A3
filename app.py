from dash import Dash, html, dcc, Input, Output, State, dash_table #importing the necessary libraries
import pandas as pd
import joblib
import numpy as np

app = Dash() #initilising the app

df = pd.read_csv("Cars.csv") # importing the dataset

# filename = './model/st126438-a3-model.pkl' # importing the car_prediction model from the car prediction
# loaded_model = joblib.load(filename) #loading the model

app.layout = html.Div(children=[ # initializing the app layout 
    html.Div([
    html.H1(children='Car Price Prediction'), # Header
    ],style={"display":"flex","justify-content":"center"}),
    html.Div([
        html.Button("Car prices", id="carprices",  # Car price Button 
                    style={"background-color": "green", "color": "white", "padding": "10px",
                        "border-radius": "10px", "margin-right": '10px', "cursor": "pointer","height": "40px"}),

        html.Button("Instructions", id="instruction", # Instructions button
                    style={"background-color": "green", "color": "white", "padding": "8px",
                        "border-radius": "10px", "cursor": "pointer","height": "42px"}),
    ],style={"display": "flex","justify-content":"space-around","margin-top": "12px"}),

    html.Div(id="table-container"),
    dcc.Store(id="table-visible", data=False),  
    
    html.Div(id="instruction-container"),
    dcc.Store(id="instruction-visible", data=False),  
    
    html.Div([
    html.Div([
    html.Div([
    html.Label("Car Brand Name:"), # car brand names. it was turned into numerical value now tuenred back for the user
    dcc.Dropdown(
        id="brand-input",
        options=[
            {"label": "Ambassador", "value": 0},
            {"label": "Ashok", "value": 1},
            {"label": "Audi", "value": 2},
            {"label": "BMW", "value": 3},
            {"label": "Chevrolet", "value": 4},
            {"label": "Daewoo", "value": 5},
            {"label": "Datsun", "value": 6},
            {"label": "Fiat", "value": 7},
            {"label": "Force", "value": 8},
            {"label": "Ford", "value": 9},
            {"label": "Honda", "value": 10},
            {"label": "Hyundai", "value": 11},
            {"label": "Isuzu", "value": 12},
            {"label": "Jaguar", "value": 13},
            {"label": "Jeep", "value": 14},
            {"label": "Kia", "value": 15},
            {"label": "Land", "value": 16},
            {"label": "Lexus", "value": 17},
            {"label": "MG", "value": 18},
            {"label": "Mahindra", "value": 19},
            {"label": "Maruti", "value": 20},
            {"label": "Mercedes-Benz", "value": 21},
            {"label": "Mitsubishi", "value": 22},
            {"label": "Nissan", "value": 23},
            {"label": "Opel", "value": 24},
            {"label": "Peugeot", "value": 25},
            {"label": "Renault", "value": 26},
            {"label": "Skoda", "value": 27},
            {"label": "Tata", "value": 28},
            {"label": "Toyota", "value": 29},
            {"label": "Volkswagen", "value": 30},
            {"label": "Volvo", "value": 31}
        ],
        placeholder="Select Brand",
        style={"margin-bottom": "10px","width": "22rem"}
        ),
    ]),
    
    html.Div([ #kilometers driven 
        html.Label("Kms Driven:"),
        dcc.Input(id="km", type="number", placeholder="Enter Km driven by Car",
                  style={"margin-bottom": "10px", "display": "block","width": "22rem","padding": "6px"}),
    ]),

    html.Div([# fuel type
        html.Label("Fuel Type:"), 
        dcc.Dropdown(
            id="fuel-input",
            options=[
            {"label": "Diesel", "value": 0}, 
            {"label": "Petrol", "value": 1}
            ],
            placeholder="Select fuel",
            style={"margin-bottom": "10px","width": "22rem"}
        ),
    ]),

    html.Div([ # seller type
        html.Label("Seller Type:"),
        dcc.Dropdown(
            id="seller-type-input",   #select seller type with the dropdown
            options=[
            {"label": "Dealer", "value": 0}, # dealer = 0 initial string data to numerical data equivalence
            {"label": "Individual", "value": 1},# individual = 1
            {"label": "Trustmark Dealer", "value": 2} # trustmark dealer = 2 
            ],
            placeholder="Select Seller Type",
            style={"margin-bottom": "10px","width": "22rem"}
        ),
    ]),

    html.Div([ # seller type
        html.Label("Transmission:"),
        dcc.Dropdown(
            id="transmission-input",   #select seller type with the dropdown
            options=[
            {"label": "Manual", "value": 0}, # manual
            {"label": "Automatic", "value": 1},# automatic
            ],
            placeholder="Select Transmission Type",
            style={"margin-bottom": "10px","width": "22rem"}
        ),
    ]),

    html.Label("Mileage (kmpl):"), # mileage 
    dcc.Input(id="mileage", type="number", placeholder="Enter mileage",
                style={"margin-bottom": "10px", "display": "block","width": "22rem","padding": "6px"}),

    html.Label("Engine (CC):"), # engine
    dcc.Input(id="engine", type="number", placeholder="Enter engine capacity",
                style={"margin-bottom": "10px", "display": "block","width": "22rem","padding": "6px"}),

    html.Label("Seats:"), # seats
    dcc.Input(id="seats", type="number", placeholder="Enter number of seats",
                style={"margin-bottom": "10px", "display": "block","width": "22rem","padding": "6px"}),

    html.Label("Max Power (bhp):"), #max power
    dcc.Input(id="max_power", type="number", placeholder="Enter max power",
                style={"margin-bottom": "10px", "display": "block","width": "22rem","padding": "6px"}),
    ],style={"margin-left": "0.2rem"}),
    ],style={"display": "flex","justify-content": "center"}),

    html.Div([
    html.Button("Predict", id="prediction", #prediction 
                style={"background-color": "green", "color": "white", "padding": "10px", # styling 
                       "border-radius": "10px", "cursor": "pointer"}), 
    ],style={"display":"flex","justify-content":"center"}),
    html.Div(id="prediction-container"),
    dcc.Store(id="prediction-visible", data=False)
], style={"padding": "1rem"})


@app.callback( #calling back the id
    Output("table-container", "children"),
    Output("table-visible", "data"),
    Input("carprices", "n_clicks"),
    State("table-visible", "data"),  
    prevent_initial_call=True
)

def show_carprices(n_clicks, visible): #Display the car prices
    if not visible:
        return dash_table.DataTable(
            columns=[{"name": "Brand", "id": "name"},
                     {"name": "Max Power", "id": "max_power"},
                     {"name": "Kms Driven", "id": "km_driven"},
                     {"name": "Engine", "id": "engine"},
                     {"name": "Mileage", "id": "mileage"},
                     {"name": "Selling Price", "id": "selling_price"}],
            data=df[["name","max_power","km_driven","engine","mileage", "selling_price"]].to_dict('records'),
            page_size=10,
            style_table={"width": "98%"}
        ), True
    else:
        return "", False


@app.callback(
    Output("instruction-container", "children"),
    Output("instruction-visible", "data"),
    Input("instruction", "n_clicks"),
    State("instruction-visible", "data"),
    prevent_initial_call=True
)
def instructiontoggle(n_clicks, visible):  # instructions for running the app. 
    if not visible:
        return html.Div("You can check the car prices by clicking the Car Price's button. " \
        "To predict the price of your car, enter brand name, km driven, fuel type, seller type, milage, engine, number of seats and max power" \
        "Click predict.",
                        style={"margin-left": "10px","margin-right":"10px", "padding-left":"10px","padding-right":"10px"}), True
    else:
        return "", False

@app.callback(
    Output("prediction-container", "children"),
    Output("prediction-visible", "data"),
    Input("prediction", "n_clicks"),
    State("brand-input", "value"),     
    State("km", "value"),
    State("fuel-input", "value"),
    State("seller-type-input", "value"),
    State("mileage", "value"),
    State("engine", "value"),
    State("seats", "value"),
    State("max_power", "value"),
    State("prediction-visible", "data"),
    prevent_initial_call=True
)
def prediction(n_clicks, brand, km, fuel, seller, mileage, engine, seats, max_power, visible): # for car price prediction
    # Dummy prediction logic since model is commented out
    if not visible:
        if None in [brand, km, fuel, seller, mileage, engine, seats, max_power]:
            return "Please fill all fields to get a prediction.", True
        # Replace this with your actual model prediction logic
        predicted_price = (float(km) * 0.01 + float(mileage) * 100 + float(engine) * 0.5 + float(max_power) * 10) / (float(seats) + 1)
        return f"Predicted Car Price: {predicted_price:,.0f}", True
    else:
        return "", False

if __name__ == '__main__': # main app running on the port
    app.run(debug=True, host="0.0.0.0")