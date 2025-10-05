# Car Price Prediction Model

A multi-class classification model that predicts car selling prices using Logistic Regression implemented from scratch. The model classifies cars into 4 price quartiles based on vehicle features.

## 🚀 Features

- **Custom Logistic Regression** implementation with:
  - Xavier weight initialization
  - Softmax activation for multi-class classification
  - L2 regularization (λ=0.1)
  - Momentum-based gradient descent (β=0.9)
- **MLflow Integration** for experiment tracking and model registry
- **Production-ready** prediction pipeline with preprocessing

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | ~85-90% |
| Macro F1 | ~0.85 |
| Macro Precision | ~0.85 |
| Macro Recall | ~0.85 |

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/aashutosh-raut/Carprediction.git
cd Carprediction

# Install dependencies
pip install -r requirements.txt
```

### Requirements
```
numpy
pandas
scikit-learn
matplotlib
seaborn
mlflow
joblib
```

## 📁 Project Structure

```
Carprediction/
├── src.py                      # Core model classes
├── test.py                     # Model validation tests
├── Cars.csv                    # Dataset
├── model/
│   └── st126438-a3-model.pkl  # Trained model
└── README.md
```

## 🎯 Usage

### Training the Model

```python
from src import LogisticRegression, CarPricePredictor
import pandas as pd
import numpy as np

# Load and preprocess data
df = pd.read_csv('Cars.csv')
# ... (preprocessing steps)

# Initialize and train model
model = LogisticRegression(
    k=4,                    # 4 price classes
    n=X_train.shape[1],     # number of features
    lr=0.01,                # learning rate
    max_iter=1000,          # iterations
    l2_penalty=True,        # L2 regularization
    lambda_=0.1             # regularization strength
)

model.fit(X_train_std, y_train_encoded)
```

### Making Predictions

```python
import joblib
import pandas as pd

# Load the trained model
loaded_model = joblib.load("model/st126438-a3-model.pkl")

# Prepare input data
sample_df = pd.DataFrame([{
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

# Get prediction (0-3, representing price quartiles)
prediction = loaded_model.predict(sample_df)
print(f"Predicted price class: {prediction[0]}")
```

### Running Tests

```bash
python test.py
```

Expected output:
```
Running Model Tests...

✓ Model accepts correct input format
✓ Output shape is correct: 1 prediction(s) for 1 input(s)
  Predicted value: 0

All tests passed! ✓
```

## 📋 Input Features

### Numeric Features
- `year`: Manufacturing year
- `km_driven`: Kilometers driven
- `mileage`: Fuel efficiency (km/l or km/kg)
- `owner`: Owner number (1-4)
- `engine`: Engine capacity (CC)
- `max_power`: Maximum power (bhp)
- `seats`: Number of seats

### Categorical Features
- `brand`: Car manufacturer (e.g., BMW, Maruti, Honda)
- `fuel`: Fuel type (Diesel, Petrol)
- `seller_type`: Type of seller (Individual, Dealer, Trustmark Dealer)
- `transmission`: Transmission type (Manual, Automatic)

## 🎓 Model Classes

### Output Classes
The model predicts one of 4 price quartiles:
- **Class 0**: Lowest price quartile
- **Class 1**: Second quartile
- **Class 2**: Third quartile
- **Class 3**: Highest price quartile

## 📈 MLflow Integration

The model is tracked and registered using MLflow:

```python
# Load from MLflow registry
import mlflow

model_uri = "models:/st126438-A3-model/latest"
loaded_model = mlflow.pyfunc.load_model(model_uri)

prediction = loaded_model.predict(sample_df)
```

**MLflow Tracking Server**: `https://mlflow.ml.brain.cs.ait.ac.th/`  
**Experiment Name**: `st126438-A3`  
**Registered Model**: `st126438-A3-model`

## 🔍 Model Architecture

```
Input Features (11) 
    ↓
Standardization (z-score)
    ↓
Linear Transformation (W·X + b)
    ↓
Softmax Activation
    ↓
Class Probabilities (4)
    ↓
Argmax → Predicted Class
```

## 📊 Data Preprocessing

1. **Cleaning**: Removed LPG/CNG fuel types, test drive cars
2. **Feature Engineering**: Extracted numeric values from text fields
3. **Missing Values**: Median imputation for numeric, mode for categorical
4. **Encoding**: Ordinal encoding for categorical variables
5. **Scaling**: Z-score standardization
6. **Target**: Converted selling price to 4 quartile classes

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License.

## 👤 Author

**Aashutosh Raut**
- GitHub: [@aashutosh-raut](https://github.com/aashutosh-raut)

## 🙏 Acknowledgments

- Dataset: Car Dekho (used cars dataset)
- Institution: Asian Institute of Technology (AIT)
- Course: Machine Learning (st126438-A3)

---