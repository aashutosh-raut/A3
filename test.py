import pandas as pd
import numpy as np
import joblib
from src import *

# Load model once at module level
MODEL_URI = "model/st126438-a3-model.pkl"
loaded_model = joblib.load(MODEL_URI)

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
          
def test_model_input():
    """Test that the model accepts correct input format"""
    
    # Test that model accepts the input without errors
    try:
        prediction = loaded_model.predict(sample_df)
        print("✓ Model accepts correct input format")
        return True
    except Exception as e:
        print(f"✗ Model failed to accept input: {e}")
        return False


def test_model_output_shape():
    """Test that the model output has the correct shape"""
    
    # Get prediction
    prediction = loaded_model.predict(sample_df)
    
    # Check output shape
    expected_length = len(sample_df)
    actual_length = len(prediction)
    
    if actual_length == expected_length:
        print(f"✓ Output shape is correct: {actual_length} prediction(s) for {expected_length} input(s)")
        print(f"  Predicted value: {prediction[0]}")
        return True
    else:
        print(f"✗ Output shape is incorrect: expected {expected_length}, got {actual_length}")
        return False

if __name__ == "__main__":
    import sys
    
    print("Running Model Tests...\n")

    test1_passed = test_model_input()
    test2_passed = test_model_output_shape()
    
    print("\n" + "="*50)
    if test1_passed and test2_passed:
        print("All tests passed! ✓")
        sys.exit(0)  # Exit with success code
    else:
        print("Some tests failed! ✗")
        sys.exit(1)  # Exit with failure code