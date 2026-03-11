import pickle
import pandas as pd

class Insurance_Prediction:
    def __init__(self):
        with open("artifacts/model.pkl","rb") as f:
            self.model=pickle.load(f)
        with open("artifacts/scaler.pkl","rb") as f:
            self.scaler=pickle.load(f)
    
    def predict(self, Age, Annual_Income_LPA, Policy_Term_Years, Sum_Assured_Lakhs):
        # Create DataFrame with proper column names to match training data
        X = pd.DataFrame({
            'Age': [Age],
            'Annual_Income_LPA': [Annual_Income_LPA],
            'Policy_Term_Years': [Policy_Term_Years],
            'Sum_Assured_Lakhs': [Sum_Assured_Lakhs]
        })
        X_scaled = self.scaler.transform(X)
        prediction = self.model.predict(X_scaled)
        return float(prediction[0][0])  # Extract scalar value from 2D array