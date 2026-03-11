import pickle
class Insurance_Prediction:
    def __init__(self):
        with open("artifacts/model.pkl","rb") as f:
            self.model=pickle.load(f)
        with open("artifacts/scaler.pkl","rb") as f:
            self.scaler=pickle.load(f)

    def predict(self, Age, Annual_Income_LPA, Policy_Term_Years, Sum_Assured_Lakhs):
        X = [[Age, Annual_Income_LPA, Policy_Term_Years, Sum_Assured_Lakhs]]
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)[0]