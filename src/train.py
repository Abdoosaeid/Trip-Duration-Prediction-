import joblib
from src.utils import *
from sklearn.linear_model import LinearRegression
import os

if __name__=="__main__":
    path = r'D:\Trip-Duration-Prediction-\input\train.csv'
    df = load_data(path)

    X ,t = wrangle(df)

    model = LinearRegression()

    model.fit(X,t)

    joblib.dump(model, "D:\Trip-Duration-Prediction-\models\model1_linear_regression.pkl")
    print("✅ Model saved successfully at models/model1_linear_regression.pkl")




