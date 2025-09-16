from utils import *
from sklearn.metrics import mean_squared_error ,r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

import pandas as pd
from sklearn.metrics import root_mean_squared_error, r2_score

def predict_eval(model, df, train_features, name):
    y_true = df.log_trip_duration
    y_pred = model.predict(df[train_features])

    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {"Model": name, "RMSE": rmse, "R2": r2}


if __name__ == "__main__":
    import os, joblib

    path = r"D:\Trip-Duration-Prediction-\input\test.csv"
    df = prepare_data(path)
    _, train_features = column_transformation(df)

    models_dir = r"D:\Trip-Duration-Prediction-\models"

    results = []

    for model_file in os.listdir(models_dir):
        if model_file.endswith(".pkl"):
            model_path = os.path.join(models_dir, model_file)
            model = joblib.load(model_path)

            res = predict_eval(model, df, train_features, model_file)
            results.append(res)

    # اعرض النتائج كجدول
    results_df = pd.DataFrame(results)
    print(results_df)
