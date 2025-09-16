import joblib
from src.utils import *
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge , Lasso
from sklearn.pipeline import Pipeline
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

def Linear_regression(data_path):

    df = prepare_data(data_path)

    column_transformer,train_features =   column_transformation(df)

    pipeline = Pipeline(steps=[
        ('ohe', column_transformer),
        ('regression', LinearRegression())
    ])

    model = pipeline.fit(df[train_features], df.log_trip_duration)
    print(train_features)
    joblib.dump(model, "D:\Trip-Duration-Prediction-\models\model1_linear_regression.pkl")
    print("✅ Model saved successfully at models/model1_linear_regression.pkl")


def ridge(data_path):
    df = prepare_data(data_path)

    column_transformer, train_features = column_transformation(df)


    pipeline = Pipeline(steps=[
            ('ohe', column_transformer),
            ('ridge', Ridge())
        ])

    model = pipeline.fit(df[train_features], df.log_trip_duration)
    joblib.dump(model, f"D:\\Trip-Duration-Prediction-\\models\\model2_Ridge.pkl")

    print("✅  Ridge model saved successfully!")

def lasso(data_path):
    # Load and preprocess
    df = prepare_data(data_path)

    column_transformer, train_features = column_transformation(df)


    pipeline = Pipeline(steps=[
        ('ohe', column_transformer),
        ('lasso', Lasso(alpha=0.01, random_state=42, max_iter=1000))
    ])

    model = pipeline.fit(df[train_features], df.log_trip_duration)

    joblib.dump(model, f"D:\\Trip-Duration-Prediction-\\models\\model3_Lasso.pkl")

    print("✅  Lasso model saved successfully!")

import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression

def polynomial(data_path, degree=3):
    # 1. Load dataset
    df = prepare_data(data_path)

    # 2. Create log target if not exists
    if "log_trip_duration" not in df.columns:
        df["log_trip_duration"] = np.log1p(df["trip_duration"])

    # 3. Select feature groups
    numeric_features =['pickup_latitude', 'pickup_longitude','euclidean_distance']
    categorical_features = ['pickup_dayofweek', 'pickup_month', 'hour', 'dayofyear', 'passenger_count','vendor_id']

    # 4. Column transformer
    column_transformer = ColumnTransformer([
        ("poly", Pipeline([
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("scaler", StandardScaler())
        ]), numeric_features),
        ("ohe", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ])

    # 5. Pipeline
    pipeline = Pipeline(steps=[
        ("preprocessor", column_transformer),
        ("regression", LinearRegression())
    ])

    # 6. Fit model
    X = df.drop(columns=["log_trip_duration"])
    y = df["log_trip_duration"]
    model = pipeline.fit(X, y)

    # 7. Save model
    save_path = f"D:\\Trip-Duration-Prediction-\\models\\linear_regression_poly_deg{degree}.pkl"
    joblib.dump(model, save_path)
    print(f"✅ Model trained and saved successfully at: {save_path}")


if __name__=="__main__":
    path = r'D:\Trip-Duration-Prediction-\input\train.csv'

    # Train data in linear regression
   # Linear_regression(path)

    # Train data in Ridge
   # ridge(path)
   # lasso(path)
    polynomial(path,6)
