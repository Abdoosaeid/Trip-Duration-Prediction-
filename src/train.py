import joblib
from src.utils import *
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
import matplotlib as plt
import seaborn as sns
def Linear_regression(data_path):
    df = load_data(data_path)

    df  = wrangle(df)

    df = replace_outliers(df)
    print(df.columns)
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
    df = load_data(data_path)

    df  = wrangle(df)

    df = replace_outliers(df)
    column_transformer,train_features =   column_transformation(df)

    pipeline = Pipeline(steps=[
        ('ohe', column_transformer),
        ('ridge', Ridge())
    ])

    model = pipeline.fit(df[train_features], df.log_trip_duration)
    joblib.dump(model, "D:\Trip-Duration-Prediction-\models\model2_Ridge.pkl")
    print("✅ Model saved successfully at models\model2_Ridge.pkl")

if __name__=="__main__":
    path = r'D:\Trip-Duration-Prediction-\input\train.csv'

    # Train data in linear regression
    #Linear_regression(path)

    # Train data in Ridge
    ridge(path)