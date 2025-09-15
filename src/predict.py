from utils import *
from sklearn.metrics import mean_squared_error ,r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
def predict_eval(model, train, train_features, name):
    print(train_features)
    y_train_pred = model.predict(train[train_features])
    rmse = mean_squared_error(train.log_trip_duration, y_train_pred, squared=False)
    r2 = r2_score(train.log_trip_duration, y_train_pred)
    print(f"{name} RMSE = {rmse:.4f} - R2 = {r2:.4f}")
    print([y_train_pred[0:10]])
    print(train.log_trip_duration[0:10])

if __name__=="__main__":
    path = r'D:\Trip-Duration-Prediction-\input\val.csv'
    df = load_data(path)
    df  = wrangle(df)
    df = replace_outliers(df)
    _,train_features = column_transformation(df)


    # load linear regression
    #model1 = joblib.load( "D:\Trip-Duration-Prediction-\models\model1_linear_regression.pkl")
    model2 = joblib.load( "D:\Trip-Duration-Prediction-\models\model2_Ridge.pkl")


    predict_eval(model2,df,train_features,"Train")