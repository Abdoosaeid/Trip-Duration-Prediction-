from utils import *
from sklearn.metrics import mean_squared_error ,r2_score
import joblib

if __name__=="__main__":
    path = r'D:\Trip-Duration-Prediction-\input\train.csv'
    df = load_data(path)

    X,t = wrangle(df)

    model = joblib.load( "D:\Trip-Duration-Prediction-\models\model1_linear_regression.pkl")

    y_predict = model.predict(X)

    print("mean square error: ",mean_squared_error(y_predict,t))
    print("R2 score : ",r2_score(y_predict,t))