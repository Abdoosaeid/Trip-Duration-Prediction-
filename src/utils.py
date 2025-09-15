import pandas as pd
import numpy as np
import math
from sklearn.compose import ColumnTransformer , make_column_selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def load_data(dataset_path):

    df = pd.read_csv(dataset_path)

    return df

def column_transformation(df):
    numeric_features = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude','euclidean_distance']
    categorical_features = ['pickup_dayofweek', 'pickup_month', 'hour', 'dayofyear', 'passenger_count','vendor_id']
    train_features = categorical_features + numeric_features

    column_transformer = ColumnTransformer([
        ('ohe', OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ('scaling', StandardScaler(), numeric_features)
    ]
        , remainder='passthrough'
    )

    return column_transformer ,train_features

def replace_outliers(df, q1=0.01, q3=0.99):
    df_clean = df.copy()

    for col in df_clean.select_dtypes(include=[np.number]).columns:
        Q1 = df_clean[col].quantile(q1)
        Q3 = df_clean[col].quantile(q3)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = (df_clean[col] < lower) | (df_clean[col] > upper)
        mean_val = df_clean[col].mean()

        df_clean.loc[outliers, col] = mean_val

    return df_clean


def euclidean_distance_row(row):
    # constants
    km_per_degree_lat = 111  # approx. km per degree latitude

    # longitude correction depends on latitude
    avg_lat_rad = math.radians((row['pickup_latitude'] + row['dropoff_latitude']) / 2)
    km_per_degree_lon = 111 * math.cos(avg_lat_rad)

    dx = (row['dropoff_longitude'] - row['pickup_longitude']) * km_per_degree_lon
    dy = (row['dropoff_latitude'] - row['pickup_latitude']) * km_per_degree_lat

    return math.sqrt(dx**2 + dy**2)



def wrangle(df):
    df = df.drop(["id", "store_and_fwd_flag"], axis=1)

    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])

    df["pickup_year"] = df["pickup_datetime"].dt.year
    df["pickup_month"] = df["pickup_datetime"].dt.month
    df["pickup_day"] = df["pickup_datetime"].dt.day
    df['hour'] = df.pickup_datetime.dt.hour
    df["pickup_dayofweek"] = df["pickup_datetime"].dt.dayofweek
    df['dayofyear'] = df.pickup_datetime.dt.dayofyear

    df = df.drop(columns=["pickup_datetime"],axis=1)

    df = df.drop(columns=["pickup_year"],axis=1)

    df['log_trip_duration'] = np.log1p(df.trip_duration)
    df.drop('trip_duration', axis=1, inplace=True)
    df['euclidean_distance'] = df.apply(euclidean_distance_row, axis=1)

    return df






