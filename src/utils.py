import pandas as pd
import numpy as np
import math

def load_data(dataset_path):

    df = pd.read_csv(dataset_path)

    return df



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
    df = df.drop(columns=["id", "store_and_fwd_flag"], axis=True)

    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])

    df["pickup_year"] = df["pickup_datetime"].dt.year
    df["pickup_month"] = df["pickup_datetime"].dt.month
    df["pickup_day"] = df["pickup_datetime"].dt.day
    df["pickup_hour"] = df["pickup_datetime"].dt.hour + df["pickup_datetime"].dt.minute / 60
    df["pickup_dayofweek"] = df["pickup_datetime"].dt.dayofweek

    df = df.drop(columns=["pickup_datetime"])

    df = df.drop(columns=["pickup_year"])

    df['trip_duration_log'] = np.log1p(df['trip_duration'])
    df.drop('trip_duration', axis=1, inplace=True)

    X = df.drop("trip_duration_log",axis=1)
    t = df["trip_duration_log"]

    return X ,t






