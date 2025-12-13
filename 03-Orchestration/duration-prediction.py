#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from sklearn.feature_extraction import DictVectorizer

from sklearn.metrics import root_mean_squared_error
import pickle
import numpy as np
import pickle
import xgboost as xgb
import mlflow
import sys, sklearn

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_registry_uri("http://localhost:5000")
mlflow.set_experiment("nyc-taxi-regressor")


def data_read_pre_processing(year,month):
    
    url = "https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month}.parquet"
    #reading the data from the defined path
    df = pd.read_parquet(url)
    

    #Calculating duration of the trip
    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df['duration'] = df.duration.apply(lambda td: td.total_seconds()/60)
    
    #filetring the data 
    df = df[(df.duration>=1) & (df.duration<=60)]


    categorical = ['PULocationID','DOLocationID']
    df[categorical] = df[categorical].astype(str)

    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    
    return df




def create_X(df, dv=None):


    categorical =['PU_DO']
    numerical =['trip_distance']
    dicts = df[categorical+numerical].to_dict(orient = 'records')

    if dv is None:
        dv = DictVectorizer()
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)

    return X,dv


def train_model(X_train,y_train,X_val,y_val,dv):
    
    with mlflow.start_run() as run:
        train = xgb.DMatrix(X_train,label = y_train)
        valid = xgb.DMatrix(X_val,label = y_val)    

        best_params = {
            "learning_rate" : 0.1832607471256377,
            "max_depth" : 53,
            "min_child_weight" : 1.5248259426208242,
            "objective" : "reg:squarederror",
            "reg_alpha" : 0.03914337949093102,
            "reg_lambda" : 0.20458604590859147,
            "seed" : 42,
        }

        mlflow.set_tag("developer","Tej")
        mlflow.log_params(best_params)

        booster = xgb.train(
            params =best_params,
            dtrain =train,
            num_boost_round =30,
            evals=[(valid,"validation")],
            early_stopping_rounds = 50
        )
        y_pred = booster.predict(valid)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)

        # Log preprocessor
        with open("preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)

        mlflow.log_artifact("preprocessor.b", artifact_path="preprocessor")

        #  ONLY ONE MODEL LOGGING — ZOOMCAMP STYLE
        result = mlflow.xgboost.log_model(
            xgb_model=booster,
            artifact_path="model",
            registered_model_name="nyc-taxi-regressor"
        )

        return run.info.run_id

def run(year,month):
    df_train = data_read_pre_processing(year = year,month = month)

    next_year = year if month < 12 else year + 1
    next_month = month+1 if month <12 else 1

    df_val =  data_read_pre_processing(year = next_year,month = next_month)

    X_train, dv = create_X(df_train)
    X_val, _ = create_X(df_val,dv)

    target = 'duration'
    y_train = df_train[target].values
    y_val = df_val[target].values

    run_id = train_model(X_train,y_train,X_val,y_val,dv)

    print(f"MLflow run_id:{run_id}")

    return run_id

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train a model to predict taxi trip duration.')
    parser.add_argument('--year', type=int, required=True, help='Year of the data to train on')
    parser.add_argument('--month', type=int, required=True, help='Month of the data to train on')
    args = parser.parse_args()

    run_id = run(year=args.year, month=args.month)

    with open("run_id.txt", "w") as f:
        f.write(run_id)

