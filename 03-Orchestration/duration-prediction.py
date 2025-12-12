#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('python -V')


# In[ ]:





# In[2]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
import pickle
import sklearn
import numpy as np


# In[3]:


import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_registry_uri("sqlite:///mlflow.db")
mlflow.set_experiment("taxi-exp-2")



# In[4]:


def data_read_pre_processing(file_path):
    
   
    #reading the data from the defined path
    df = pd.read_parquet(file_path)
    
    #Converting the columns to timestamp
    df.lpep_pickup_datetime = pd.to_datetime(df.lpep_pickup_datetime)
    df.lpep_dropoff_datetime = pd.to_datetime(df.lpep_dropoff_datetime)
    
    #Calculating duration of the trip
    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df['duration'] = df.duration.apply(lambda td: td.total_seconds()/60)
    
    #filetring the data 
    df = df[(df.duration>=1) & (df.duration<=60)]
    
    return df


# In[5]:


df_train = data_read_pre_processing('https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2021-01.parquet')

df_val = data_read_pre_processing('https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2021-02.parquet')


# In[6]:


import sys, sklearn
print(sys.executable)
print(sklearn.__version__)


# In[ ]:





# In[7]:


#### Created new Feature
df_train['PU_DO'] = df_train['PULocationID'].astype(str) + '_' + df_train['DOLocationID'].astype(str)
df_val['PU_DO']  = df_val['PULocationID'].astype(str)  + '_' + df_val['DOLocationID'].astype(str)


# In[8]:


### defining the train set 
categorical = ['PU_DO']#['PULocationID','DOLocationID']
numerical = ['trip_distance']

### Converting categorical variable into string for preprocessing
df_train[categorical] = df_train[categorical].astype(str)
train_dicts = df_train[categorical+numerical].to_dict(orient = 'records') 

### Converting categorical variable into string for preprocessing
df_val[categorical] = df_val[categorical].astype(str)
val_dicts = df_val[categorical+numerical].to_dict(orient = 'records') 


# In[9]:


### Vectoriser
dv = DictVectorizer()


# In[10]:


## vectorizing traing Features
X_train = dv.fit_transform(train_dicts)
X_train


# In[11]:


## vectorizing validation Features
X_val = dv.transform(val_dicts)
X_val


# In[12]:


target = 'duration'
y_train = df_train[target].values
y_val = df_val[target].values


# In[13]:


lr = LinearRegression()
lr.fit(X_train,y_train)


# In[14]:


# prediting the values for train set
y_predict_train = lr.predict(X_train)

# prediting the values for train set
y_predict_val = lr.predict(X_val)


# In[15]:


import inspect
inspect.signature(mean_squared_error)


# In[16]:


rmse = np.sqrt(mean_squared_error(y_train,y_predict_train))
rmse


# In[17]:


rmse_val =  mean_squared_error(y_val,y_predict_val,squared = False)
rmse_val


# In[18]:


with open('models/lin_reg.bin','wb') as f_out:
    pickle.dump((dv,lr),f_out)


# # Checking a different Model like LASSO or Ridge

# In[19]:


with mlflow.start_run():

    mlflow.set_tag("developer","Tej")

    mlflow.log_param("train-data-url","https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2021-01.parquet")
    mlflow.log_param("validation-data-url","https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2021-02.parquet")
    
    alpha =0.1
    mlflow.log_param("alpha",alpha)
    lr_new = Lasso(alpha)
    lr_new.fit(X_train,y_train)
    
    y_pred_new = lr_new.predict(X_val)
    rmse = mean_squared_error(y_val,y_pred_new,squared = False)
    mlflow.log_metric("rmse",rmse)
    mlflow.log_artifact(local_path = "models/lin_reg.bin", artifact_path = "models_pickle")


# ## Performing Hyper Parameter Tuning and loggoing the best results in ML flow  

# In[20]:


import xgboost as xgb

from hyperopt import fmin,hp,tpe,STATUS_OK, Trials
from hyperopt.pyll import scope


# In[21]:


train = xgb.DMatrix(X_train,label = y_train)
valid = xgb.DMatrix(X_val,label = y_val)


# In[22]:


def objective(params):
    with mlflow.start_run():
        mlflow.set_tag("model","xgboost")
        mlflow.log_params(params)
        booster = xgb.train(
            params =params,
            dtrain =train,
            num_boost_round =10,
            evals=[(valid,"validation")],
            early_stopping_rounds = 50
        )
        y_pred = booster.predict(valid)
        rmse = mean_squared_error(y_val,y_pred,squared = False)
        mlflow.log_metric("rmse", rmse)

    return {"loss":rmse,'status':STATUS_OK}
        

        
        


# In[23]:


search_space = {
    'max_depth' :scope.int(hp.quniform('max_depth',4,100,1)),
    'learning_rate' :hp.loguniform('learning_rate',-3,0),
    'reg_alpha' :hp.loguniform('reg_alpha',-5,-1),
    'reg_lambda' :hp.loguniform('reg_lambda',-6,-1),
    'min_child_weight' :hp.loguniform('min_child_weight',-1,3),
    'objective' :'reg:squarederror',
    'seed' : 42,
}

best_result =fmin(
    fn=objective,
    space=search_space,
    algo=tpe.suggest,
    max_evals=50,
    trials=Trials()
    )


# In[24]:


import os, pickle
with mlflow.start_run():

    best_params = {
    "learning_rate" : 0.1832607471256377,
    "max_depth" : 53,
    "min_child_weight" : 1.5248259426208242,
    "objective" : "reg:squarederror",
    "reg_alpha" : 0.03914337949093102,
    "reg_lambda" : 0.20458604590859147,
    "seed" : 42,
    }
    mlflow.log_params(best_params)
    booster = xgb.train(
            params =best_params,
            dtrain =train,
            num_boost_round =100,
            evals=[(valid,"validation")],
            early_stopping_rounds = 50
        )

    y_pred = booster.predict(valid)
    rmse = mean_squared_error(y_val,y_pred,squared = False)
    mlflow.log_metric("rmse", rmse)
    #os.makedirs("models", exist_ok=True)
    with open('models/preprocessor.b','wb') as f_out:
       pickle.dump(dv,f_out)
    mlflow.log_artifact("models/preprocessor.b",artifact_path = "preprocessor")
    
    mlflow.xgboost.log_model(booster,name="xgb_model")
    


# In[28]:


with mlflow.start_run():

    best_params = {
        "learning_rate" : 0.1832607471256377,
        "max_depth" : 53,
        "min_child_weight" : 1.5248259426208242,
        "objective" : "reg:squarederror",
        "reg_alpha" : 0.03914337949093102,
        "reg_lambda" : 0.20458604590859147,
        "seed" : 42
    }

    mlflow.log_params(best_params)

    booster = xgb.train(
        params=best_params,
        dtrain=train,
        num_boost_round=100,
        evals=[(valid, "validation")],
        early_stopping_rounds=50
    )

    y_pred = booster.predict(valid)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
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

print(result.model_uri)


# In[26]:


result = mlflow.xgboost.log_model(
    xgb_model=booster,
    artifact_path="model",
    registered_model_name="nyc-taxi-regressor"
)

print("Model logged at:")
print(result.model_uri)


# In[27]:





# In[ ]:




