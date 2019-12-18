import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import h5py
import copy
from datetime import datetime
from tsforest import forecaster
from tsforest.metrics import compute_rmse, compute_rmsle
from utils import reduce_mem_usage
from config import get_model_params
from scaling import target_transform, target_inverse_transform
from precompute import precompute_model, precompute_models
import optuna
from optuna.integration import LightGBMPruningCallback

# available methods
AVAILABLE_CLASSES = ["CatBoostForecaster",
                     "LightGBMForecaster",
                     "XGBoostForecaster",
                     "H2OGBMForecaster"]
# excluded features to avoid data leakage
EXCLUDE_FEATURES = ["year","month","days_in_month","year_week","year_day",
                    "month_day","year_day_cos","year_day_sin","year_week_cos",
                    "year_week_sin","month_cos","month_sin","month_progress"]

parser = argparse.ArgumentParser()
parser.add_argument("-m",
                    "--model_class", 
                    type=str)
parser.add_argument("-mt",
                    "--meter",
                    type=int)
parser.add_argument("-st",
                    "--site",
                    type=int)
args = parser.parse_args()

if os.path.exists(f"./results/hs_lgb_meter{args.meter}_site{args.site}.csv"):
    logger = open(f"./results/hs_lgb_meter{args.meter}_site{args.site}.csv", "a")
else:
    logger = open(f"./results/hs_lgb_meter{args.meter}_site{args.site}.csv", "w")
    logger.write("trial;params;best_iteration;error\n")

model_class_name = args.model_class
if model_class_name not in AVAILABLE_CLASSES:
    print(f"{model_class_name} is not a valid model class.")
    sys.exit()
model_class = getattr(forecaster, model_class_name)

print("[INFO] loading data")
tic = time.time()
# loading train data
train_data = pd.read_csv(f"./mirrors/train_data_meter{args.meter}_site{args.site}.csv", parse_dates=["timestamp"])
train_data.rename({"timestamp":"ds", "meter_reading":"y"}, axis=1, inplace=True)
# loading leak data
leak_data = (pd.read_csv(f"./mirrors/leak_data_meter{args.meter}_site{args.site}.csv", parse_dates=["timestamp"])
             .query("timestamp >= '2017-01-01 00:00:00'"))
leak_data.rename({"timestamp":"ds", "meter_reading":"y"}, axis=1, inplace=True)
# merge of both datasets
train_data = (pd.concat([train_data, leak_data.loc[:, train_data.columns]])
              .reset_index(drop=True))
train_data["median_reading"] = np.log1p(train_data["median_reading"].values)
train_data["square_feet"] = np.log1p(train_data["square_feet"].values)
train_data["y"] = np.log1p(train_data["y"].values)
# index for validation data
if args.meter==0:
    valid_index = train_data.query("site_id != 0 & ds >= '2017-01-01 00:00:00'").index
    valid_index = valid_index.union(train_data.query("site_id == 0 & ds >= '2017-05-21 00:00:00'").index)
elif args.meter==1:
    valid_index = train_data.query("site_id != 0 & ds >= '2017-01-01 00:00:00'").index
    valid_index = valid_index.union(train_data.query("site_id == 0 & ds >= '2017-03-01 00:00:00'").index)
else:
    valid_index = train_data.query("ds >= '2017-01-01 00:00:00'").index
# removes not useful columns
train_data.drop(["site_id","meter"], axis=1, inplace=True)
predict_columns = [feat for feat in train_data.columns if feat!="y"]
tac = time.time()
print(f"[INFO] time elapsed loading data: {(tac-tic)/60.} min.\n")

print("[INFO] precomputing the model")
tic = time.time()
model_kwargs = {"feature_sets":['calendar', 'calendar_cyclical'],
                "exclude_features":EXCLUDE_FEATURES,
                "categorical_features":{"building_id":"default",
                                        "primary_use":"default"},
                "ts_uid_columns":["building_id"],
                "detrend":False,
                "target_scaler":None}
precomputed_model = precompute_model(train_data, valid_index, model_class_name, model_kwargs)
tac = time.time()
print(f"[INFO] time elapsed precomputing the features: {(tac-tic)/60.} min.\n")
   
def objective(trial):
    sampled_params = {
        "num_leaves":trial.suggest_int("num_leaves", 32, 1024),
        "min_data_in_leaf":trial.suggest_int("min_data_in_leaf", 1, 30),
        "feature_fraction":trial.suggest_discrete_uniform("feature_fraction", 0.5, 1.0, 0.1),
        "feature_fraction_bynode":trial.suggest_discrete_uniform("feature_fraction_bynode", 0.9, 1.0, 0.05),
        "lambda_l2":trial.suggest_discrete_uniform("lambda_l2", 0., 5.0, 1.0)
    }
    default_model_params = get_model_params(model_class_name)
    model_params = {**default_model_params, **sampled_params}
    model_params["learning_rate"] = 0.01

    print(f"[INFO] preparing the features")
    tic = time.time()
    fcaster = copy.deepcopy(precomputed_model)
    fcaster.set_params(model_params=model_params)
    tac = time.time()
    print(f"[INFO] time elapsed preparing the features: {(tac-tic)/60.} min.\n")

    print(f"[INFO] fitting the model")
    tic = time.time()
    fcaster.fit(fit_kwargs={"verbose_eval":20})
    tac = time.time()
    print(f"[INFO] time elapsed fitting the model: {(tac-tic)/60.} min.\n")
        
    valid_error = fcaster.model.model.best_score["valid_0"]["rmse"]
    best_iteration = fcaster.best_iteration
    print(f"[INFO] validation error: {valid_error}")
    print(f"[INFO] best iteration: {best_iteration}")
    
    logger.write(f"{trial.number};{sampled_params};{best_iteration};{valid_error}\n")
    logger.flush()
    return valid_error

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
logger.close()
