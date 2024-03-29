import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import h5py
from datetime import datetime
from tsforest import forecaster
from tsforest.metrics import compute_rmse, compute_rmsle
from utils import reduce_mem_usage
from config import get_model_params
from scaling import target_transform, target_inverse_transform
from precompute import precompute_model, precompute_models
import optuna
import copy

# available methods
AVAILABLE_CLASSES = ["CatBoostForecaster",
                     "LightGBMForecaster",
                     "XGBoostForecaster",
                     "H2OGBMForecaster"]

# excluded features to avoid data leakage
#EXCLUDE_FEATURES = ["year","days_in_month","year_day",
#                    "month_day","year_day_cos","year_day_sin"]
EXCLUDE_FEATURES = ["year","quarter","month","days_in_month","year_week","year_day",
                    "month_day","year_day_cos","year_day_sin","year_week_cos",
                    "year_week_sin","month_cos","month_sin","month_progress"]

parser = argparse.ArgumentParser()
parser.add_argument("-m",
                    "--model_class", 
                    type=str)
parser.add_argument("-s",
                    "--site", 
                    type=int)
parser.add_argument("-m",
                    "--meter, 
                    type=int)
args = parser.parse_args()

if os.path.exists(f"../results/hs_site{args.site}_meter{args.meter}.csv"):
    logger = open(f"../results/hs_site{args.site}_meter{args.meter}.csv", "a")
else:
    logger = open(f"../results/hs_site{args.site}_meter{args.meter}.csv", "w")
    logger.write("trial;params;best_iteration;error\n")

model_class_name = args.model_class
if model_class_name not in AVAILABLE_CLASSES:
    print(f"{model_class_name} is not a valid model class.")
    sys.exit()
model_class = getattr(forecaster, model_class_name)

print("[INFO] loading data")
tic = time.time()
# loading train data
train_data = (pd.read_hdf("../data/train_data.h5", "train_data")
              .query(f"site_id == {args.site} & meter == {args.meter}"))
train_data.rename({"timestamp":"ds", "meter_reading":"y"}, axis=1, inplace=True)
# loading leak data
leak_data = (pd.read_feather("../data/leakage.feather")
             .query(f"site_id == {args.site} & meter == {args.meter}")
             .pipe(reduce_mem_usage)
             .query("timestamp >= '2017-01-01 00:00:00'"))
leak_data.rename({"timestamp":"ds", "meter_reading":"y"}, axis=1, inplace=True)
# merge of both datasets
train_data = (pd.concat([train_data, leak_data.loc[:, train_data.columns]])
              .reset_index(drop=True))
train_data["y"] = np.log1p(train_data["y"].values)
predict_columns = [feat for feat in train_data.columns if feat!="y"]

if args.site == 0: 
    valid_index = train_data.query("ds >= '2017-05-21 00:00:00'").index
else: 
    valid_index = train_data.query("ds >= '2017-01-01 00:00:00'").index
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
        "num_leaves":int(trial.suggest_discrete_uniform("num_leaves", 8, 32, 8)),
        "learning_rate":trial.suggest_discrete_uniform("learning_rate", 0.01, 0.1, 0.0025),
        "min_data_in_leaf":int(trial.suggest_discrete_uniform("min_data_in_leaf", 5, 25, 10)),
        "feature_fraction":trial.suggest_discrete_uniform("feature_fraction", 0.8, 1.0, 0.1),
        "lambda_l2":trial.suggest_discrete_uniform("lambda_l2", 0., 3.0, 1.0)
    }
    default_model_params = get_model_params(model_class_name)
    model_params = {**default_model_params, **sampled_params}

    print(f"[INFO] preparing the features")
    tic = time.time()
    fcaster = copy.deepcopy(precomputed_model)
    fcaster.set_params(model_params=model_params)
    tac = time.time()
    print(f"[INFO] time elapsed preparing the features: {(tac-tic)/60.} min.\n")

    print(f"[INFO] fitting the model")
    tic = time.time()
    fcaster.fit(fit_kwargs={"verbose_eval":10})
    tac = time.time()
    print(f"[INFO] time elapsed fitting the model: {(tac-tic)/60.} min.\n")
        
    valid_error = (fcaster.model.model.best_score["valid_0"]["l2"])**0.5
    best_iteration = fcaster.best_iteration
    print(f"[INFO] validation error: {valid_error}")
    print(f"[INFO] best iteration: {best_iteration}")
    
    logger.write(f"{trial.number};{sampled_params};{best_iteration};{valid_error}\n")
    logger.flush()
    return valid_error

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)
logger.close()
study_dataframe = study.trials_dataframe()
study_dataframe.to_csv(f"results/study_site{args.site}_meter{args.meter}.csv")
