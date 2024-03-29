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
from precompute import precompute_models
import optuna
import copy

# timestamp of the starting execution time
timestamp = datetime.now().strftime("%Y/%m/%d, %H:%M:%S").replace("/","-").replace(" ","")

AVAILABLE_CLASSES = ["CatBoostForecaster",
                     "LightGBMForecaster",
                     "XGBoostForecaster",
                     "H2OGBMForecaster"]

# excluded features to avoid data leakage
EXCLUDE_FEATURES = ["year","quarter","month","days_in_month","year_week","year_day",
                    "month_day","year_day_cos","year_day_sin","year_week_cos",
                    "year_week_sin","month_cos","month_sin"]

parser = argparse.ArgumentParser()
parser.add_argument("-m",
                    "--model_class", 
                    type=str)
parser.add_argument("-lt",
                    "--log_transform", 
                    action='store_true')
parser.add_argument("-st",
                    "--scale_transform", 
                    action='store_true')
args = parser.parse_args()

model_class_name = args.model_class
if model_class_name not in AVAILABLE_CLASSES:
    print(f"{model_class_name} is not a valid model class.")
    sys.exit()
model_class = getattr(forecaster, model_class_name)

print("[INFO] loading data")
tic = time.time()
train_data = pd.read_hdf('data/train_data.h5', 'train_data')
train_data.rename({"timestamp":"ds", "meter_reading":"y"}, axis=1, inplace=True)
predict_columns = [feat for feat in train_data.columns if feat!="y"]
if args.log_transform:
    train_data["y"] = np.log1p(train_data["y"].values)
if args.scale_transform:
    robust_scaler = pd.read_csv("data/robust_scaler.csv")
    train_data = target_transform(train_data, robust_scaler, target="y")
tac = time.time()
print(f"[INFO] time elapsed loading data: {(tac-tic)/60.} min.\n")

print("[INFO] loading validation data")
tic = time.time()
h5f = h5py.File("data/valid_sm_custom_3fold.h5", "r")
valid_indexes = [h5f[key][:] for key in h5f.keys()]
h5f.close()
tac = time.time()
print(f"[INFO] time elapsed loading validation data: {(tac-tic)/60.} min.\n")

print("[INFO] precomputing the models")
tic = time.time()
model_kwargs = {"feature_sets":['calendar', 'calendar_cyclical'],
                "exclude_features":EXCLUDE_FEATURES,
                "categorical_features":{"building_id":"default",
                                        "meter":"default",
                                        "site_id":"default",
                                        "primary_use":"default"},
                "ts_uid_columns":["building_id","meter"],
                "detrend":False,
                "target_scaler":None}
models_by_fold = precompute_models(train_data, valid_indexes, model_class_name, model_kwargs)
n_folds = len(models_by_fold)
tac = time.time()
print(f"[INFO] time elapsed precomputing the features: {(tac-tic)/60.} min.\n")
   
def objective(trial):
    sampled_params = {
        "num_leaves":int(trial.suggest_loguniform('num_leaves', 2**6, 2**10+1)),
        "learning_rate":trial.suggest_uniform('learning_rate', 0.05, 0.31),
        "min_data_in_leaf":int(trial.suggest_discrete_uniform("min_data_in_leaf", 20, 40, 20)),
        "feature_fraction":trial.suggest_discrete_uniform("feature_fraction", 0.8, 1.0, 0.1),
        "lambda_l2":trial.suggest_discrete_uniform("lambda_l2", 0., 3.0, 1.0)
    }
    default_model_params = get_model_params(model_class_name)
    model_params = {**default_model_params, **sampled_params}

    valid_errors = list()
    for fold,valid_index in enumerate(valid_indexes):
        print(f"[INFO] preparing the features - fold: {fold}")
        tic = time.time()
        fcaster = copy.deepcopy(models_by_fold[fold])
        fcaster.set_params(model_params=model_params)
        tac = time.time()
        print(f"[INFO] time elapsed preparing the features: {(tac-tic)/60.} min.\n")

        print(f"[INFO] fitting the model - fold: {fold}")
        tic = time.time()
        fcaster.fit(fit_kwargs={"verbose_eval":20})
        tac = time.time()
        print(f"[INFO] time elapsed fitting the model: {(tac-tic)/60.} min.\n")
        
        valid_error = (fcaster.model.model.best_score["valid_0"]["l2"])**0.5
        best_iteration = fcaster.best_iteration
        print(f"[INFO] validation error on fold{fold}: {valid_error}")
        print(f"[INFO] best iteration on fold{fold}: {best_iteration}")
        valid_errors.append(valid_error)
    
    return np.mean(valid_errors)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=200)
study_dataframe = study.trials_dataframe()
study_dataframe.to_csv("results/study_03.csv")
