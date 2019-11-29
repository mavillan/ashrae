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

# timestamp of the starting execution time
timestamp = datetime.now().strftime("%Y/%m/%d, %H:%M:%S").replace("/","-").replace(" ","")

AVAILABLE_CLASSES = ["CatBoostForecaster",
                     "LightGBMForecaster",
                     "XGBoostForecaster",
                     "H2OGBMForecaster"]

parser = argparse.ArgumentParser()
parser.add_argument("-m",
                    "--model_class", 
                    type=str)
parser.add_argument("-lt",
                    "--log_transform", 
                    action='store_true')
args = parser.parse_args()
print(args.log_transform)

model_class_name = args.model_class
if model_class_name not in AVAILABLE_CLASSES:
    print(f"{model_class_name} is not a valid model class.")
    sys.exit()
model_class = getattr(forecaster, model_class_name)

# file logger
logger = open(f"results/{model_class_name}_smcv_{timestamp}.meta", "w")

print("[INFO] loading data")
tic = time.time()
train_data = pd.read_hdf('data/train_data.h5', 'train_data')
train_data.rename({"timestamp":"ds", "meter_reading":"y"}, axis=1, inplace=True)
if args.log_transform:
    train_data["y"] = np.log1p(train_data["y"].values)
test_data = pd.read_hdf('data/test_data.h5', 'test_data')
test_data.rename({"timestamp":"ds"}, axis=1, inplace=True)
tac = time.time()
print(f"[INFO] time elapsed loading data: {(tac-tic)/60.} min.\n")

print("[INFO] loading validation data")
tic = time.time()
h5f = h5py.File("data/valid_sm_skfold_4fold_shuffle.h5", "r")
valid_indexes = [h5f[key][:] for key in h5f.keys()]
h5f.close()
tac = time.time()
print(f"[INFO] time elapsed loading validation data: {(tac-tic)/60.} min.\n")

model_kwargs = {"model_params":get_model_params(model_class_name),
                "feature_sets":['calendar', 'calendar_cyclical'],
                "exclude_features":["year","days_in_month"],
                "categorical_features":{"building_id":"default",
                                        "meter":"default",
                                        "site_id":"default",
                                        "primary_use":"default"},
                "ts_uid_columns":["building_id","meter"],
                "detrend":False,
                "target_scaler":None}

all_predictions = list()
for i,valid_index in enumerate(valid_indexes):
    fcaster = model_class(**model_kwargs)

    print(f"[INFO] preparing the features - fold: {i}")
    tic = time.time()
    fcaster.prepare_features(train_data=train_data, valid_index=valid_index)
    fcaster.train_features = reduce_mem_usage(fcaster.train_features)
    fcaster.valid_features = reduce_mem_usage(fcaster.valid_features)
    tac = time.time()
    print(f"[INFO] time elapsed preparing the features: {(tac-tic)/60.} min.\n")

    print(f"[INFO] fitting the model - fold: {i}")
    tic = time.time()
    fcaster.fit()
    tac = time.time()
    print(f"[INFO] time elapsed fitting the model: {(tac-tic)/60.} min.\n")

    print(f"[INFO] evaluating the model - fold: {i}")
    tic = time.time()
    valid_predictions = fcaster.predict(train_data.loc[valid_index, test_data.columns])
    if args.log_transform:
        valid_predictions["y_pred"] = np.expm1(valid_predictions["y_pred"].values)
    idx = valid_predictions.query("y_pred < 0").index
    valid_predictions.loc[idx, "y_pred"] = 0
    valid_error = compute_rmsle(train_data.loc[valid_index, "y"].values, 
                                valid_predictions.y_pred.values)
    print(f"[INFO] validation error on fold{i}: {valid_error}")
    logger.write(f"validation error on fold{i}: {valid_error}\n")
    logger.write(f"best_iteration on fold {i}: {fcaster.best_iteration}\n")
    tac = time.time()
    print(f"[INFO] time elapsed evaluating the model: {(tac-tic)/60.} min.\n")

    print(f"[INFO] predicting - fold: {i}")
    tic = time.time()
    predictions = fcaster.predict(test_data)
    if args.log_transform:
        predictions["y_pred"] = np.expm1(predictions["y_pred"].values)
    idx = predictions.query("y_pred < 0").index
    predictions.loc[idx, "y_pred"] = 0
    tac = time.time()
    print(f"[INFO] time elapsed predicting: {(tac-tic)/60.} min.\n")
    
    fcaster.save_model(f"results/{model_class_name}_smcv_{timestamp}.model_fold{i}")
    all_predictions.append(predictions.y_pred.values)

logger.write(f"model_params: {fcaster.model_params}\n")
logger.write(f"input_features: {fcaster.input_features}\n")

print("[INFO] saving submission")
tic = time.time()
predictions = np.mean(np.asarray(all_predictions), axis=0)
submission = pd.DataFrame({"row_id":test_data.row_id.values,
                           "meter_reading":predictions})
submission.to_csv(f"results/{model_class_name}_smcv_{timestamp}.csv.gz", 
                  index=False, 
                  compression="gzip")
tac = time.time()
print(f"[INFO] time elapsed saving submission: {(tac-tic)/60.} min.\n")

logger.close()
