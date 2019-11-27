import os
import sys
import ast
import time
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from tsforest import forecaster
from tsforest.grid_search import GridSearch
from tsforest.utils import make_time_range
from utils import reduce_mem_usage 
from config import (get_model_params,
                    get_gs_hyperparams,
                    get_gs_hyperparams_fixed)

AVAILABLE_CLASSES = ["CatBoostForecaster",
                     "LightGBMForecaster",
                     "XGBoostForecaster",
                     "H2OGBMForecaster"]

parser = argparse.ArgumentParser()
parser.add_argument("-m",
                    "--model_class", 
                    type=str)
args = parser.parse_args()

model_class_name = args.model_class
if model_class_name not in AVAILABLE_CLASSES:
    print(f"{model_class_name} is not a valid model class.")
    sys.exit()
model_class = getattr(forecaster, model_class_name)

# loading the data
print("[INFO] loading data")
tic = time.time()
train_data = pd.read_hdf("data/train_data_nw.h5", "train_data")
train_data.rename({"timestamp":"ds", "meter_reading":"y"}, axis=1, inplace=True)
test_data = pd.read_hdf("data/test_data_nw.h5", "test_data")
test_data.rename({"timestamp":"ds"}, axis=1, inplace=True)
tac = time.time()
print(f"[INFO] time elapsed loading data: {(tac-tic)/60.} min.\n")


model_kwargs = {"feature_sets":['calendar', 'calendar_cyclical'],
                "exclude_features":["year","days_in_month"],
                "categorical_features":{"building_id":"default",
                                        "site_id":"default",
                                        "primary_use":"default"},
                "ts_uid_columns":["building_id"],
                "detrend":False,
                "target_scaler":None}

all_predictions_list = list()
for meter in np.sort(train_data.meter.unique()):
    print(f"Building model for meter: {meter}".center(100, "-"))
    train_data_cut = (train_data.query("meter == @meter")
                      .drop("meter", axis=1))
    test_data_cut = (test_data.query("meter == @meter")
                     .drop("meter", axis=1))
    
    hyperparams_fname = sorted([fn for fn in os.listdir("./results") 
                                if f"{model_class_name}_gs_meters{meter}" in fn])[-1]
    hyperparams = pd.read_csv(f"./results/{hyperparams_fname}")
    model_params = ast.literal_eval(hyperparams.head(1).hyperparams.values[0])
    model_params["num_iterations"] = max(hyperparams.head(1).best_iteration.values[0], 100)
    model_kwargs["model_params"] = model_params
    fcaster = model_class(**model_kwargs)

    print("[INFO] preparing the features")
    tic = time.time()
    fcaster.prepare_features(train_data = train_data_cut)
    fcaster.train_features = reduce_mem_usage(fcaster.train_features)
    tac = time.time()
    print(f"[INFO] time elapsed preparing the features: {(tac-tic)/60.} min.\n")

    print("[INFO] fitting the model")
    tic = time.time()
    fcaster.fit()
    tac = time.time()
    print(f"[INFO] time elapsed fitting the model: {(tac-tic)/60.} min.\n")

    print("[INFO] predicting")
    tic = time.time()
    predictions = fcaster.predict(test_data_cut)
    idx = predictions.query("y_pred < 0").index
    predictions.loc[idx, "y_pred"] = 0
    predictions["meter"] = meter
    all_predictions_list.append(predictions)
    tac = time.time()
    print(f"[INFO] time elapsed predicting: {(tac-tic)/60.} min.\n")

print("[INFO] saving submission")
tic = time.time()
all_predictions = pd.concat(all_predictions_list)
submission = (pd.merge(test_data.loc[:, ["row_id", "building_id", "meter", "ds"]],
                       all_predictions,
                       how="left",
                       left_on=["building_id", "meter", "ds"],
                       right_on=["building_id", "meter", "ds"])
              .loc[:, ["row_id", "y_pred"]]
              .rename({"y_pred":"meter_reading"}, axis=1))
timestamp = datetime.now().strftime("%Y/%m/%d, %H:%M:%S").replace("/","-").replace(" ","")
submission.to_csv(f"results/{model_class_name}_model_by_meter_{timestamp}.csv.gz", 
                  index=False, 
                  compression="gzip")
tac = time.time()
print(f"[INFO] time elapsed saving submission: {(tac-tic)/60.} min.\n")