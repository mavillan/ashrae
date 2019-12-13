import sys
import time
import yaml
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from tsforest import forecaster
from utils import reduce_mem_usage
from config import get_model_params

# loading best hyperparams for model
with open("./results/hyperparams.yml", "r") as file:
    hyperparams = yaml.load(file, Loader=yaml.FullLoader)
# available methods
AVAILABLE_CLASSES = ["CatBoostForecaster",
                     "LightGBMForecaster",
                     "XGBoostForecaster",
                     "H2OGBMForecaster"]
# excluded features to avoid data leakage
EXCLUDE_FEATURES = ["year","quarter","month","days_in_month","year_week","year_day",
                    "month_day","year_day_cos","year_day_sin","year_week_cos",
                    "year_week_sin","month_cos","month_sin","month_progress"]
# energy conversion for site0
kWh_to_kBTU = 3.4118
# correction factor for building=1099 on meter=2
kappa_building1099_meter2 = 2.190470e7/5e4
# current time
timestamp = datetime.now().strftime("%Y/%m/%d, %H:%M:%S").replace("/","-").replace(" ","")

parser = argparse.ArgumentParser()
parser.add_argument("-m",
                    "--model_class", 
                    type=str)
parser.add_argument("-lt",
                    "--log_transform", 
                    action='store_true')
args = parser.parse_args()

model_class_name = args.model_class
if model_class_name not in AVAILABLE_CLASSES:
    print(f"{model_class_name} is not a valid model class.")
    sys.exit()
model_class = getattr(forecaster, model_class_name)

print("[INFO] loading data")
tic = time.time()
# loading train data
train_data = pd.read_hdf("./data/train_data.h5", "train_data")
train_data.rename({"timestamp":"ds", "meter_reading":"y"}, axis=1, inplace=True)
# augmentation of site0 training data with leak data
leak_data = pd.read_hdf("./data/leak_data.h5", "leak_data")
leak_data.rename({"timestamp":"ds", "meter_reading":"y"}, axis=1, inplace=True)
buildings_in_train = (train_data.query("site_id==0 & ds < '2016-05-20 00:00:00'")
                      .loc[:, ["building_id","meter"]]
                      .drop_duplicates())
buildings_in_leak = (leak_data.query("site_id==0 & '2017-01-01 00:00:00' <= ds <= '2017-05-20 00:00:00'")
                     .loc[:, ["building_id","meter"]]
                     .drop_duplicates())
only_in_leak = (pd.merge(buildings_in_leak, buildings_in_train, how="left", indicator=True)
                .query("_merge == 'left_only'")
                .drop("_merge", axis=1))
leak_augmentation = pd.merge(leak_data.query("'2017-01-01 00:00:00' <= ds <= '2017-05-20 18:00:00'"),
                             only_in_leak, how="inner")
train_data = pd.concat([train_data, leak_augmentation.loc[:, train_data.columns]])
# meter_reading transform
if args.log_transform:
    train_data["y"] = np.log1p(train_data["y"].values)
    train_data["square_feet"] = np.log1p(train_data["square_feet"].values)
# loading test data
test_data = pd.read_hdf('data/test_data.h5', 'test_data')
test_data.rename({"timestamp":"ds"}, axis=1, inplace=True)
# index of test data where to apply meter_reading corrections
idx_site0_meter0 = test_data.query("site_id==0 & meter==0").index
idx_building1099_meter2 = test_data.query("building_id==1099 & meter==2").index
tac = time.time()
print(f"[INFO] time elapsed loading data: {(tac-tic)/60.} min.\n")

model_kwargs = {"feature_sets":["calendar", "calendar_cyclical"],
                "exclude_features":EXCLUDE_FEATURES,
                "categorical_features":{"building_id":"default",
                                        "primary_use":"default"},
                "ts_uid_columns":["building_id"],
                "detrend":False,
                "target_scaler":None}

all_predictions = list()
uid_partition = train_data.loc[:, ["meter","site_id"]].drop_duplicates().sort_values(["meter","site_id"])
for _,row in uid_partition.iterrows():
    print(f"[INFO] Building predictions for meter{row.meter}-site{row.site_id}")
    train_data_ = train_data.query("meter==@row.meter & site_id==@row.site_id")
    test_data_ = test_data.query("meter==@row.meter & site_id==@row.site_id")
    
    default_model_params = get_model_params(model_class_name)
    best_model_params = hyperparams[f"meter{row.meter}-site{row.site_id}"]
    model_params = {**default_model_params, **best_model_params}
    model_params["learning_rate"] = 0.01

    print("[INFO] preparing the features")
    tic = time.time()
    model_kwargs["model_params"] = model_params
    fcaster = model_class(**model_kwargs)
    fcaster.prepare_features(train_data = train_data_)
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
    predictions = fcaster.predict(test_data_)
    if args.log_transform:
        predictions["y_pred"] = np.expm1(predictions["y_pred"].values)
    idx = predictions.query("y_pred < 0").index
    predictions.loc[idx, "y_pred"] = 0
    predictions["meter"] = row.meter
    all_predictions.append(predictions)
    tac = time.time()
    print(f"[INFO] time elapsed predicting: {(tac-tic)/60.} min.\n")

predictions = (pd.merge(test_data, pd.concat(all_predictions), how="left",
                        left_on=["ds", "building_id", "meter"],
                        right_on=["ds", "building_id", "meter"])
               .loc[:, ["ds","building_id","meter","y_pred"]])
# apply meter reading corrections
predictions.loc[idx_site0_meter0, "y_pred"] = kWh_to_kBTU*predictions.loc[idx_site0_meter0, "y_pred"]
predictions.loc[idx_building1099_meter2, "y_pred"] = kappa_building1099_meter2*predictions.loc[idx_building1099_meter2, "y_pred"]

print("[INFO] replacing predictions with leak data")
tic = time.time()
leak_data = pd.read_feather("./data/leak.feather")
leak_data.meter = leak_data.meter.astype(int)
predictions_ = pd.merge(predictions, leak_data, how="left",
                        left_on=["ds","building_id","meter"],
                        right_on=["timestamp","building_id","meter"],
                        indicator=True)
idx = predictions_.query("_merge == 'both'").index
predictions_.loc[idx, "y_pred"] = predictions_.loc[idx, "meter_reading"]
predictions_.drop(["meter_reading","timestamp","_merge"], axis=1, inplace=True)
tac = time.time()
print(f"[INFO] time elapsed replacing predictions with leak data: {(tac-tic)/60.} min.\n")

print("[INFO] saving submission")
tic = time.time()
# clean submission
submission = pd.DataFrame({"row_id":test_data.row_id.values,
                           "meter_reading":predictions.y_pred.values})
submission.to_csv(f"./results/{model_class_name}_meter_site_models_{timestamp}.csv.gz", index=False, compression="gzip")
# dirty submission
submission_ = pd.DataFrame({"row_id":test_data.row_id.values,
                            "meter_reading":predictions_.y_pred.values})
submission_.to_csv(f"./results/{model_class_name}_meter_site_models_wlk_{timestamp}.csv.gz", index=False, compression="gzip")
tac = time.time()
print(f"[INFO] time elapsed saving submission: {(tac-tic)/60.} min.\n")
