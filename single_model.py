import sys
import time
import argparse
import pandas as pd
from datetime import datetime
from tsforest import forecaster
from utils import reduce_mem_usage, get_model_params

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
train_data = pd.read_hdf('data/train_data.h5', 'train_data')
train_data.rename({"timestamp":"ds", "meter_reading":"y"}, axis=1, inplace=True)
test_data = pd.read_hdf('data/test_data.h5', 'test_data')
test_data.rename({"timestamp":"ds"}, axis=1, inplace=True)
tac = time.time()
print(f"[INFO] time elapsed loading data: {(tac-tic)/60.} min.\n")

model_kwargs = {"model_params":get_model_params(model_class_name),
                "feature_sets":['calendar', 'calendar_cyclical'],
                "exclude_features":["year","days_in_month"],
                "categorical_features":{"building_id":"default",
                                        "meter":"OneHotEncoder",
                                        "site_id":"default",
                                        "primary_use":"default"},
                "ts_uid_columns":["building_id","meter"],
                "detrend":False,
                "target_scaler":None}
fcaster = model_class(**model_kwargs)

print("[INFO] preparing the features")
tic = time.time()
fcaster.prepare_features(train_data = train_data)
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
predictions = fcaster.predict(test_data)
idx = predictions.query("y_pred < 0").index
predictions.loc[idx, "y_pred"] = 0
tac = time.time()
print(f"[INFO] time elapsed predicting: {(tac-tic)/60.} min.\n")

print("[INFO] saving submission")
tic = time.time()
submission = (predictions
              .assign(row_id = test_data.row_id.values)
              .loc[:, ["row_id","y_pred"]]
              .rename({"y_pred":"meter_reading"}, axis=1))
timestamp = datetime.now().strftime("%Y/%m/%d, %H:%M:%S").replace("/","-").replace(" ","")
submission.to_csv(f"results/preds_sm_{model_class_name}_{timestamp}.csv.gz", 
                  index=False, 
                  compression="gzip")
handler = open(f"results/preds_sm_{model_class_name}_{timestamp}.meta", "w")
handler.write(f"model_params: {fcaster.model_params}\n")
handler.write(f"input_features: {fcaster.input_features}\n")
handler.close()
tac = time.time()
print(f"[INFO] time elapsed saving submission: {(tac-tic)/60.} min.\n")
