import sys
import time
import argparse
import pandas as pd
from datetime import datetime
from tsforest import forecaster
from tsforest.grid_search import GridSearch
from tsforest.utils import make_time_range
from utils import reduce_mem_usage
from config import (get_model_params,
                    get_gs_hyperparams,
                    get_gs_hyperparams_fixed)
from sklearn.model_selection import StratifiedKFold

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

print("[INFO] loading data")
tic = time.time()
train_data = pd.read_hdf('data/train_data_nw.h5', 'train_data')
train_data.rename({"timestamp":"ds", "meter_reading":"y"}, axis=1, inplace=True)
tac = time.time()
print(f"[INFO] time elapsed loading data: {(tac-tic)/60.} min.\n")

print("[INFO] generating validation data")
tic = time.time()
splitter = StratifiedKFold(n_splits=4, shuffle=False, random_state=23)
valid_indexes = [valid_index for _,valid_index in splitter.split(train_data, train_data['building_id'])]
tac = time.time()
print(f"[INFO] time elapsed generating validation data: {(tac-tic)/60.} min.\n")

gs_kwargs = {"model_class":model_class,
             "feature_sets":['calendar', 'calendar_cyclical'],
             "exclude_features":["year","days_in_month"],
             "categorical_features":{"building_id":"default",
                                     "meter":"OneHotEncoder",
                                     "site_id":"default",
                                     "primary_use":"default"},
             "ts_uid_columns":["building_id","meter"],
             "detrend":False,
             "target_scaler":None,
             "n_jobs":1,
             "hyperparams":get_gs_hyperparams(model_class_name),
             "hyperparams_fixed":get_gs_hyperparams_fixed(model_class_name)}
gs = GridSearch(**gs_kwargs)

print("[INFO] fitting the grid of models")
tic = time.time()
gs.fit(train_data=train_data, valid_indexes=valid_indexes, metric="rmsle")
tac = time.time()
print(f"[INFO] time elapsed fitting the model: {(tac-tic)/60.} min.\n")

print("[INFO] storing the hyperparams")
hyperparams = gs.get_grid(top_n=10000000000000)
timestamp = datetime.now().strftime("%Y/%m/%d, %H:%M:%S").replace("/","-").replace(" ","")
hyperparams.to_csv(f"results/{model_class_name}_gs_{timestamp}.csv", index=False)
