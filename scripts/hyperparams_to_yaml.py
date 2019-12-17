import os
import yaml
import pandas as pd
from ast import literal_eval

train_data = pd.read_hdf("./data/train_data.h5", "train_data")
uid_partition = train_data.loc[:, ["meter","site_id"]].drop_duplicates().sort_values(["meter","site_id"])
hyperparams = dict()

for _,row in uid_partition.iterrows():
    hp = (pd.read_csv(f"./results/hs_meter{row.meter}_site{row.site_id}.csv", sep=";")
          .sort_values("error")
          .reset_index(drop=True))
    params = literal_eval(hp.loc[0, "params"])
    params["num_iterations"] = int(hp.loc[0, "best_iteration"])
    hyperparams[f"meter{row.meter}-site{row.site_id}"] = params

with open("./results/hyperparams.yml", "w") as file:
    yaml.dump(hyperparams, file)