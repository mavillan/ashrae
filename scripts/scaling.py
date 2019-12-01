import pandas as pd

def target_transform(data, scaling, target="y"):
    data = pd.merge(data, scaling, how="left", on=["building_id","meter"])
    data[target] = data.eval(f"({target}-center)/scale")
    data.drop(["center","scale"], axis=1, inplace=True)
    return data

def target_inverse_transform(data, scaling, target="y"):
    data = pd.merge(data, scaling, how="left", on=["building_id","meter"])
    data[target] = data.eval(f"({target}*scale)+center")
    data.drop(["center","scale"], axis=1, inplace=True)
    return data
