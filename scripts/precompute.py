from tsforest import forecaster
from utils import reduce_mem_usage
from config import get_model_params

def precompute_models(train_data, valid_indexes, model_class_name):
    models_by_fold = list()
    
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
    model_class = getattr(forecaster, model_class_name)
    
    for i,valid_index in enumerate(valid_indexes):
        fcaster = model_class(**model_kwargs)
        fcaster.prepare_features(train_data=train_data, valid_index=valid_index)
        fcaster.train_features = reduce_mem_usage(fcaster.train_features)
        fcaster.valid_features = reduce_mem_usage(fcaster.valid_features)
        model_by_fold.append(fcaster)

    return models_by_fold