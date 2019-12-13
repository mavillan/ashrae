from tsforest import forecaster
from utils import reduce_mem_usage
from config import get_model_params

def precompute_model(train_data, valid_index, model_class_name, model_kwargs):
    model_class = getattr(forecaster, model_class_name)
    fcaster = model_class(**model_kwargs)
    fcaster.prepare_features(train_data=train_data, valid_index=valid_index)
    fcaster.train_features = reduce_mem_usage(fcaster.train_features)
    if valid_index is not None:
        fcaster.valid_features = reduce_mem_usage(fcaster.valid_features)
    return fcaster

def precompute_models(train_data, valid_indexes, model_class_name, model_kwargs):
    models_by_fold = list()
    model_class = getattr(forecaster, model_class_name)
    for _,valid_index in enumerate(valid_indexes):
        fcaster = model_class(**model_kwargs)
        fcaster.prepare_features(train_data=train_data, valid_index=valid_index)
        fcaster.train_features = reduce_mem_usage(fcaster.train_features)
        fcaster.valid_features = reduce_mem_usage(fcaster.valid_features)
        models_by_fold.append(fcaster)
    return models_by_fold