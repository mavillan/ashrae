import numpy as np

def get_model_params(model_class_name):
    if model_class_name == "CatBoostForecaster":
        return {"iterations":1000,
                "early_stopping_rounds":50,
                "learning_rate":0.01,
                "l2_leaf_reg":1.0,
                "depth":6,
                "has_time":False,
                "bootstrap_type":"No",
                "logging_level":"Info",
                "use_best_model":True}
    elif model_class_name == "LightGBMForecaster":
        return {"boosting_type":"gbrt",
                "objective":"regression",
                "num_iterations":500,
                "early_stopping_rounds":50,
                "num_leaves":512,
                "min_data_in_leaf":5,
                "learning_rate":0.01,
                "feature_fraction":0.7,
                "lambda_l2":1.0,
                "verbosity":1}
    elif model_class_name == "XGBoostForecaster":
        return {"objective":"reg:squarederror",
                "eval_metric":"rmse",
                "num_boost_round":1000,
                "early_stopping_rounds":50,
                "learning_rate":0.01,
                "max_depth":6,
                "lambda":1,
                "min_child_weight":1,
                "colsample_bytree":0.7,
                "verbosity":1}
    elif model_class_name == "H2OGBMForecaster":
        return {"ntrees":1000,
                "max_depth":6,
                "nbins":20,
                "learn_rate":0.3,
                "stopping_metric":"mse",
                "score_each_iteration":True,
                "categorical_encoding":"enum",
                "sample_rate":1.0,
                "col_sample_rate":1.0,
                "min_rows":20,
                "distribution":"gaussian",
                "verbose":True}


def get_gs_hyperparams(model_class_name):
    if model_class_name == "LightGBMForecaster":
        return {"num_leaves":(2**np.arange(5, 11.1, 0.5)).astype(int),
                "learning_rate":[0.01, 0.05, 0.1, 0.2],
                "min_data_in_leaf":[20, 50],
                "feature_fraction":[0.9, 1.0],
                "lambda_l2":[0., 1.]}

def get_gs_hyperparams_fixed(model_class_name):
    if model_class_name == "LightGBMForecaster":
        return {"num_iterations":1000,
                "early_stopping_round":100}