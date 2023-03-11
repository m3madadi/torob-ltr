import properties
import xgboost as xgb

train_data = xgb.DMatrix(properties.train_dat_path)
# validation_data = xgb.DMatrix(properties.validation_dat_path)

param = {
    "max_depth": 20,
    "eta": 0.3,
    "objective": "rank:ndcg",
    "verbosity": 1,
    "num_parallel_tree": 1,
    "tree_method": "gpu_hist",
    "eval_metric": ["map", "ndcg"],
}
# eval_list = [(train_data, "train"), (validation_data, "validation")]
eval_list = [(train_data, "train")]

model = xgb.train(
    param,
    train_data,
    early_stopping_rounds=40,
    num_boost_round=250,
    evals=eval_list,
)

model.save_model(properties.model_path)