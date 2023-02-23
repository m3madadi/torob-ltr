import os
import xgboost as xgb


output_dir = os.path.join('output_data')
volume_dir = ('/mnt/h/torob_data/')

train_dat_path = os.path.join(volume_dir, 'train.dat')
validation_dat_path = os.path.join(volume_dir, 'validation.dat')
model_path = os.path.join(output_dir, 'ranker_full_ndcg_306features.json')

train_data = xgb.DMatrix(train_dat_path)
validation_data = xgb.DMatrix(validation_dat_path)

param = {
    "max_depth": 20,
    "eta": 0.2,
    "objective": "rank:ndcg",
    "verbosity": 1,
    "num_parallel_tree": 1,
    "tree_method": "gpu_hist",
    "eval_metric": ["map", "ndcg"],
}
eval_list = [(train_data, "train"), (validation_data, "validation")]

model = xgb.train(
    param,
    train_data,
    early_stopping_rounds=50,
    num_boost_round=400,
    evals=eval_list,
)

model.save_model(model_path)