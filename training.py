import os
import xgboost as xgb


output_dir = os.path.join('output_data')
volume_dir = ('/mnt/h/torob_data/')

train_dat_path = os.path.join(volume_dir, 'train.dat')
model_path = os.path.join(output_dir, 'ranker_full_ndcg.json')

train_data = xgb.DMatrix(train_dat_path)

param = {
    "max_depth": 20,
    "eta": 0.3,
    "objective": "rank:ndcg",
    "verbosity": 1,
    "num_parallel_tree": 1,
    "tree_method": "gpu_hist",
    "eval_metric": ["ndcg"],
}
eval_list = [(train_data, "train")]

model = xgb.train(
    param,
    train_data,
    num_boost_round=200,
    evals=eval_list,
)

model.save_model(model_path)