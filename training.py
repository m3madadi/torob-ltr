import properties
from typing import List, Callable
import os
import gc
import xgboost as xgb
from sklearn.datasets import load_svmlight_file


class Iterator(xgb.DataIter):
    def __init__(self, svm_file_paths: List[str]):
        self._file_paths = svm_file_paths
        self._it = 0
        # XGBoost will generate some cache files under current directory with the prefix
        # "cache"
        super().__init__(cache_prefix=os.path.join(".", "cache"))

    def next(self, input_data: Callable):
        """Advance the iterator by 1 step and pass the data to XGBoost.  This function is
        called by XGBoost during the construction of ``DMatrix``

        """
        if self._it == len(self._file_paths):
            # return 0 to let XGBoost know this is the end of iteration
            return 0

        # input_data is a function passed in by XGBoost who has the exact same signature of
        # ``DMatrix``
        X, y = load_svmlight_file(self._file_paths[self._it])
        input_data(X, y)
        self._it += 1
        # Return 1 to let XGBoost know we haven't seen all the files yet.
        return 1

    def reset(self):
        """Reset the iterator to its beginning"""
        self._it = 0


# it = Iterator([properties.train_dat_file_path, properties.validation_dat_file_path])
# train_data = xgb.DMatrix(it)


train_data = xgb.DMatrix(properties.train_dat_file_path + "#dstrain.cache")
# validation_data = xgb.DMatrix(properties.validation_dat_file_path)

param = {
    "max_depth": 20,
    "eta": 0.3,
    "objective": "rank:ndcg",
    "verbosity": 1,
    "num_parallel_tree": 1,
    "tree_method": "hist",
    "eval_metric": ["map", "ndcg"],
}
# eval_list = [(train_data, "train"), (validation_data, "validation")]
eval_list = [(train_data, "train")]

model = xgb.train(
    param,
    train_data,
    # early_stopping_rounds=40,
    num_boost_round=200,
    evals=eval_list,
)

model.save_model(properties.model_path)