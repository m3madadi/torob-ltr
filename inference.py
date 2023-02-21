import os
import pickle
import json

from tqdm import tqdm
import pandas as pd
import numpy as np
import xgboost as xgb


def read_json_lines(path, n_lines=None):
    """Creates a generator which reads and returns lines of
    a json lines file, one line at a time, each as a dictionary.
    
    This could be used as a memory-efficient alternative of `pandas.read_json`
    for reading a json lines file.
    """
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if n_lines == i:
                break
            yield json.loads(line)


data_dir = os.path.join('data')
output_dir = os.path.join('output_data')
volume_dir = ('/mnt/h/torob_data/')

test_data_path = os.path.join(data_dir, 'test-offline-data_v1.jsonl')

product_features_path = os.path.join(volume_dir, 'product_features.npy')
queries_test_features_path = os.path.join(volume_dir, 'queries_test_features.npy')
products_id_to_idx_path = os.path.join(volume_dir, 'products_id_to_idx.pkl')

predictions_path = os.path.join(output_dir, 'predictions.txt')

model_path = os.path.join(output_dir, 'ranker_full_ndcg.json')


# Load projected products and queries data.
products_projected = np.load(product_features_path)
queries_test_projected = np.load(queries_test_features_path)
with open(products_id_to_idx_path, 'rb') as f:
    products_id_to_idx = pickle.load(f)


# Load original test data which contains the result to be ranked.
test_data_df = pd.DataFrame(read_json_lines(test_data_path))

# Load trained LambdaMART model.
param = {}
model = xgb.Booster(**param)
model.load_model(model_path)

BATCH_SIZE = 64
test_predictions = []
for batch_idx in tqdm(range(0, len(test_data_df), BATCH_SIZE)):
    batch_data = test_data_df['result_not_ranked'].iloc[batch_idx:batch_idx + BATCH_SIZE]
    batch_features = []
    for test_qid, test_candidates in enumerate(batch_data, start=batch_idx):
        test_query_projected = queries_test_projected[test_qid]
        for candidate_pid in test_candidates:
            p_idx = products_id_to_idx[candidate_pid]
            features = np.concatenate((products_projected[p_idx], test_query_projected))
            batch_features.append(features)
    
    batch_features = np.stack(batch_features)
    batch_features = xgb.DMatrix(batch_features)
    batch_preds = model.predict(batch_features)
    
    start_idx = 0
    for test_candidates in batch_data:
        preds_sample = batch_preds[start_idx:start_idx + len(test_candidates)]
        sorted_idx = np.argsort(preds_sample)[::-1]
        sorted_candidates = [test_candidates[i] for i in sorted_idx]
        test_predictions.append(sorted_candidates)
        start_idx += len(test_candidates)


def write_test_predictions(predictions_path, predictions):
    lines = []
    for preds in predictions:
        lines.append(",".join([str(p_id) for p_id in preds]))

    with open(predictions_path, 'w') as f:
        f.write("\n".join(lines))

write_test_predictions(predictions_path, test_predictions)