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
preprocessed_products_path = os.path.join(output_dir, 'preprocessed_products.jsonl')

product_features_path = os.path.join(volume_dir, 'product_features.npy')
queries_test_features_path = os.path.join(volume_dir, 'queries_test_features.npy')
products_id_to_idx_path = os.path.join(volume_dir, 'products_id_to_idx.pkl')
# query_pca_path = os.path.join(volume_dir, 'query_pca.pkl')

predictions_path = os.path.join(output_dir, 'predictions.txt')

model_path = os.path.join(output_dir, 'ranker.json')

products_data_df = pd.DataFrame(read_json_lines(preprocessed_products_path))

# queries_test_projected = np.load(queries_test_features_path)
# products_projected = np.load(product_features_path)

with open(products_id_to_idx_path, 'rb') as f:
    products_id_to_idx = pickle.load(f)

# Load original test data which contains the result to be ranked.
test_data_df = pd.DataFrame(read_json_lines(test_data_path))


test_predictions = []
for _, test_candidates in tqdm(enumerate(test_data_df.itertuples()), total=len(test_data_df)):
    clicks = []
    views = []
    for product_id in test_candidates.result_not_ranked:
        total_clicks = products_data_df[products_data_df['id'] == product_id].total_click.values[0]
        total_view = products_data_df[products_data_df['id'] == product_id].total_view.values[0]
        clicks.append(total_clicks)
    clicks = np.array(clicks)
    sorted_idx = np.argsort(clicks)[::-1]
    sorted_candidates = [test_candidates.result_not_ranked[i] for i in sorted_idx]
    test_predictions.append(sorted_candidates)


def write_test_predictions(predictions_path, predictions):
    lines = []
    for preds in predictions:
        lines.append(",".join([str(p_id) for p_id in preds]))

    with open(predictions_path, 'w') as f:
        f.write("\n".join(lines))

write_test_predictions(predictions_path, test_predictions)
