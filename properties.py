import os
import torch

# Number of training samples to use (set to None to use all samples).
NUM_TRAIN_SAMPLES = None
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EMBEDDING_DIM = 128

data_dir = os.path.join('data')
output_dir = os.path.join('output_data')
volume_dir = ('/mnt/h/torob_data/')

model_path = os.path.join(output_dir, 'ranker.json')

predictions_path = os.path.join(output_dir, 'predictions.txt')

test_data_path = os.path.join(data_dir, 'test-offline-data_v1.jsonl')

aggregated_search_data_path = os.path.join(output_dir, 'aggregated_search_data.jsonl')
preprocessed_products_path = os.path.join(output_dir, 'preprocessed_products.jsonl')
preprocessed_test_queries_path = os.path.join(output_dir, 'preprocessed_test_queries.jsonl')

train_dat_file_path = os.path.join(volume_dir, 'train.dat')
validation_dat_file_path = os.path.join(volume_dir, 'validation.dat')

# random_projection_mat_path = os.path.join(output_dir, 'random_projection_mat.npy')
product_name_features_path = os.path.join(volume_dir, 'product_name_features.npy')
product_category_features_path = os.path.join(volume_dir, 'product_category_features.npy')
queries_train_features_path = os.path.join(volume_dir, 'queries_train_features.npy')
queries_test_features_path = os.path.join(volume_dir, 'queries_test_features.npy')
product_features_path = os.path.join(volume_dir, 'product_features.npy')

products_id_to_idx_path = os.path.join(volume_dir, 'products_id_to_idx.pkl')
# query_pca_path = os.path.join(volume_dir, 'query_pca.pkl')