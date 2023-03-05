import os
import json
import gc
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
#import fasttext

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


output_dir = os.path.join('output_data')
volume_dir = ('/mnt/h/torob_data/')

aggregated_search_data_path = os.path.join(output_dir, 'aggregated_search_data.jsonl')
preprocessed_products_path = os.path.join(output_dir, 'preprocessed_products.jsonl')
preprocessed_test_queries_path = os.path.join(output_dir, 'preprocessed_test_queries.jsonl')

train_dat_file_path = os.path.join(volume_dir, 'train.dat')
validation_dat_file_path = os.path.join(volume_dir, 'validation.dat')

# random_projection_mat_path = os.path.join(output_dir, 'random_projection_mat.npy')
product_features_path = os.path.join(volume_dir, 'product_features.npy')
queries_train_features_path = os.path.join(volume_dir, 'queries_train_features.npy')
queries_test_features_path = os.path.join(volume_dir, 'queries_test_features.npy')
products_id_to_idx_path = os.path.join(volume_dir, 'products_id_to_idx.pkl')

#-----TF-IDF not used
# Number of tokens in the vocabulary of TF-IDF.
#VOCAB_SIZE = 4096
# Embedding dimension used for random projection of TF-IDF vectors.
#EMBEDDING_DIM = 256
# Number of training samples to use (set to None to use all samples).
NUM_TRAIN_SAMPLES = 10_000

# Load aggregated search data which will be used as training data.
aggregated_searches_df = pd.DataFrame(read_json_lines(aggregated_search_data_path, n_lines=NUM_TRAIN_SAMPLES))
# Load preprocessed product data.
products_data_df = pd.DataFrame(read_json_lines(preprocessed_products_path))
# Load preprocessed test queries.
test_offline_queries_df = pd.DataFrame(read_json_lines(preprocessed_test_queries_path))

# Create a mapping from ID of products to their integer index.
products_id_to_idx = dict((p_id, idx) for idx, p_id in enumerate(products_data_df['id']))

# Load fasttext model and vectorize all text
# ft_model = fasttext.load_model('../models/cc.fa.300.bin')
# queries_test_projected = np.array([ft_model.get_sentence_vector(x) for x in tqdm(test_offline_queries_df['raw_query_normalized'].values)])
# queries_train_projected = np.array([ft_model.get_sentence_vector(x) for x in tqdm(aggregated_searches_df['raw_query_normalized'].values)])
# products_projected = np.array([ft_model.get_sentence_vector(x) for x in tqdm(products_data_df['title_normalized'].values)])

# Since memory is limited, we store all the neccessary data
# such as extracted features on disk. Later, in inference
# step we may need some of these files.
# np.save(product_features_path, products_projected)
# np.save(queries_train_features_path, queries_train_projected)
# np.save(queries_test_features_path, queries_test_projected)
# with open(products_id_to_idx_path, 'wb') as f:
#     pickle.dump(products_id_to_idx, f)

aggregated_searches_train_df, aggregated_searches_validation_df = train_test_split(aggregated_searches_df, test_size=0.2, random_state=42)

queries_test_projected = np.load(queries_test_features_path)
queries_train_projected, queries_validation_projected = train_test_split(np.load(queries_train_features_path), test_size=0.2, random_state=42)
products_projected = np.load(product_features_path)
# products_data_df = products_data_df.fillna(-1)

# pca = PCA(n_components=128)
# test = pca.fit_transform(products_projected)
test = np.concatenate((products_projected, products_data_df.drop(['id', 'title_normalized'], axis=1).values), axis=1)
test = np.nan_to_num(test, nan=-1)
# test = pca.fit_transform(test)

# del ft_model
# gc.collect();


def create_dat_file(
    dat_file_path,
    agg_searches_df,
    query_features,
    product_features,
    n_candidates=None,
):
    """
    Create a `dat` file which is the training data of LambdaMart model.

    The file format of the training and test files is the same as for SVMlight,
    with the exception that the lines in the input files have to be sorted by increasing qid.
    The first lines may contain comments and are ignored if they start with #.
    Each of the following lines represents one training example and is of the following format:

    <line> .=. <target> qid:<qid> <feature>:<value> <feature>:<value> ... <feature>:<value> # <info>
    <target> .=. <float>
    <qid> .=. <positive integer>
    <feature> .=. <positive integer>
    <value> .=. <float>
    <info> .=. <string>

    The target value and each of the feature/value pairs are separated by a space character.
    Feature/value pairs MUST be ordered by increasing feature number.
    Features with value zero can be skipped.
    The target value defines the order of the examples for each query.
    Implicitly, the target values are used to generated pairwise preference constraints as described in [Joachims, 2002c].
    A preference constraint is included for all pairs of examples in the example_file, for which the target value differs.
    The special feature "qid" can be used to restrict the generation of constraints.
    Two examples are considered for a pairwise preference constraint only if the value of "qid" is the same.

    For example, given the example_file

    3 qid:1 1:1 2:1 3:0 4:0.2 5:0 # 1A
    2 qid:1 1:0 2:0 3:1 4:0.1 5:1 # 1B
    1 qid:1 1:0 2:1 3:0 4:0.4 5:0 # 1C
    1 qid:1 1:0 2:0 3:1 4:0.3 5:0 # 1D
    1 qid:2 1:0 2:0 3:1 4:0.2 5:0 # 2A
    2 qid:2 1:1 2:0 3:1 4:0.4 5:0 # 2B
    1 qid:2 1:0 2:0 3:1 4:0.1 5:0 # 2C
    1 qid:2 1:0 2:0 3:1 4:0.2 5:0 # 2D
    2 qid:3 1:0 2:0 3:1 4:0.1 5:1 # 3A
    3 qid:3 1:1 2:1 3:0 4:0.3 5:0 # 3B
    4 qid:3 1:1 2:0 3:0 4:0.4 5:1 # 3C
    1 qid:3 1:0 2:1 3:1 4:0.5 5:0 # 3D

    the following set of pairwise constraints is generated (examples are referred to by the info-string after the # character):

    1A>1B, 1A>1C, 1A>1D, 1B>1C, 1B>1D, 2B>2A, 2B>2C, 2B>2D, 3C>3A, 3C>3B, 3C>3D, 3B>3A, 3B>3D, 3A>3D

    More information:
     - https://xgboost.readthedocs.io/en/latest/tutorials/input_format.html#embedding-additional-information-inside-libsvm-file
     - https://www.cs.cornell.edu/people/tj/svm_light/svm_rank.html
    """
    with open(dat_file_path, "w") as file:
        for qid, agg_search in tqdm(enumerate(agg_searches_df.itertuples(index=False)), total=len(agg_searches_df)):
            if n_candidates is None:
                limit = len(agg_search.results)
            else:
                limit = min(n_candidates, len(agg_search.results))
            clicks = dict(zip(agg_search.clicks, agg_search.clicks_count))

            for candidate_product_id in agg_search.results[:limit]:
                if candidate_product_id is None:
                    continue
                candidate_score = clicks.get(candidate_product_id, 0)
                candidate_score = np.log2(candidate_score + 1)

                p_idx = products_id_to_idx[candidate_product_id]
                features = np.concatenate((product_features[p_idx], query_features[qid]))
                features = np.around(features, 3)

                # new_str = f"{candidate_score} qid:{qid} " + " ".join([f"{i}:{s}" for i, s in enumerate(features)]) + "\n"

                file.write(
                    f"{candidate_score} qid:{qid} "
                    + " ".join([f"{i}:{s}" for i, s in enumerate(features)])
                    + "\n"
                )


create_dat_file(
    train_dat_file_path,
    aggregated_searches_train_df,
    queries_train_projected,
    test,
    n_candidates=200,
)

create_dat_file(
    validation_dat_file_path,
    aggregated_searches_validation_df,
    queries_validation_projected,
    test,
    n_candidates=200,
)