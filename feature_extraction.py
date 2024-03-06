import json
import gc
import pickle
import properties
from collections import Counter

from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from tqdm import tqdm
import numpy as np
import pandas as pd


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


# Load aggregated search data which will be used as training data.
aggregated_searches_df = pd.DataFrame(read_json_lines(properties.aggregated_search_data_path, n_lines=properties.NUM_TRAIN_SAMPLES))
# Load preprocessed product data.
# products_data_df = pd.DataFrame(read_json_lines(properties.preprocessed_products_path))

queries_train_projected = np.load(properties.queries_train_features_path)
# products_name_projected = np.load(properties.product_name_features_path)
product_vector = np.load(properties.product_features_path)
# products_category_projected = np.load(product_category_features_path)

# Fit PCA model for queries
# query_pca = PCA(n_components=64)
# queries_train_projected = query_pca.fit_transform(queries_train_projected)
# with open(query_pca_path, 'wb') as f:
#     pickle.dump(query_pca, f)

with open(properties.products_id_to_idx_path, 'rb') as f:
    products_id_to_idx = pickle.load(f)

aggregated_searches_train_df, aggregated_searches_validation_df = train_test_split(aggregated_searches_df, test_size=0.04, random_state=42)
queries_train_projected, queries_validation_projected = train_test_split(queries_train_projected, test_size=0.04, random_state=42)

aggregated_searches_train_df = aggregated_searches_train_df.reset_index()
remove_index = aggregated_searches_train_df[aggregated_searches_train_df['query_count'] == 1].index
aggregated_searches_train_df = aggregated_searches_train_df[~aggregated_searches_train_df.index.isin(remove_index)]
queries_train_projected = np.delete(queries_train_projected, remove_index.values, axis=0)

# ss = StandardScaler()
# le = LabelEncoder()
# mm = MinMaxScaler()

# Fit PCA model for product names
# product_pca = PCA(n_components=64)
# products_name_projected = product_pca.fit_transform(products_name_projected)

# product_vector = np.concatenate((products_name_projected, ss.fit_transform(products_data_df.drop(['id', 'title_normalized', 'category_name'], axis=1).values)), axis=1)
# product_vector = np.nan_to_num(product_vector, nan=0)
# np.save(properties.product_features_path, product_vector)
# exit()

def create_dat_file(
    dat_file_path,
    agg_searches_df,
    query_features,
    product_features,
    n_candidates=None,
    train=True,
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
            # if len(agg_search.clicks_count) <= 10 and all(i <= 4 for i in agg_search.clicks_count):
            # if len(agg_search.clicks_count) == 1:
            # if all(i <= 7 for i in agg_search.clicks_count) and train:
                # continue
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

                # similarity = cosine(product_features[p_idx][:-6], query_features[qid])
                features = np.concatenate((product_features[p_idx], query_features[qid]))
                features = np.around(features, 3)

                # new_str = f"{candidate_score} qid:{qid} " + " ".join([f"{i}:{s}" for i, s in enumerate(features)]) + "\n"

                file.write(
                    f"{candidate_score} qid:{qid} "
                    + " ".join([f"{i}:{s}" for i, s in enumerate(features)])
                    + "\n"
                )


create_dat_file(
    properties.train_dat_file_path,
    aggregated_searches_train_df,
    queries_train_projected,
    product_vector,
    n_candidates=200,
)

create_dat_file(
    properties.validation_dat_file_path,
    aggregated_searches_validation_df,
    queries_validation_projected,
    product_vector,
    n_candidates=200,
    train=False
)

# create_dat_file(
#     properties.train_dat_file_path,
#     aggregated_searches_df,
#     queries_train_projected,
#     product_vector,
#     n_candidates=200,
# )


# splits = 30
# counter = 1
# for i, j in zip(np.array_split(aggregated_searches_df, splits), np.array_split(queries_train_projected, splits)):
#     create_dat_file(
#         properties.train_dat_file_path + str(counter),
#         i,
#         j,
#         product_vector,
#         n_candidates=200,
#     )
#     counter += 1