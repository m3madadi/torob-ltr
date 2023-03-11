import json
import gc
import pickle
import properties

from sklearn.decomposition import PCA
from tqdm import tqdm
import numpy as np
import pandas as pd
import fasttext
import torch
from sentence_transformers import models, SentenceTransformer

def transform_and_normalize(vecs, kernel, bias):
    """
        Applying transformation then standardize
    """
    if not (kernel is None or bias is None):
        vecs = (vecs + bias).dot(kernel)
    return normalize(vecs)
    
def normalize(vecs):
    """
        Standardization
    """
    return vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5
    
def compute_kernel_bias(vecs):
    """
    Calculate Kernal & Bias for the final transformation - y = (x + bias).dot(kernel)
    """
    vecs = np.concatenate(vecs, axis=0)
    mu = vecs.mean(axis=0, keepdims=True)
    cov = np.cov(vecs.T)
    u, s, vh = np.linalg.svd(cov)
    W = np.dot(u, np.diag(s**0.5))
    W = np.linalg.inv(W.T)
    return W, -mu

def dim_reduction(sentences, model):
    '''
        This method will accept array of sentences, roberta tokenizer & model
        next it will call methods for dimention reduction
    '''

    vecs = model.encode(sentences, show_progress_bar=True, batch_size=128)
    # with torch.no_grad():

    #     for sentence in sentences:
    #         inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True,  max_length=64)
    #         inputs['input_ids'] = inputs['input_ids'].to(DEVICE)
    #         inputs['attention_mask'] = inputs['attention_mask'].to(DEVICE)

    #         hidden_states = model(**inputs, return_dict=True, output_hidden_states=True).hidden_states

    #         #Averaging the first & last hidden states
    #         output_hidden_state = (hidden_states[-1] + hidden_states[1]).mean(dim=1)

    #         vec = output_hidden_state.cpu().numpy()[0]

    #         vecs.append(vec)

    
    #Finding Kernal
    kernel, bias = compute_kernel_bias([vecs])
    #If you want to reduce it to 128 dim
    kernel = kernel[:, :properties.EMBEDDING_DIM]
    embeddings = []
    embeddings = np.vstack(vecs)

    #Sentence embeddings can be converted into an identity matrix
    #by utilizing the transformation matrix
    embeddings = transform_and_normalize(embeddings, 
                kernel=kernel,
                bias=bias
            )

    return embeddings


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

def load_st_model(model_name_or_path):
    word_embedding_model = models.Transformer(model_name_or_path)
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode_cls_token  = True,
        pooling_mode_mean_tokens=False,
        pooling_mode_max_tokens=False)

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    model.to(properties.DEVICE)
    return model


# Load aggregated search data which will be used as training data.
aggregated_searches_df = pd.DataFrame(read_json_lines(properties.aggregated_search_data_path, n_lines=properties.NUM_TRAIN_SAMPLES))
# Load preprocessed product data.
products_data_df = pd.DataFrame(read_json_lines(properties.preprocessed_products_path))
# Load preprocessed test queries.
test_offline_queries_df = pd.DataFrame(read_json_lines(properties.preprocessed_test_queries_path))

# Create a mapping from ID of products to their integer index.
products_id_to_idx = dict((p_id, idx) for idx, p_id in enumerate(products_data_df['id']))


embedder = load_st_model('/mnt/d/Models/bert-zwnj-wnli-mean-tokens')

# queries_test_bert_projected = dim_reduction(test_offline_queries_df['raw_query_normalized'].values, embedder)
# np.save(properties.queries_test_features_path, queries_test_bert_projected)

# queries_train_bert_projected = dim_reduction(aggregated_searches_df['raw_query_normalized'].values, embedder)
# np.save(properties.queries_train_features_path, queries_train_bert_projected)

products_name_bert_projected = dim_reduction(products_data_df['title_normalized'].values, embedder)
np.save(properties.product_name_features_path, products_name_bert_projected)

# var = round(len(products_data_df) / 2)
# products_name_bert_projected = dim_reduction(products_data_df['title_normalized'].values[:var], embedder)
# np.save(product_name_features_path, products_name_bert_projected)
# del products_name_bert_projected

# products_name_bert_projected = dim_reduction(products_data_df['title_normalized'].values[var:], embedder)
# test = np.load(product_name_features_path)
# test = np.concatenate((test, products_name_bert_projected), axis=0)
# np.save(product_name_features_path, test)
# del products_name_bert_projected

# gc.collect()
# torch.cuda.empty_cache()


# Load fasttext model and vectorize all text
# ft_model = fasttext.load_model('/mnt/h/models/cc.fa.300.bin')
# queries_test_projected = np.array([ft_model.get_sentence_vector(x) for x in tqdm(test_offline_queries_df['raw_query_normalized'].values)])
# queries_train_projected = np.array([ft_model.get_sentence_vector(x) for x in tqdm(aggregated_searches_df['raw_query_normalized'].values)])
# products_name_projected = np.array([ft_model.get_sentence_vector(x) for x in tqdm(products_data_df['title_normalized'].values)])
# products_category_projected = np.array([ft_model.get_sentence_vector(x) for x in tqdm(products_data_df['category_name'].values)])

# Since memory is limited, we store all the neccessary data
# such as extracted features on disk. Later, in inference
# step we may need some of these files.
# np.save(product_category_features_path, products_category_projected)
# np.save(product_name_features_path, products_name_projected)
# np.save(queries_train_features_path, queries_train_projected)
# np.save(queries_test_features_path, queries_test_projected)
with open(properties.products_id_to_idx_path, 'wb') as f:
    pickle.dump(products_id_to_idx, f)
# exit()