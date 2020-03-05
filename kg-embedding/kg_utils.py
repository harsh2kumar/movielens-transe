from __future__ import absolute_import, division, print_function

import sys
import os
import pickle
import random
import logging
import logging.handlers
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfTransformer
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EPS = np.finfo(np.float32).eps.item()

DATA_DIR = dict(
        beauty='./tmp/Beauty',
        cell='./tmp/Cellphones_Accessories',
        clothing='./tmp/Clothing',
        cd='./tmp/CDs_Vinyl',
)

# Embedding model file
MODEL_FILE = {
    1: {
        'beauty': DATA_DIR['beauty'] + '/train_embedding_final/embedding_des_epoch_29.ckpt',
        # beauty=DATA_DIR['beauty'] + '/train_embedding_011519/embedding_epoch_30.ckpt',
        'cell': DATA_DIR['cell'] + '/train_embedding_final/embedding_des_epoch_30.ckpt',
        # 'cell': DATA_DIR['cell'] + '/train_embedding_011119/embedding_epoch_30.ckpt',
        'clothing': DATA_DIR['clothing'] + '/train_embedding_final/embedding_des_epoch_29.ckpt',
        # 'clothing': DATA_DIR['clothing'] + '/train_embedding_best1204/embedding_epoch_20.ckpt',
        'cd': DATA_DIR['cd'] + '/train_embedding/embedding_epoch_20.ckpt'
    },
    2: {
        'beauty': DATA_DIR['beauty'] + '/train_embedding_final/embedding_2hop_des_epoch_29.ckpt',
        'clothing': DATA_DIR['clothing'] + '/train_embedding_final/embedding_2hop_des_epoch_29.ckpt',
    }
}

LABELS = dict(
        beauty=(DATA_DIR['beauty'] + '/Beauty_train_label.pkl',
                DATA_DIR['beauty'] + '/Beauty_test_label.pkl'),
        clothing=(DATA_DIR['clothing'] + '/Clothing_train_label.pkl',
                  DATA_DIR['clothing'] + '/Clothing_test_label.pkl'),
        cell=(DATA_DIR['cell'] + '/Cellphones_Accessories_train_label.pkl',
              DATA_DIR['cell'] + '/Cellphones_Accessories_test_label.pkl'),
        cd=(DATA_DIR['cd'] + '/CDs_Vinyl_train_label.pkl',
            DATA_DIR['cd'] + '/CDs_Vinyl_test_label.pkl')
)


def load_dataset(dataset):
    dataset_file = DATA_DIR[dataset] + '/dataset.pkl'
    dataset = pickle.load(open(dataset_file, 'rb'))
    return dataset


def load_kg(dataset):
    kg_file = DATA_DIR[dataset] + '/kg.pkl'
    kg = pickle.load(open(kg_file, 'rb'))
    return kg


def save_kg(dataset, kg):
    kg_file = DATA_DIR[dataset] + '/kg.pkl'
    pickle.dump(kg, open(kg_file, 'wb'))


def load_embed_model(dataset, hop):
    model_file = MODEL_FILE[hop][dataset]
    state_dict = torch.load(model_file, map_location=lambda storage, loc: storage)
    return state_dict


def load_embed(dataset, hop):
    embed_file = '{}/embed_hop{}.pkl'.format(DATA_DIR[dataset], hop)
    print('Load embedding:', embed_file)
    embed = pickle.load(open(embed_file, 'rb'))
    return embed


def save_embed(dataset, hop, embed):
    embed_file = '{}/embed_hop{}.pkl'.format(DATA_DIR[dataset], hop)
    pickle.dump(embed, open(embed_file, 'wb'))


'''
def load_top_matches(dataset, match_type):
    assert match_type in ['u_u', 'u_p', 'u_w']
    match_file = '{}/{}_matches.npy'.format(DATA_DIR[dataset], match_type)
    top_matches = np.load(match_file)
    return top_matches


def save_top_matches(dataset, top_matches, match_type):
    assert match_type in ['u_u', 'u_p', 'u_w']
    match_file = '{}/{}_matches.npy'.format(DATA_DIR[dataset], match_type)
    np.save(match_file, top_matches)
'''


def load_paths(dataset, pattern_id):
    path_file = DATA_DIR[dataset] + '/paths/{}.pkl'.format(pattern_id)
    paths = pickle.load(open(path_file, 'rb'))
    return paths


def save_paths(dataset, pattern_id, paths):
    root_dir = DATA_DIR[dataset] + '/paths'
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    path_file = '{}/{}.pkl'.format(root_dir, pattern_id)
    pickle.dump(paths, open(path_file, 'wb'))


def load_labels(dataset, mode='train'):
    if mode == 'train':
        label_file = LABELS[dataset][0]
    elif mode == 'test':
        label_file = LABELS[dataset][1]
    else:
        raise Exception('mode should be one of {train, test}.')
    user_products = pickle.load(open(label_file, 'rb'))
    return user_products


USER = 'user'
PRODUCT = 'product'
WORD = 'word'
RPRODUCT = 'related_product'
BRAND = 'brand'
CATEGORY = 'category'

PURCHASE = 'purchase'
MENTION = 'mentions'
DESCRIBED_AS = 'described_as'
PRODUCED_BY = 'produced_by'
BELONG_TO = 'belongs_to'
ALSO_BOUGHT = 'also_bought'
ALSO_VIEWED = 'also_viewed'
BOUGHT_TOGETHER = 'bought_together'

SELF_LOOP = 'self_loop'  # only for kg env

KG_RELATION = {
    USER: {
        PURCHASE: PRODUCT,
        MENTION: WORD,
    },
    WORD: {
        MENTION: USER,
        DESCRIBED_AS: PRODUCT,
    },
    PRODUCT: {
        PURCHASE: USER,
        DESCRIBED_AS: WORD,
        PRODUCED_BY: BRAND,
        BELONG_TO: CATEGORY,
        ALSO_BOUGHT: RPRODUCT,
        ALSO_VIEWED: RPRODUCT,
        BOUGHT_TOGETHER: RPRODUCT,
    },
    BRAND: {
        PRODUCED_BY: PRODUCT,
    },
    CATEGORY: {
        BELONG_TO: PRODUCT,
    },
    RPRODUCT: {
        ALSO_BOUGHT: PRODUCT,
        ALSO_VIEWED: PRODUCT,
        BOUGHT_TOGETHER: PRODUCT,
    }
}

PATH_PATTERN = {
    # length = 3
    1: ((None, USER), (MENTION, WORD), (DESCRIBED_AS, PRODUCT)),
    # length = 4
    11: ((None, USER), (PURCHASE, PRODUCT), (PURCHASE, USER), (PURCHASE, PRODUCT)),
    12: ((None, USER), (PURCHASE, PRODUCT), (DESCRIBED_AS, WORD), (DESCRIBED_AS, PRODUCT)),
    13: ((None, USER), (PURCHASE, PRODUCT), (PRODUCED_BY, BRAND), (PRODUCED_BY, PRODUCT)),
    14: ((None, USER), (PURCHASE, PRODUCT), (BELONG_TO, CATEGORY), (BELONG_TO, PRODUCT)),
    15: ((None, USER), (PURCHASE, PRODUCT), (ALSO_BOUGHT, RPRODUCT), (ALSO_BOUGHT, PRODUCT)),
    16: ((None, USER), (PURCHASE, PRODUCT), (ALSO_VIEWED, RPRODUCT), (ALSO_VIEWED, PRODUCT)),
    17: ((None, USER), (PURCHASE, PRODUCT), (BOUGHT_TOGETHER, RPRODUCT), (BOUGHT_TOGETHER, PRODUCT)),
    18: ((None, USER), (MENTION, WORD), (MENTION, USER), (PURCHASE, PRODUCT)),
    # length = 5
    101: ((None, USER), (MENTION, WORD), (DESCRIBED_AS, PRODUCT), (PURCHASE, USER), (PURCHASE, PRODUCT)),
    102: ((None, USER), (MENTION, WORD), (DESCRIBED_AS, PRODUCT), (DESCRIBED_AS, WORD), (DESCRIBED_AS, PRODUCT)),
    103: ((None, USER), (MENTION, WORD), (DESCRIBED_AS, PRODUCT), (PRODUCED_BY, BRAND), (PRODUCED_BY, PRODUCT)),
    104: ((None, USER), (MENTION, WORD), (DESCRIBED_AS, PRODUCT), (BELONG_TO, CATEGORY), (BELONG_TO, PRODUCT)),
    105: ((None, USER), (MENTION, WORD), (DESCRIBED_AS, PRODUCT), (ALSO_BOUGHT, RPRODUCT), (ALSO_BOUGHT, PRODUCT)),
    106: ((None, USER), (MENTION, WORD), (DESCRIBED_AS, PRODUCT), (ALSO_VIEWED, RPRODUCT), (ALSO_VIEWED, PRODUCT)),
    107: (
        (None, USER), (MENTION, WORD), (DESCRIBED_AS, PRODUCT), (BOUGHT_TOGETHER, RPRODUCT),
        (BOUGHT_TOGETHER, PRODUCT)),
    108: ((None, USER), (MENTION, WORD), (MENTION, USER), (MENTION, WORD), (DESCRIBED_AS, PRODUCT)),
}


def get_entities():
    return list(KG_RELATION.keys())


def get_relations(entity_head):
    return list(KG_RELATION[entity_head].keys())


def get_entity_tail(entity_head, relation):
    return KG_RELATION[entity_head][relation]


def compute_tfidf_fast(vocab, docs):
    """Compute TFIDF scores for all vocabs.

    Args:
        docs: list of list of integers, e.g. [[0,0,1], [1,2,0,1]]

    Returns:
        sp.csr_matrix, [num_docs, num_vocab]
    """
    # (1) Compute term frequency in each doc.
    data, indices, indptr = [], [], [0]
    for d in docs:
        term_count = {}
        for term_idx in d:
            if term_idx not in term_count:
                term_count[term_idx] = 1
            else:
                term_count[term_idx] += 1
        indices.extend(term_count.keys())
        data.extend(term_count.values())
        indptr.append(len(indices))
    tf = sp.csr_matrix((data, indices, indptr), dtype=int, shape=(len(docs), len(vocab)))

    # (2) Compute normalized tfidf for each term/doc.
    transformer = TfidfTransformer(smooth_idf=True)
    tfidf = transformer.fit_transform(tf)
    return tfidf


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_logger(logname):
    logger = logging.getLogger(logname)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(levelname)s]  %(message)s')
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    fh = logging.handlers.RotatingFileHandler(logname, mode='w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


if __name__ == '__main__':
    print(get_entity_tail(USER, PURCHASE))
    print(get_relations(USER))
