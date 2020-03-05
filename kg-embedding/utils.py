from __future__ import absolute_import, division, print_function

import math
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfTransformer
from deprecated import deprecated


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

DATA_DIR = dict(
        movies=('./data/CIKM2017/reviews_Movies_and_TV_5.json.gz.stem.nostop/min_count5',
                './tmp/Moives_TV'),
        cds=('./data/CIKM2017/reviews_CDs_and_Vinyl_5.json.gz.stem.nostop/min_count5',
             './tmp/CDs_Vinyl'),
        beauty=('./data/CIKM2017/reviews_Beauty_5.json.gz.stem.nostop/min_count5',
                './tmp/Beauty'),
        cell=('./data/CIKM2017/reviews_Cell_Phones_and_Accessories_5.json.gz.stem.nostop/min_count5',
              './tmp/Cellphones_Accessories'),
        clothing=('./data/CIKM2017/reviews_Clothing_Shoes_and_Jewelry_5.json.gz.stem.nostop/min_count5',
                  './tmp/Clothing')
)


def _create_freq_dict(docs):
    freq_dict_list = []
    i = 0
    for doc in docs:
        i += 1
        freq_dict = {}
        for word in doc:
            if word in freq_dict:
                freq_dict[word] += 1
            else:
                freq_dict[word] = 1
        temp = {'doc_id': i, 'freq_dict': freq_dict, 'freq_dict_len': len(doc)}
        freq_dict_list.append(temp)
    return freq_dict_list


def _compute_tf(freq_dict_list):
    tf_scores = []
    for temp_dict in freq_dict_list:
        temp_dict['tf_score'] = {}
        for word, freq in temp_dict['freq_dict'].items():
            freq = freq / temp_dict['freq_dict_len']
            temp_dict['tf_score'][word] = freq
        tf_scores.append(temp_dict)
    return tf_scores


def _compute_idf(freq_dict_list, docs):
    idf_scores = []
    for temp_dict in freq_dict_list:
        idf_dict = {}
        for k in temp_dict['freq_dict'].keys():
            count = 0
            # count = sum([k in tempDict['freq_dict'] for tempDict in freqDict_list])
            for temp_dict in freq_dict_list:
                if k in temp_dict['freq_dict']:
                    count += 1
            idf_dict[k] = math.log(len(docs) / (count + 1))
        idf_scores.append(idf_dict)
    return idf_scores

@deprecated(reason='Use compute_tfidf_fast instead.')
def compute_tfidf(docs):
    """Compute TFIDF scores for all vocabs.

    Args:
        docs: list of list of integers, e.g. [[0,0,1], [1,2,0,1]]

    Returns:
        list, each entry is a dict (representing a doc) with key=token_id, value=tfidf
    """
    freq_dict_list = _create_freq_dict(docs)
    tf_scores = _compute_tf(freq_dict_list)
    idf_scores = _compute_idf(freq_dict_list, docs)
    tfidf_scores = []
    for (i, j) in zip(tf_scores, idf_scores):
        doc_tfidf = {}
        # for j in idf_scores:
        for key in i['tf_score'].keys():
            tfidf = i['tf_score'][key] * j[key]
            doc_tfidf[key] = tfidf
        tfidf_scores.append(doc_tfidf)
    return tfidf_scores




def compute_pagerank(vocab, docs):
    """Compute PageRank scores for all vocabs.

    Args:
        vocab: list of strings, e.g. ['AAA', 'BBB', 'CCC'].
        docs: list of list of integers, e.g. [[0,0,1], [1,2,0,1]]

    Returns:
        list of scores of the same size as vocab.
    """
    pass


def test_compute_tfidf():
    vocab = ['AAA', 'BBB', 'CCC']
    docs = [[0, 0, 1], [1, 2, 0, 1], [1, 1]]
    tfidf = compute_tfidf(docs)
    tfidf = compute_tfidf_fast(vocab, docs)
    print(tfidf)


def test_compute_pagerank():
    pass


if __name__ == '__main__':
    test_compute_tfidf()
    test_compute_pagerank()
