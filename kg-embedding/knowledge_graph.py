from __future__ import absolute_import, division, print_function

import os
import sys
import argparse
from math import log
from tqdm import tqdm
from copy import deepcopy
import numpy as np
import gzip
import pickle
import random
from datetime import datetime
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kg_utils import *
from data_utils import AmazonDataset


class KnowledgeGraph(object):

    def __init__(self, dataset):
        self.G = dict()
        self._load_entities(dataset)
        self._load_reviews(dataset)
        self._load_knowledge(dataset)
        self._clean()
        self.top_matches = None

    def _load_entities(self, dataset):
        print('Load entities...')
        num_nodes = 0
        for entity in get_entities():
            self.G[entity] = {}
            vocab_size = getattr(dataset, entity).vocab_size
            for eid in range(vocab_size):
                self.G[entity][eid] = {r: [] for r in get_relations(entity)}
            num_nodes += vocab_size
        print('Total {:d} nodes.'.format(num_nodes))

    def _load_reviews(self, dataset, word_tfidf_threshold=0.1, word_freq_threshold=5000):
        print('Load reviews...')
        # (1) Filter words by both tfidf and frequency.
        vocab = dataset.word.vocab
        reviews = [d[2] for d in dataset.review.data]
        review_tfidf = compute_tfidf_fast(vocab, reviews)
        distrib = dataset.review.word_distrib

        num_edges = 0
        all_removed_words = []
        for rid, data in enumerate(dataset.review.data):
            uid, pid, review = data
            doc_tfidf = review_tfidf[rid].toarray()[0]
            remained_words = [wid for wid in set(review)
                              if doc_tfidf[wid] >= word_tfidf_threshold
                              and distrib[wid] <= word_freq_threshold]
            removed_words = set(review).difference(remained_words)  # only for visualize
            removed_words = [vocab[wid] for wid in removed_words]
            all_removed_words.append(removed_words)
            if len(remained_words) <= 0:
                continue

            # (2) Add edges.
            self._add_edge(USER, uid, PURCHASE, PRODUCT, pid)
            num_edges += 2
            for wid in remained_words:
                self._add_edge(USER, uid, MENTION, WORD, wid)
                self._add_edge(PRODUCT, pid, DESCRIBED_AS, WORD, wid)
                num_edges += 4
        print('Total {:d} review edges.'.format(num_edges))

        with open('./tmp/review_removed_words.txt', 'w') as f:
            f.writelines([' '.join(words) + '\n' for words in all_removed_words])

    def _load_knowledge(self, dataset):
        for relation in [PRODUCED_BY, BELONG_TO, ALSO_BOUGHT, ALSO_VIEWED, BOUGHT_TOGETHER]:
            print('Load knowledge {}...'.format(relation))
            data = getattr(dataset, relation).data
            num_edges = 0
            for pid, eids in enumerate(data):
                if len(eids) <= 0:
                    continue
                for eid in set(eids):
                    et_type = get_entity_tail(PRODUCT, relation)
                    self._add_edge(PRODUCT, pid, relation, et_type, eid)
                    num_edges += 2
            print('Total {:d} {:s} edges.'.format(num_edges, relation))

    def _add_edge(self, etype1, eid1, relation, etype2, eid2):
        self.G[etype1][eid1][relation].append(eid2)
        self.G[etype2][eid2][relation].append(eid1)

    def _clean(self):
        print('Remove duplicates...')
        for etype in self.G:
            for eid in self.G[etype]:
                for r in self.G[etype][eid]:
                    data = self.G[etype][eid][r]
                    data = tuple(sorted(set(data)))
                    self.G[etype][eid][r] = data

    def compute_degrees(self):
        print('Compute node degrees...')
        self.degrees = {}
        self.max_degree = {}
        for etype in self.G:
            self.degrees[etype] = {}
            for eid in self.G[etype]:
                count = 0
                for r in self.G[etype][eid]:
                    count += len(self.G[etype][eid][r])
                self.degrees[etype][eid] = count

    def plot_degrees(self):
        all_degrees = 0.
        all_count = 0
        for etype in self.G:
            print(etype)
            data = [self.degrees[etype][eid] for eid in self.degrees[etype]]
            data = sorted(data, reverse=True)
            all_degrees += np.sum(data)
            all_count += len(data)
            num_ignore = int(0.005 * len(data))
            print(num_ignore, data[num_ignore])
            # print(data[:num_ignore])
            # plt.hist(data)
            # plt.title(etype)
            # plt.show()
        print('average degree:', all_degrees / all_count)

    def get(self, eh_type, eh_id=None, relation=None):
        data = self.G
        if eh_type is not None:
            data = data[eh_type]
        if eh_id is not None:
            data = data[eh_id]
        if relation is not None:
            data = data[relation]
        return data

    def __call__(self, eh_type, eh_id=None, relation=None):
        return self.get(eh_type, eh_id, relation)

    def get_tails(self, entity_type, entity_id, relation):
        return self.G[entity_type][entity_id][relation]

    def get_tails_given_user(self, entity_type, entity_id, relation, user_id):
        """ Very important!
        :param entity_type:
        :param entity_id:
        :param relation:
        :param user_id:
        :return:
        """
        tail_type = KG_RELATION[entity_type][relation]
        tail_ids = self.G[entity_type][entity_id][relation]
        if tail_type not in self.top_matches:
            return tail_ids
        top_match_set = set(self.top_matches[tail_type][user_id])
        top_k = len(top_match_set)
        if len(tail_ids) > top_k:
            tail_ids = top_match_set.intersection(tail_ids)
        return list(tail_ids)

    def trim_edges(self):
        degrees = {}
        for entity in self.G:
            degrees[entity] = {}
            for eid in self.G[entity]:
                for r in self.G[entity][eid]:
                    if r not in degrees[entity]:
                        degrees[entity][r] = []
                    degrees[entity][r].append(len(self.G[entity][eid][r]))

        for entity in degrees:
            for r in degrees[entity]:
                tmp = sorted(degrees[entity][r], reverse=True)
                print(entity, r, tmp[:10])

    def set_top_matches(self, u_u_match, u_p_match, u_w_match):
        self.top_matches = {
            USER: u_u_match,
            PRODUCT: u_p_match,
            WORD: u_w_match,
        }

    def heuristic_search(self, uid, pid, pattern_id, trim_edges=False):
        if trim_edges and self.top_matches is None:
            raise Exception('To enable edge-trimming, must set top_matches of users first!')
        if trim_edges:
            _get = lambda e, i, r: self.get_tails_given_user(e, i, r, uid)
        else:
            _get = lambda e, i, r: self.get_tails(e, i, r)

        pattern = PATH_PATTERN[pattern_id]
        paths = []
        if pattern_id == 1:  # OK
            wids_u = set(_get(USER, uid, MENTION))  # USER->MENTION->WORD
            wids_p = set(_get(PRODUCT, pid, DESCRIBED_AS))  # PRODUCT->DESCRIBE->WORD
            intersect_nodes = wids_u.intersection(wids_p)
            paths = [(uid, x, pid) for x in intersect_nodes]
        elif pattern_id in [11, 12, 13, 14, 15, 16, 17]:
            pids_u = set(_get(USER, uid, PURCHASE))  # USER->PURCHASE->PRODUCT
            pids_u = pids_u.difference([pid])  # exclude target product
            nodes_p = set(_get(PRODUCT, pid, pattern[3][0]))  # PRODUCT->relation->node2
            if pattern[2][1] == USER:
                nodes_p.difference([uid])
            for pid_u in pids_u:
                relation, entity_tail = pattern[2][0], pattern[2][1]
                et_ids = set(_get(PRODUCT, pid_u, relation))  # USER->PURCHASE->PRODUCT->relation->node2
                intersect_nodes = et_ids.intersection(nodes_p)
                tmp_paths = [(uid, pid_u, x, pid) for x in intersect_nodes]
                paths.extend(tmp_paths)
        elif pattern_id == 18:
            wids_u = set(_get(USER, uid, MENTION))  # USER->MENTION->WORD
            uids_p = set(_get(PRODUCT, pid, PURCHASE))  # PRODUCT->PURCHASE->USER
            uids_p = uids_p.difference([uid])  # exclude source user
            for uid_p in uids_p:
                wids_u_p = set(_get(USER, uid_p, MENTION))  # PRODUCT->PURCHASE->USER->MENTION->WORD
                intersect_nodes = wids_u.intersection(wids_u_p)
                tmp_paths = [(uid, x, uid_p, pid) for x in intersect_nodes]
                paths.extend(tmp_paths)
        # elif len(pattern) == 5:  # DOES NOT WORK SO FAR!
        #    nodes_from_user = set(self.G[USER][uid][pattern[1][0]])  # USER->MENTION->WORD
        #    nodes_from_product = set(self.G[PRODUCT][pid][pattern[-1][0]])
        #    if pattern[-2][1] == USER:
        #        nodes_from_product.difference([uid])
        #    count = 0
        #    for wid in nodes_from_user:
        #        pids_from_wid = set(self.G[WORD][wid][pattern[2][0]])  # USER->MENTION->WORD->DESCRIBE->PRODUCT
        #        pids_from_wid = pids_from_wid.difference([pid])  # exclude target product
        #        for nid in nodes_from_product:
        #            if pattern[-2][1] == WORD:
        #                if nid == wid:
        #                    continue
        #            other_pids = set(self.G[pattern[-2][1]][nid][pattern[-2][0]])
        #            intersect_nodes = pids_from_wid.intersection(other_pids)
        #            count += len(intersect_nodes)
        #    return count

        return paths


def generate_embeddings(dataset_str, hop, use_describe=True):
    """Note that last entity embedding is of size [vocab_size+1, d]."""
    print('Load embeddings...')
    state_dict = load_embed_model(dataset_str, hop)
    if use_describe:
        describe_rel = 'describe_as'
    else:
        describe_rel = 'mentions'
    print(state_dict.keys())
    embeds = {
        USER: state_dict['user.weight'].cpu().data.numpy()[:-1],  # Must remove last dummy 'user' with 0 embed.
        PRODUCT: state_dict['product.weight'].cpu().data.numpy()[:-1],
        WORD: state_dict['word.weight'].cpu().data.numpy()[:-1],
        BRAND: state_dict['brand.weight'].cpu().data.numpy()[:-1],
        CATEGORY: state_dict['category.weight'].cpu().data.numpy()[:-1],
        RPRODUCT: state_dict['related_product.weight'].cpu().data.numpy()[:-1],

        PURCHASE: (
            state_dict['purchase'].cpu().data.numpy()[0],
            state_dict['purchase_bias.weight'].cpu().data.numpy()
        ),
        MENTION: (
            state_dict['mentions'].cpu().data.numpy()[0],
            state_dict['mentions_bias.weight'].cpu().data.numpy()
        ),
        DESCRIBED_AS: (
            state_dict[describe_rel].cpu().data.numpy()[0],
            state_dict[describe_rel + '_bias.weight'].cpu().data.numpy()
        ),
        PRODUCED_BY: (
            state_dict['produced_by'].cpu().data.numpy()[0],
            state_dict['produced_by_bias.weight'].cpu().data.numpy()
        ),
        BELONG_TO: (
            state_dict['belongs_to'].cpu().data.numpy()[0],
            state_dict['belongs_to_bias.weight'].cpu().data.numpy()
        ),
        ALSO_BOUGHT: (
            state_dict['also_bought'].cpu().data.numpy()[0],
            state_dict['also_bought_bias.weight'].cpu().data.numpy()
        ),
        ALSO_VIEWED: (
            state_dict['also_viewed'].cpu().data.numpy()[0],
            state_dict['also_viewed_bias.weight'].cpu().data.numpy()
        ),
        BOUGHT_TOGETHER: (
            state_dict['bought_together'].cpu().data.numpy()[0],
            state_dict['bought_together_bias.weight'].cpu().data.numpy()
        ),
    }
    save_embed(dataset_str, hop, embeds)


'''
def compute_heuristic_scores(dataset_str, top=1000):
    """Compute top k matches of users/products/words compared to users.
    Each computes a matrix of size [n, k], where n is number of users and each element is an int ID.
    These matrix is used to filter edges.
    """
    print('Compute heuristic scores...')
    embeds = load_embed(dataset_str)
    user_embed = embeds[USER]
    product_embed = embeds[PRODUCT]
    word_embed = embeds[WORD]
    purchase_embed, purchase_bias = embeds[PURCHASE]
    mention_embed, mention_bias = embeds[MENTION]

    # Compute user-user matrix
    t1 = datetime.now()
    u_u_scores = np.matmul(user_embed, user_embed.T)
    u_u_top_matches = np.argsort(u_u_scores, axis=1)
    u_u_top_matches = u_u_top_matches[::, -top:]
    save_top_matches(dataset_str, u_u_top_matches, 'u_u')
    t2 = datetime.now()
    print(u_u_top_matches.shape, (t2 - t1).total_seconds())

    # Compute user-product matrix
    t1 = datetime.now()
    u_p_scores = np.matmul(user_embed + purchase_embed, product_embed.T)
    u_p_top_matches = np.argsort(u_p_scores, axis=1)
    u_p_top_matches = u_p_top_matches[::, -top:]
    save_top_matches(dataset_str, u_p_top_matches, 'u_p')
    t2 = datetime.now()
    print(u_p_top_matches.shape, (t2 - t1).total_seconds())

    # Compute user-word matrix
    t1 = datetime.now()
    u_w_scores = np.matmul(user_embed + mention_embed, word_embed.T)
    u_w_top_matches = np.argsort(u_w_scores, axis=1)
    u_w_top_matches = u_w_top_matches[::, -top:]
    save_top_matches(dataset_str, u_w_top_matches, 'u_w')
    t2 = datetime.now()
    print(u_w_top_matches.shape, (t2 - t1).total_seconds())
'''

'''
def compute_topk_user_products(dataset_str, topk=100):
    """Compute topk user-products in ascending order. (smallest to largest)"""
    embeds = load_embed(dataset_str)
    user_embed = embeds[USER]
    product_embed = embeds[PRODUCT]
    purchase_embed, purchase_bias = embeds[PURCHASE]
    t1 = datetime.now()
    u_p_scores = np.matmul(user_embed + purchase_embed, product_embed.T)
    print(u_p_scores.shape)
    u_p_top_matches = np.argsort(u_p_scores, axis=1)
    u_p_top_matches = u_p_top_matches[:, -topk:]
    print(u_p_scores[0][u_p_top_matches[0]])
    save_top_matches(dataset_str, u_p_top_matches, 'u_p')
    t2 = datetime.now()
    print(u_p_top_matches.shape, (t2 - t1).total_seconds())
    
    
def generate_paths(dataset_str, kg):
    """Generate paths for top 10 user-product pairs.
    Path length of 3 is enough.
    """
    u_p_matches = load_top_matches(dataset_str, 'u_p')
    for pattern_id in [1, 11, 12, 13, 14, 15, 16, 17, 18]:
        pattern = PATH_PATTERN[pattern_id]
        print('Generate path', pattern)
        paths = []
        counts = []
        for uid in kg.get(USER):
            for pid in u_p_matches[uid][-10:]:
                tmp_paths = kg.heuristic_search(uid, pid, pattern_id)
                paths.extend(tmp_paths)
                counts.append(len(tmp_paths))
        print(np.mean(counts), np.max(counts))
        save_paths(dataset_str, pattern_id, paths)
'''

def check_test_path(dataset_str, kg):
    # Check if there exists at least one path for any user-product in test set.
    test_user_products = load_labels(dataset_str, 'test')
    for uid in test_user_products:
        for pid in test_user_products[uid]:
            count = 0
            for pattern_id in [1, 11, 12, 13, 14, 15, 16, 17, 18]:
                tmp_path = kg.heuristic_search(uid, pid, pattern_id)
                count += len(tmp_path)
            if count == 0:
                print(uid, pid)


def main(args):
    # Run following codes for the first time!
    ################## BEGIN ##################
    dataset = load_dataset(args.dataset)
    kg = KnowledgeGraph(dataset)
    kg.compute_degrees()
    save_kg(args.dataset, kg)
    #generate_embeddings(args.dataset, args.hop, use_describe=False)
    ################### END ###################

    #compute_topk_user_products(dataset_str, topk=100)

    #kg = load_kg(dataset_str)
    # kg.plot_degrees()
    # kg.trim_edges()

    # Run following codes to generate paths.
    #generate_paths(dataset_str, kg)
    # check_test_path(dataset_str, kg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cd', help='One of {clothing, cell, beauty, cd}')
    parser.add_argument('--hop', type=int, default=1, help='embedding hop')
    args = parser.parse_args()
    main(args)
