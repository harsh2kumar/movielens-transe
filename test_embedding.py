from __future__ import absolute_import, division, print_function

import os
import sys
import argparse
import random
import numpy as np
import pickle
from easydict import EasyDict as edict
from math import log
import torch
import torch.nn as nn

# from data_utils import *
from models import KnowledgeEmbedding


def load_embedding(args):
  state_dict = torch.load(args.model_file)
  user_embed = state_dict['user.weight'].cpu()
  movie_embed = state_dict['movie.weight'].cpu()
  watched_embed = state_dict['user_watched_movie'].cpu()
  watched_bias = state_dict['user_watched_movie_bias.weight'].cpu()
  # product_embed = state_dict['product.weight'].cpu()
  # purchase_embed = state_dict['purchase'].cpu()
  # purchase_bias = state_dict['purchase_bias.weight'].cpu()
  results = edict(
    user_embed=user_embed.data.numpy(),
    movie_embed=movie_embed.data.numpy(),
    watched_embed=watched_embed.data.numpy(),
    watched_bias=watched_bias.data.numpy(),
    # product_embed=product_embed.data.numpy(),
    # purchase_embed=purchase_embed.data.numpy(),
    # purchase_bias=purchase_bias.data.numpy(),
  )
  output_file = '{}/{}_embedding.pkl'.format(args.dataset_dir, args.dataset)
  with open(output_file, 'wb') as f:
    pickle.dump(results, f)

def load_train_reviews(args):
  user_watched = {}  # {uid: [pid,...], ...}
  with open(args.train_review_file, 'r') as f:
    for line in f:
      line = line.strip()
      arr = line.split('\t')
      user_idx = int(arr[0])
      movie_idx = int(arr[2])
      if user_idx not in user_watched:
        user_watched[user_idx] = []
      user_watched[user_idx].append(movie_idx)
  output_file = '{}/{}_train_label.pkl'.format(args.dataset_dir, args.dataset)
  with open(output_file, 'wb') as f:
    pickle.dump(user_watched, f)

# def load_train_reviews(args):
#     user_products = {}  # {uid: [pid,...], ...}
#     with gzip.open(args.train_review_file, 'r') as f:
#         for line in f:
#             line = line.decode('utf-8').strip()
#             arr = line.split('\t')
#             user_idx = int(arr[0])
#             product_idx = int(arr[1])
#             if user_idx not in user_products:
#                 user_products[user_idx] = []
#             user_products[user_idx].append(product_idx)
#     output_file = '{}/{}_train_label.pkl'.format(args.dataset_dir, args.dataset)
#     with open(output_file, 'wb') as f:
#         pickle.dump(user_products, f)


# def load_test_reviews(args):
#     user_products = {}  # {uid: [pid,...], ...}
#     with open(args.test_review_file, 'r') as f:
#         for line in f:
#             line = line.decode('utf-8').strip()
#             arr = line.split('\t')
#             user_idx = int(arr[0])
#             product_idx = int(arr[1])
#             if user_idx not in user_products:
#                 user_products[user_idx] = []
#             user_products[user_idx].append(product_idx)
#     output_file = '{}/{}_test_label.pkl'.format(args.dataset_dir, args.dataset)
#     with open(output_file, 'wb') as f:
#         pickle.dump(user_products, f)

def load_test_reviews(args):
  user_watched = {}  # {uid: [pid,...], ...}
  with open(args.test_review_file, 'r') as f:
    for line in f:
      line = line.strip()
      arr = line.split('\t')
      user_idx = int(arr[0])
      movie_idx = int(arr[2])
      if user_idx not in user_watched:
        user_watched[user_idx] = []
      user_watched[user_idx].append(movie_idx)
  output_file = '{}/{}_test_label.pkl'.format(args.dataset_dir, args.dataset)
  with open(output_file, 'wb') as f:
    pickle.dump(user_watched, f)

def test(args, topk=10):
  embed_file = '{}/{}_embedding.pkl'.format(args.dataset_dir, args.dataset)
  with open(embed_file, 'rb') as f:
    embeddings = pickle.load(f)

  train_labels_file = '{}/{}_train_label.pkl'.format(args.dataset_dir, args.dataset)
  with open(train_labels_file, 'rb') as f:
    train_user_movie = pickle.load(f)

  test_labels_file = '{}/{}_test_label.pkl'.format(args.dataset_dir, args.dataset)
  with open(test_labels_file, 'rb') as f:
    test_user_movie = pickle.load(f)
  test_user_idxs = list(test_user_movie.keys())
  # print('Num of users:', len(user_idxs))
  # print('User:', user_idxs[0], 'Products:', user_products[user_idxs[0]])

  user_embed = embeddings['user_embed'][:-1]  # remove last dummy user
  watched_embed = embeddings['watched_embed']
  movie_embed = embeddings['movie_embed'][:-1]
  print('user embed:', user_embed.shape, 'movie embed:', movie_embed.shape)

  # calculate user + watched embeddings
  calulated_movie_emb = user_embed + watched_embed
  # normalize embeddings(TBD)
  # calulated_product_emb = calulated_product_emb/LA.norm(calulated_product_emb, axis=1, keepdims=True)
  # calculate Nearest Neighbors
  scores_matrix = np.dot(calulated_movie_emb, movie_embed.T)
  print('Max score:', np.max(scores_matrix))

  # normalize embeddings(TBD)
  # norm_calulated_product_emb = calulated_product_emb/LA.norm(calulated_product_emb, axis=1, keepdims=True)
  # norm_product_embed = product_embed/LA.norm(product_embed, axis=1, keepdims=True)
  # scores_matrix = np.dot(norm_calulated_product_emb, np.transpose(norm_product_embed))
  # print (scores_matrix.shape)

  # filter the test data item which trained in train data
  idx_list = []
  for uid in train_user_movie:
    pids = train_user_movie[uid]
    tmp = list(zip([uid] * len(pids), pids))
    idx_list.extend(tmp)
  idx_list = np.array(idx_list)
  scores_matrix[idx_list[:, 0], idx_list[:, 1]] = -99

  if scores_matrix.shape[1] <= 30000:
    top_matches = np.argsort(scores_matrix)  # sort row by row
    topk_matches = top_matches[:, -topk:]  # user-product matrix, from lowest rank to highest
  else:  # sort in batch way
    topk_matches = np.zeros((scores_matrix.shape[0], topk), dtype=np.int)
    i = 0
    while i < scores_matrix.shape[0]:
      start_row = i
      end_row = np.min([i + 100, scores_matrix.shape[0]])
      batch_scores = scores_matrix[start_row:end_row, :]
      matches = np.argsort(batch_scores)
      topk_matches[start_row:end_row] = matches[:, -topk:]
      i = end_row
  # f = open('results.txt', 'w')
  # Compute metrics
  precisions, recalls, ndcgs, hits = [], [], [], []
  for uid in test_user_idxs:
    pred_list, rel_set = topk_matches[uid][::-1], test_user_movie[uid]

    dcg = 0.0
    hit_num = 0.0
    for i in range(len(pred_list)):
      if pred_list[i] in rel_set:
        dcg += 1. / (log(i + 2) / log(2))
        hit_num += 1
    # idcg
    idcg = 0.0
    for i in range(min(len(rel_set), len(pred_list))):
      idcg += 1. / (log(i + 2) / log(2))
    ndcg = dcg / idcg
    recall = hit_num / len(rel_set)
    precision = hit_num / len(pred_list)
    hit = 1.0 if hit_num > 0.0 else 0.0

    ndcgs.append(ndcg)
    recalls.append(recall)
    precisions.append(precision)
    hits.append(hit)
    # print(uid)

  #   f.write(' '.join(map(str, pred_list)))
  #   f.write('\n')
  #   f.write(' '.join(map(str, rel_set)))
  #   f.write('\n')
  # f.close()
  avg_precision = np.mean(precisions) * 100
  avg_recall = np.mean(recalls) * 100
  avg_ndcg = np.mean(ndcgs) * 100
  avg_hit = np.mean(hits) * 100
  print('NDCG={:.3f} |  Recall={:.3f} | HR={:.3f} | Precision={:.3f}'.format(
          avg_ndcg, avg_recall, avg_hit, avg_precision))
  # print('NDCG={} |  Recall={} | HR={} | Precision={}'.format(
  #   avg_ndcg, avg_recall, avg_hit, avg_precision))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', type=str, default='MovieLens20M_300epochs',
                      help='One of {MovieLens20M}')
  args = parser.parse_args()
  args.dataset_dir = './tmp/{}'.format(args.dataset)
  # args.dataset_file = args.dataset_dir + '/dataset.pkl'
  # print(args.dataset_dir + '/train_embedding/embedding_1hop_des_epoch_77.ckpt')
  model_files = {
    'MovieLens20M': args.dataset_dir + '/train_3hop_des_embedding/embedding_3hop_des_epoch_30.ckpt',
    'MovieLens20M_100epochs': args.dataset_dir + '/train_embedding/embedding_1hop_des_epoch_100.ckpt',
    'MovieLens20M_300epochs': args.dataset_dir + '/train_embedding/embedding_1hop_des_epoch_300.ckpt',
    # 'Beauty': args.dataset_dir + '/train_embedding_final/embedding_des_epoch_29.ckpt',
    # 'Cellphones_Accessories': args.dataset_dir + '/train_embedding_final/embedding_des_epoch_30.ckpt',
    # 'Clothing': args.dataset_dir + '/train_embedding_final/embedding_des_epoch_29.ckpt',
    # 'CDs_Vinyl': args.dataset_dir + '/train_embedding/embedding_epoch_20.ckpt',
  }
  args.model_file = model_files[args.dataset]

  review_dir = {
    'MovieLens20M': './data_processed',
    'MovieLens20M_100epochs': './data_processed',
    'MovieLens20M_300epochs': './data_processed',
    # 'Beauty': './data/CIKM2017/reviews_Beauty_5.json.gz.stem.nostop/min_count5/query_split',
    # 'CDs_Vinyl': './data/CIKM2017/reviews_CDs_and_Vinyl_5.json.gz.stem.nostop/min_count5/query_split',
    # 'Cellphones_Accessories': './data/CIKM2017/reviews_Cell_Phones_and_Accessories_5.json.gz.stem.nostop/min_count5/query_split',
    # 'Movies_TV': './data/CIKM2017/reviews_Movies_and_TV_5.json.gz.stem.nostop/min_count5/query_split',
    # 'Clothing': './data/CIKM2017/reviews_Clothing_Shoes_and_Jewelry_5.json.gz.stem.nostop/min_count5/query_split',
  }
  # args.train_review_file = review_dir[args.dataset] + '/train.txt.gz'
  # args.test_review_file = review_dir[args.dataset] + '/test.txt.gz'

  args.train_review_file = review_dir[args.dataset] + '/relations_indices/user.watched.movie.txt'
  args.test_review_file = review_dir[args.dataset] + '/test_indices/test_data.txt'

  load_embedding(args)
  load_train_reviews(args)
  load_test_reviews(args)

  test(args)


if __name__ == '__main__':
  main()



### test number of users in train set <= 5 ###
# #################################################################################################
#
# user_watched = {}  # {uid: [pid,...], ...}
# with open('/home/hrv7/harsh/PycharmProjects/movielens-transe/data_processed/relations_indices/user.watched.movie.txt', 'r') as file_:
#   for line in file_:
#     line = line.strip()
#     arr = line.split('\t')
#     user_idx = int(arr[0])
#     movie_idx = int(arr[2])
#     if user_idx not in user_watched:
#       user_watched[user_idx] = []
#     user_watched[user_idx].append(movie_idx)
#
# for key, items in user_watched.items():
#   if len(items)<=5:
#     print('train_user: ', key)
#     print(items)
#
# user_watched = {}  # {uid: [pid,...], ...}
# with open('/home/hrv7/harsh/PycharmProjects/movielens-transe/data_processed/test_indices/test_data.txt', 'r') as f:
#   for line in f:
#     line = line.strip()
#     arr = line.split('\t')
#     user_idx = int(arr[0])
#     movie_idx = int(arr[2])
#     if user_idx not in user_watched:
#       user_watched[user_idx] = []
#     user_watched[user_idx].append(movie_idx)
#
# for key, items in user_watched.items():
#   if len(items)<=5:
#     print('test_user: ', key)