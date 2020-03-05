from lightfm.data import Dataset
from lightfm import LightFM
import csv
from easydict import EasyDict as edict
import os
import pickle
from lightfm.cross_validation import random_train_test_split
from scipy.sparse import coo_matrix as sp
from lightfm.evaluation import precision_at_k
from lightfm.evaluation import recall_at_k
from lightfm.evaluation import auc_score

def get_data():
  movieDict=edict()
  ctr=0
  with open('/home/hrv7/harsh/PycharmProjects/movielens-transe/BPR/data/ml-20m/ratings.csv', 'r') as in_file:
    rd=csv.DictReader(in_file, delimiter=',')
    print(rd.fieldnames)
    a, b, c=[], [], []
    for line in rd:
      a.append(line['userId'])
      b.append(line['movieId'])
      c.append(line['rating'])
      ctr+=1
      if(ctr%1000000==0):
        print(ctr)
    movieDict.userId=a
    movieDict.movieId = b
    movieDict.rating = c

  return movieDict
# def get_ratings():
#   get_data()
def get_ratings():
  movieDict=get_data()
  with open(os.path.join(os.getcwd(), r'preprocessed/movieDict.pkl'), 'wb') as out_file:
    pickle.dump(movieDict, out_file)

movieDict = pickle.load(open(os.path.join(os.getcwd(), r'preprocessed/movieDict.pkl'), 'rb'))

dataset = Dataset()
dataset.fit((list(set(movieDict.userId))),
            (list(set(movieDict.movieId))))


num_users, num_items = dataset.interactions_shape()
print('Num users: {}, num_items {}.'.format(num_users, num_items))
(interactions, weights) = dataset.build_interactions(list(zip(movieDict.userId, movieDict.movieId)))

print(repr(interactions))

(train, test) = random_train_test_split(interactions=interactions, test_percentage=0.2)

model = LightFM(learning_rate=0.05, loss='bpr')
model.fit(train, epochs=2, num_threads=12)
# model.fit(train, epochs=10)

train_precision = precision_at_k(model, train, k=10).mean()
test_precision = precision_at_k(model, test, k=10).mean()
r_train_precision=recall_at_k(model, train, k=10).mean()
r_test_precision=recall_at_k(model, test, k=10).mean()

train_auc = auc_score(model, train).mean()
test_auc = auc_score(model, test).mean()

print('Precision: train %.2f, test %.2f.' % (train_precision, test_precision))
print('AUC: train %.2f, test %.2f.' % (train_auc, test_auc))
#
#
# import numpy as np
#
#
#
#
# from lightfm.datasets import fetch_movielens
#
# movielens = fetch_movielens()
#
#
# for key, value in movielens.items():
#     print(key, type(value), value.shape)
#
# train = movielens['train']
# test = movielens['test']
# print(movielens['item_features'])
# # print(train)