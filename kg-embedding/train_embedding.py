from __future__ import absolute_import, division, print_function

import os
import sys
import argparse
import random
import numpy as np
import pickle
import logging
import logging.handlers
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

from data_utils import *
#from models import KnowledgeEmbedding
#from models_2hop import KnowledgeEmbedding
#from models_2hop_des import KnowledgeEmbedding
from models_3hop_des import KnowledgeEmbedding



logger = None


def set_logger(logname):
    global logger
    logger = logging.getLogger(logname)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(levelname)s]  %(message)s')
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    fh = logging.handlers.RotatingFileHandler(logname, mode='w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(args):
    with open(args.dataset_file, 'rb') as f:
        dataset = pickle.load(f)
    dataloader = AmazonDataLoader(dataset, args.batch_size)
    words_to_train = args.epochs * dataset.review.word_count + 1

    model = KnowledgeEmbedding(dataset, args).to(args.device)
    logger.info('Parameters:' + str([i[0] for i in model.named_parameters()]))
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    #optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = 0
    smooth_loss = 0.0
    train_writer = SummaryWriter(args.log_dir)

    for epoch in range(1, args.epochs + 1):
        dataloader.reset()
        while dataloader.has_next():
            # Manully set learning rate.
            lr = args.lr * max(1e-4, 1.0 - dataloader.finished_word_num / float(words_to_train))
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            # Get training batch.
            batch_idxs = dataloader.get_batch()
            batch_idxs = torch.from_numpy(batch_idxs).to(args.device)

            # Train model.
            optimizer.zero_grad()
            train_loss = model(batch_idxs)
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            smooth_loss += train_loss.item() / args.steps_per_checkpoint

            steps += 1
            if steps % args.steps_per_checkpoint == 0:
                train_writer.add_scalar('train/loss', train_loss.item(), steps)
                train_writer.add_scalar('train/average_loss', smooth_loss, steps)
                logger.info('Epoch: {:02d} | '.format(epoch) +
                            'Words: {:d}/{:d} | '.format(dataloader.finished_word_num, words_to_train) +
                            'Lr: {:.5f} | '.format(lr) +
                            'Smooth loss: {:.5f}'.format(smooth_loss))
                smooth_loss = 0.0

        torch.save(model.state_dict(), '{}/embedding_3hop_des_epoch_{}.ckpt'.format(args.log_dir, epoch))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cellphones_Accessories',
                        help='One of {Beauty, CDs_Vinyl, Cellphones_Accessories, Moives_TV, Clothing}')
    parser.add_argument('--seed', type=int, default=123, help='random seed.')
    parser.add_argument('--gpu', type=str, default='cuda:7', help='gpu device.')
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size.')
    parser.add_argument('--lr', type=float, default=0.5, help='learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay for adam.')
    parser.add_argument('--l2_lambda', type=float, default=0, help='l2 lambda')
    parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Clipping gradient.')
    parser.add_argument('--embed_size', type=int, default=100, help='knowledge embedding size.')
    parser.add_argument('--num_neg_samples', type=int, default=5, help='number of negative samples.')
    parser.add_argument('--steps_per_checkpoint', type=int, default=200, help='Number of steps for checkpoint.')
    args = parser.parse_args()

    args.dataset_dir = './tmp/{}'.format(args.dataset)
    args.dataset_file = args.dataset_dir + '/dataset.pkl'
    args.log_dir = args.dataset_dir + '/train_3hop_des_embedding'
    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)
    set_logger(args.log_dir + '/train_log.txt')
    args.device = torch.device(args.gpu) if torch.cuda.is_available() else 'cpu'
    logger.info(args)

    set_random_seed(args.seed)
    train(args)


if __name__ == '__main__':
    main()
