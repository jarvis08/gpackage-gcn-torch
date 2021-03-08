from __future__ import division
from __future__ import print_function

import os
import time
import argparse
import numpy as np

import torch
from torch.nn import BCEWithLogitsLoss
import torch.optim as optim
from models_gdp import GCN
from utils_gdp import *


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=4000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--cv', type=float, default=1,
                    help="ID of Cross Validation's Fold")

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
dataset_path = "../data/gdp/results/v3"
fold_path = os.path.join(dataset_path, "FOLD-{}".format(args.cv))
labels = load_labels(dataset_path)
labels = torch.FloatTensor(labels)
adj, features, train_mask, valid_mask = load_folded_dataset(fold_path)
train_labels = labels[train_mask]
valid_labels = labels[valid_mask]

model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=265,
            dropout=args.dropout)
optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
)
loss_func = BCEWithLogitsLoss(reduction='none')

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    train_labels = train_labels.cuda()
    valid_labels = valid_labels.cuda()
    train_mask = train_mask.cuda()
    valid_mask = valid_mask.cuda()


def micro_f1(logits, labels):
    predicted = logits.type(torch.IntTensor)
    labels = labels.type(torch.IntTensor)

    true_pos = torch.count_nonzero(predicted * labels)
    false_pos = torch.count_nonzero(predicted * (labels - 1))
    false_neg = torch.count_nonzero((predicted - 1) * labels)

    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    fmeasure = (2 * precision * recall) / (precision + recall)
    return fmeasure


def train(epoch, best_valid_f1, anger, patience):
    early_stop = 0
    t = time.time()
    model.train()
    optimizer.zero_grad()

    output = model(features, adj)
    output_masked = output[train_mask]
    loss_train = loss_func(output_masked, train_labels)
    loss_train = torch.mean(torch.sum(loss_train, -1))
    f1_train = micro_f1(torch.round(torch.sigmoid(output_masked)), train_labels)
    loss_train.backward()
    optimizer.step()

    model.eval()
    output = model(features, adj)
    output_masked = output[valid_mask]
    f1_valid = micro_f1(torch.round(torch.sigmoid(output_masked)), valid_labels)

    print('[Epoch]: {:04d}'.format(epoch+1),
          '\t[T-Loss] {:.4f}'.format(loss_train.item()),
          '\t[T-F1] {:.4f}'.format(f1_train.item()),
          '\t[V-F1] {:.4f}'.format(f1_valid.item()),
          '\t[Time] {:.4f}s'.format(time.time() - t))

    if epoch >= 1000:
        if f1_valid <= best_valid_f1:
            if anger >= patience:
                print("Stop training. Best Validation F1-score : {}".format(best_valid_f1))
                early_stop = 1
            else:
                anger += 1
        else:
            best_valid_f1 = f1_valid
            anger = 0
    return best_valid_f1, anger, early_stop


# Train model
t_total = time.time()
best_valid_f1 = 0
anger = 0
patience = 500
for epoch in range(args.epochs):
    best_valid_f1, anger, early_stop = train(epoch, best_valid_f1, anger, patience)
    if early_stop:
        break
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
