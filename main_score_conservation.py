from __future__ import division
from __future__ import print_function

import sys
import time
import argparse
import numpy as np

import torch
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import torch.optim as optim

from utils import process_data, accuracy, plot_confusion, plot_tsne
from gnns.models import GCN_4layers as GCN
from pruning.weight_pruning import ThresholdPruning
from pruning.synflow_pruning import SynflowPruning
from pruning.m1_data_synflow_pruning import m1_SynflowPruning
from pruning.m2_data_positive_weight_synflow_pruning import m2_SynflowPruning
from pruning.synflow_4layerTest import SynflowPruning_4layerTest

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--gpu', type=int, default='0',
                        help='number of GPU device to use (default: 0)')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--data', type=str, default="cora", choices=["cora", "pubmed", "citeseer"], help='dataset.')

parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--weight_pruning_threshold', type=float, default=0.1,
                    help='Dropout rate (1 - keep probability).')

#new parameters 
parser.add_argument('--experiment', type=str, default='singleshot', 
                        choices=['singleshot','multishot'],
                        help='experiment name (default: example)') 
parser.add_argument('--pre_epochs', type=int, default=100,
                    help='Number of epochs to pre-train.')
parser.add_argument('--post_epochs', type=int, default=100,
                    help='Number of epochs to post-train.')                   
parser.add_argument('--pruner', type=str, default='synflow',
                    choices = ['synflow'], help='Prune strategy')
parser.add_argument('--prune_epochs', type=int, default=10,
                    help='no. of iterations for scoring')

parser.add_argument('--prune-batchnorm', type=bool, default=False,
                    help='whether to prune batchnorm layers')
parser.add_argument('--prune-residual', type=bool, default=False,
                    help='whether to prune residual connections')
parser.add_argument('--compression-schedule', type=str, default='exponential', choices=['linear','exponential'],
                        help='whether to use a linear or exponential compression schedule (default: exponential)')
parser.add_argument('--compression', type=float, default=0.25,
                    help='power of 10(should not be set to 0.0)') 
parser.add_argument('--separate_compression', type=bool, default=False,
                    help='whether to use separate compression')                  
parser.add_argument('--compression_weight', type=float, default=0.25,
                    help='power of 10') 
parser.add_argument('--compression_adj', type=float, default=2.5,
                    help='power of 10') 
parser.add_argument('--compression_bias', type=float, default=5.0,
                    help='power of 10') 

parser.add_argument('--prune_weight', type=bool, default=True,
                    help='whether to prune weight')
parser.add_argument('--prune_bias', type=bool, default=True,
                    help='whether to prune bias params')
parser.add_argument('--prune_adj', type=bool, default=False,
                    help='whether to prune adj matrix')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

device = torch.device(("cuda:" + str(args.gpu)) if args.cuda else "cpu")
print('device', device)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj, features, labels, idx_train, idx_val, idx_test = process_data("./data/", args.data)

# Model and optimizer
gcn = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout, adj=adj, prune_adj = args.prune_adj, cuda = args.cuda)

optimizer = optim.Adam(gcn.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    gcn.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

def init_mask(model):
  mask_gc1_w = torch.ones(model.gc1.weight.shape).to(device)
  mask_gc2_w = torch.ones(model.gc2.weight.shape).to(device)
  mask_gc3_w = torch.ones(model.gc3.weight.shape).to(device)
  mask_gc4_w = torch.ones(model.gc4.weight.shape).to(device)
  mask_gc1_b = torch.ones(model.gc1.bias.shape).to(device)
  mask_gc2_b = torch.ones(model.gc2.bias.shape).to(device)
  mask_gc3_b = torch.ones(model.gc3.bias.shape).to(device)
  mask_gc4_b = torch.ones(model.gc4.bias.shape).to(device)
  adj_mask = adj

  return mask_gc1_w, mask_gc2_w, mask_gc3_w, mask_gc4_w, mask_gc1_b, mask_gc2_b, mask_gc3_b, mask_gc4_b, adj_mask

def apply_mask(model, mask_gc1_w, mask_gc2_w, mask_gc3_w, mask_gc4_w, mask_gc1_b, mask_gc2_b, mask_gc3_b, mask_gc4_b):
  #apply_mask
  if args.prune_weight:
    prune.custom_from_mask(model.gc1, name='weight', mask = mask_gc1_w)
    prune.custom_from_mask(model.gc2, name='weight', mask = mask_gc2_w)
    prune.custom_from_mask(model.gc3, name='weight', mask = mask_gc3_w)
    prune.custom_from_mask(model.gc4, name='weight', mask = mask_gc4_w)
  if(args.prune_bias):
    prune.custom_from_mask(model.gc1, name='bias', mask = mask_gc1_b)
    prune.custom_from_mask(model.gc2, name='bias', mask = mask_gc2_b)
    prune.custom_from_mask(model.gc3, name='bias', mask = mask_gc3_b)
    prune.custom_from_mask(model.gc4, name='bias', mask = mask_gc4_b)

def calc_sparsity(compression):
  sparsity = 10**(-float(compression))
  return sparsity


#initialize masks
mask_gc1_weight, mask_gc2_weight, mask_gc3_weight, mask_gc4_weight, mask_gc1_bias, mask_gc2_bias, mask_gc3_bias, mask_gc4_bias, adj_mask = init_mask(gcn)


#apply mask
apply_mask(gcn, mask_gc1_weight, mask_gc2_weight, mask_gc3_weight, mask_gc4_weight, mask_gc1_bias, mask_gc2_bias, mask_gc3_bias, mask_gc4_bias)
gcn.set_adj_mask(adj_mask)

#pruning
print('--------------Pruning--------------')

for epoch in range(args.prune_epochs):
  #calculate sparsity
  sparsity = calc_sparsity(args.compression)
  weight_sparsity = calc_sparsity(args.compression_weight)
  adj_sparsity = calc_sparsity(args.compression_adj)
  bias_sparsity = calc_sparsity(args.compression_bias)

  if args.pruner == 'synflow':
    synflow_pruning = SynflowPruning_4layerTest(
      model = gcn, prune_epochs=args.prune_epochs, epoch=epoch, threshold=args.weight_pruning_threshold, schedule = args.compression_schedule,
        sparsity=sparsity, separate_compression = args.separate_compression, weight_sparsity = weight_sparsity, adj_sparsity = adj_sparsity, bias_sparsity = bias_sparsity,
        optimizer=optimizer, dataset = args.data, prune_weight = args.prune_weight, prune_bias=args.prune_bias, prune_adj = args.prune_adj,
        cuda = args.cuda,
    )
  
  synflow_pruning.compute_mask()

synflow_pruning.compute_score_conservation()