from __future__ import division
from __future__ import print_function

import sys
import time
import argparse
import numpy as np
from tensorflow.python.framework.test_util import for_all_test_methods

import torch
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import torch.optim as optim
from torchprofile import profile_macs
import warnings

from utils import process_data, accuracy, plot_confusion, plot_tsne
from gnns.models import GCN
from pruning.weight_pruning import ThresholdPruning
from pruning.synflow_pruning import SynflowPruning
from pruning.m1_data_synflow_pruning import m1_SynflowPruning
from pruning.m2_data_positive_weight_synflow_pruning import m2_SynflowPruning
from pruning.gradient_score_based_pruning import GradientScoreBasedPruning
from pruning.m1_gradient_score_based_pruning import m1_GradientScoreBasedPruning
from pruning.m2_gradient_score_based_pruning import m2_GradientScoreBasedPruning
from pruning.top_k_gradient_score_based_pruning import TopK_GradientScoreBasedPruning

warnings.filterwarnings('ignore')
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--gpu', type=int, default='0',
                        help='number of GPU device to use (default: 0)')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--data', type=str, default="wiki", choices=["cora", "pubmed", "citeseer", "wiki"], help='dataset.')

#wiki dataset
parser.add_argument('--wiki_split_index', type=int, default=0, help='20(0-19) different training splits')

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
parser.add_argument('--pruner', type=str, default='top_k_gradient_score',
                    choices = ['synflow', 'm1_synflow', 'm2_synflow', 'gradient_score', 'm1_gradient_score', 'm2_gradient_score', 'top_k_gradient_score'], help='Prune strategy')
parser.add_argument('--prune_epochs', type=int, default=10,
                    help='no. of iterations for scoring')
parser.add_argument('--top_k_adj', type=int, default=1,
                    help='value of top k adj')
parser.add_argument('--top_k_weight', type=int, default=15,
                    help='value of top k weight')
parser.add_argument('--top_k_bias', type=int, default=15,
                    help='value of top k bias')

parser.add_argument('--prune-batchnorm', type=bool, default=False,
                    help='whether to prune batchnorm layers')
parser.add_argument('--prune-residual', type=bool, default=False,
                    help='whether to prune residual connections')
parser.add_argument('--compression-schedule', type=str, default='exponential', choices=['linear','exponential'],
                        help='whether to use a linear or exponential compression schedule (default: exponential)')
parser.add_argument('--compression', type=float, default=1.0,
                    help='power of 10(should not be set to 0.0)') 
parser.add_argument('--separate_compression', type=bool, default=True,
                    help='whether to use separate compression')                  
parser.add_argument('--compression_weight', type=float, default=0.2,
                    help='power of 10') 
parser.add_argument('--compression_adj', type=float, default=0.2,
                    help='power of 10') 
parser.add_argument('--compression_bias', type=float, default=5.0,
                    help='power of 10') 

parser.add_argument('--prune_weight', type=bool, default=True,
                    help='whether to prune weight')
parser.add_argument('--prune_bias', type=bool, default=False,
                    help='whether to prune bias params')
parser.add_argument('--prune_adj', type=bool, default=True,
                    help='whether to prune adj matrix')

#new parameters used in multishot experiment
parser.add_argument('--compression-list', type=float, nargs='*', default=[],
                        help='list of compression ratio exponents for singleshot/multishot (default: [])')
parser.add_argument('--compression-list_weight', type=float, nargs='*', default=[],
                        help='list of compression ratio exponents for singleshot/multishot (default: [])')
parser.add_argument('--compression-list_adj', type=float, nargs='*', default=[],
                        help='list of compression ratio exponents for singleshot/multishot (default: [])')
parser.add_argument('--compression-list_bias', type=float, nargs='*', default=[],
                        help='list of compression ratio exponents for singleshot/multishot (default: [])')
parser.add_argument('--level-list', type=int, nargs='*', default=[],
                        help='list of number of prune-train cycles (levels) for multishot (default: [])')
parser.add_argument('--result-dir', type=str, default='results/',
                        help='path to directory to save results (default: "results/")')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device(("cuda:" + str(args.gpu)) if args.cuda else "cpu")
print('device: ', device)
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
'''for p in gcn.parameters():
  print(p.name, p.data)
parameters_to_prune = (
    (gcn.gc1, 'weight'),
    (gcn.gc2, 'weight'),
    (gcn.gc1, 'bias'),
    (gcn.gc2, 'bias'),
    (gcn.gc1, 'adj_weight')
    (gcn.gc2, 'adj_weight')
)'''


'''prune.global_unstructured(
    parameters_to_prune,
    pruning_method=ThresholdPruning, threshold=args.weight_pruning_threshold,
)'''


optimizer = optim.Adam(gcn.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

torch.save(optimizer.state_dict(),"{}/optimizer.pt".format(args.result_dir))

if args.cuda:
    gcn.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    if args.data == 'wiki':
      idx_train = idx_train[args.wiki_split_index].cuda()
      idx_val = idx_val[args.wiki_split_index].cuda()
    else:
      idx_train = idx_train.cuda()
      idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    
def init_mask(model):
  mask_gc1_w = torch.ones(model.gc1.weight.shape).to(device)
  mask_gc2_w = torch.ones(model.gc2.weight.shape).to(device)
  mask_gc1_b = torch.ones(model.gc1.bias.shape).to(device)
  mask_gc2_b = torch.ones(model.gc2.bias.shape).to(device)
  adj_mask = adj

  return mask_gc1_w, mask_gc2_w, mask_gc1_b, mask_gc2_b, adj_mask

def apply_mask(model, mask_gc1_w, mask_gc2_w, mask_gc1_b, mask_gc2_b):
  #apply_mask
  if args.prune_weight:
    prune.custom_from_mask(model.gc1, name='weight', mask = mask_gc1_w)
    prune.custom_from_mask(model.gc2, name='weight', mask = mask_gc2_w)
  if(args.prune_bias):
    prune.custom_from_mask(model.gc1, name='bias', mask = mask_gc1_b)
    prune.custom_from_mask(model.gc2, name='bias', mask = mask_gc2_b)

def calc_sparsity(compression):
  sparsity = 10**(-float(compression))
  return sparsity

def calc_sparsity_multishot(compression, l, level):
  sparsity = (10**(-float(compression)))**((l + 1) / level)
  return sparsity

def compression_list_check():
  length = 0
  if args.prune_weight:
      if(len(args.compression_list_weight) == 0):
        print('The compression_list_weight is empty')
        sys.exit()
      else:
        length = len(args.compression_list_weight)
  if args.prune_adj:
    if(len(args.compression_list_adj) == 0):
      print('The compression_list_adj is empty')
      sys.exit()
    else:
      length = len(args.compression_list_adj)
  if args.prune_bias:
    if(len(args.compression_list_bias) == 0):
      print('The compression_list_bias is empty')
      sys.exit()
    else:
      length = len(args.compression_list_bias)

  if args.prune_weight and args.prune_adj:
    if(len(args.compression_list_weight) != len(args.compression_list_adj)):
      print('The compression_lists lengths are not equal')
      sys.exit()
  if args.prune_adj and args.prune_bias:
    if(len(args.compression_list_adj) != len(args.compression_list_bias)):
      print('The compression_lists lengths are not equal')
      sys.exit()
  if args.prune_bias and args.prune_weight:
    if(len(args.compression_list_bias) != len(args.compression_list_weight)):
      print('The compression_lists lengths are not equal')
      sys.exit()
  if args.prune_weight and args.prune_adj and args.prune_bias:
    if(not(len(args.compression_list_weight) == len(args.compression_list_adj) == len(args.compression_list_bias))):
      print('The compression_lists lengths are not equal')
      sys.exit()
    
  return length

def top_k_selection(top_k_list, param_dict, top_k_value):
    if len(top_k_list) < top_k_value:
      top_k_list.append(param_dict)
      top_k_list = sorted(top_k_list, key=lambda param: param['val_acc'], reverse=True)
    else:
      is_found = False
      for param in top_k_list:
        if is_found is False:
          if param_dict['val_acc'] > param['val_acc']:
            is_found = True
            top_k_list.pop()
            top_k_list.append(param_dict)
            top_k_list = sorted(top_k_list, key=lambda param: param['val_acc'], reverse=True)

def train(model, epoch, optim):
    t = time.time()
    model.train()
    optim.zero_grad()
    macs = profile_macs(model, (features, adj))
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])

    model.gc1.weight.retain_grad()
    model.gc2.weight.retain_grad()
    #model.gc1.bias.retain_grad()
    #model.gc2.bias.retain_grad()
    if args.prune_adj:
      model.adj_weight.retain_grad()
    
    loss_train.backward()
    if args.pruner == 'top_k_gradient_score':
      if args.prune_weight:
        gc1_weight_param_dict = {}
        gc1_weight_param_dict['grad'] = torch.clone(model.gc1.weight.grad).detach()
        gc2_weight_param_dict = {}
        gc2_weight_param_dict['grad'] = torch.clone(model.gc2.weight.grad).detach()
        
      if args.prune_adj:
        adj_param_dict = {}
        adj_param_dict['adj_grad'] = torch.clone(model.adj_weight.grad).detach()
        
      if args.prune_bias:
        gc1_bias_param_dict = {}
        gc1_bias_param_dict['grad'] = torch.clone(model.gc1.bias.grad).detach()
        gc2_bias_param_dict = {}
        gc2_bias_param_dict['grad'] = torch.clone(model.gc2.bias.grad).detach()

    optim.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])

    if args.pruner == 'top_k_gradient_score':
      if args.prune_weight:
        gc1_weight_param_dict['val_acc'] = acc_val
        gc2_weight_param_dict['val_acc'] = acc_val
        gc1_weight_param_dict['weight'] = torch.clone(model.gc1.weight).detach()
        gc2_weight_param_dict['weight'] = torch.clone(model.gc2.weight).detach()

        top_k_selection(model.gc1.top_k_weight_list, gc1_weight_param_dict, args.top_k_weight)
        top_k_selection(model.gc2.top_k_weight_list, gc2_weight_param_dict, args.top_k_weight)
      if args.prune_adj:
        adj_param_dict['val_acc'] = acc_val
        adj_param_dict['adj_weight'] = torch.clone(model.adj_weight).detach()
        
        top_k_selection(model.top_k_adj_list, adj_param_dict, args.top_k_adj)
      if args.prune_bias:
        gc1_bias_param_dict['val_acc'] = acc_val
        gc1_bias_param_dict['bias'] = torch.clone(model.gc1.bias).detach()
        gc2_bias_param_dict['val_acc'] = acc_val
        gc2_bias_param_dict['bias'] = torch.clone(model.gc2.bias).detach()

        top_k_selection(model.gc1.top_k_bias_list, gc1_bias_param_dict, args.top_k_bias)
        top_k_selection(model.gc2.top_k_bias_list, gc2_bias_param_dict, args.top_k_bias)

    if args.pruner == 'm1_gradient_score' or args.pruner == 'm2_gradient_score':
      if args.prune_adj:
        adj_param_dict = {}
        adj_param_dict['val_acc'] = acc_val
        adj_param_dict['adj_weight'] = torch.clone(model.adj_weight).detach()
        adj_param_dict['adj_grad'] = torch.clone(model.adj_weight.grad).detach()
        top_k_selection(model.top_k_adj_list, adj_param_dict, args.top_k_adj)
    
    with torch.no_grad():
      if acc_val > best_val_acc['val_acc']:
        best_val_acc['val_acc'] = acc_val
        if args.prune_adj:
          model.best_adj_weight.copy_(model.adj_weight)
          if model.adj_weight.grad is not None:
            model.best_adj_grad.copy_(model.adj_weight.grad)

    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))
    
    print("Best val acc:", best_val_acc['val_acc'])
    print('Inference MACs:[{:.2f}M]'.format(macs/1e6))

    print(
        "Sparsity in gc1.weight: {:.2f}%".format(
            100. * float(torch.sum(model.gc1.weight == 0))
            / float(model.gc1.weight.nelement())
        )
    )
    print(
        "Sparsity in gc2.weight: {:.2f}%".format(
            100. * float(torch.sum(model.gc2.weight == 0))
            / float(model.gc2.weight.nelement())
        )
    )
    #weight sparsity
    print(
        "Weight Sparsity: {:.2f}%".format(
            100. * (float(torch.sum(model.gc1.weight == 0)) + float(torch.sum(model.gc2.weight == 0)))
            / (float(model.gc1.weight.nelement()) + float(model.gc2.weight.nelement()))
        )
    )
    print(
        "Sparsity in gc1.bias: {:.2f}%".format(
            100. * float(torch.sum(model.gc1.bias == 0))
            / float(model.gc1.bias.nelement())
        )
    )
    print(
        "Sparsity in gc2.bias: {:.2f}%".format(
            100. * float(torch.sum(model.gc2.bias == 0))
            / float(model.gc2.bias.nelement())
        )
    )
    if args.prune_adj:
      print(
          "Sparsity in adj_weight: {:.2f}%".format(
              100. * float(torch.sum(model.adj_weight == 0))
              / float(model.adj_weight.nelement())
          )
      )
    print(
        "Sparsity in adj_matrix: {:.2f}%".format(
            100. * float(torch.sum(adj == 0))
            / float(adj.nelement())
        )
    )
    
    print(
        "Sparsity in adj_mask: {:.2f}%".format(
            100. * float(torch.sum(model.adj_mask == 0))
            / float(model.adj_mask.nelement())
        )
    )

    print("Prune adj percent: {:.2f}%".format(100. * (1 - (float(torch.sum(model.adj_mask != 0)) / float(torch.sum(adj != 0))))))
    #added adj into global sparsity
    '''print(
        "Global sparsity: {:.2f}%".format(
            100. * float(
                + torch.sum(model.gc1.weight == 0)
                + torch.sum(model.gc2.weight == 0)
                + torch.sum(model.gc1.bias == 0)
                + torch.sum(model.gc2.bias == 0)
                + torch.sum(adj != 0 )
                - torch.sum(model.adj_mask != 0)
            )
            / float(
                + model.gc1.weight.nelement()
                + model.gc2.weight.nelement()
                + model.gc1.bias.nelement()
                + model.gc2.bias.nelement()
                + torch.sum(adj !=0)
            )
        )
    )'''


def test(model):
    model.eval()
    output = model(features, adj)
    #print(labels[idx_test])
    #print(output[idx_test])
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    #plot_confusion(output[idx_test], labels[idx_test])
    #plot_tsne(output[idx_test], labels[idx_test], features[idx_test])

    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

#initialize masks
mask_gc1_weight, mask_gc2_weight, mask_gc1_bias, mask_gc2_bias, adj_mask = init_mask(gcn)


#apply mask
apply_mask(gcn, mask_gc1_weight, mask_gc2_weight, mask_gc1_bias, mask_gc2_bias)
gcn.set_adj_mask(adj_mask)
torch.save(gcn.state_dict(),"{}/gcn_model.pt".format(args.result_dir))

best_val_acc = {'val_acc': 0}


if(args.experiment == 'singleshot'):

  # Pre-Train model
  print('--------------Pre training GCN--------------')

  t_total = time.time()
  for epoch in range(args.pre_epochs):
      train(gcn, epoch, optimizer)
  print("Optimization Finished!")
  print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
  # Pre-Testing
  test(gcn)
  #pruning
  print('--------------Pruning--------------')
  #calculate sparsity
  sparsity = calc_sparsity(args.compression)
  weight_sparsity = calc_sparsity(args.compression_weight)
  adj_sparsity = calc_sparsity(args.compression_adj)
  bias_sparsity = calc_sparsity(args.compression_bias)

  if args.pruner == 'top_k_gradient_score':
    top_k_gradient_pruning = TopK_GradientScoreBasedPruning(
        model = gcn, threshold=args.weight_pruning_threshold, schedule = args.compression_schedule,
          sparsity=sparsity, separate_compression = args.separate_compression, weight_sparsity = weight_sparsity, adj_sparsity = adj_sparsity, bias_sparsity = bias_sparsity,
          optimizer=optimizer, dataset = args.data, prune_weight = args.prune_weight, prune_bias=args.prune_bias, prune_adj = args.prune_adj,
          cuda = args.cuda, top_k_adj = args.top_k_adj, top_k_weight = args.top_k_weight, top_k_bias = args.top_k_bias
      )
    mask_gc1_weight, mask_gc2_weight, mask_gc1_bias, mask_gc2_bias, adj_mask = top_k_gradient_pruning.compute_mask(adj, features, labels, idx_train, idx_val, idx_test)
  else:
    for epoch in range(args.prune_epochs):

      if args.pruner == 'synflow':
        synflow_pruning = SynflowPruning(
          model = gcn, prune_epochs=args.prune_epochs, epoch=epoch, threshold=args.weight_pruning_threshold, schedule = args.compression_schedule,
            sparsity=sparsity, separate_compression = args.separate_compression, weight_sparsity = weight_sparsity, adj_sparsity = adj_sparsity, bias_sparsity = bias_sparsity,
            optimizer=optimizer, dataset = args.data, prune_weight = args.prune_weight, prune_bias=args.prune_bias, prune_adj = args.prune_adj,
            cuda = args.cuda,
        )
      elif args.pruner == 'm1_synflow':
        synflow_pruning = m1_SynflowPruning(
          model = gcn, prune_epochs=args.prune_epochs, epoch=epoch, threshold=args.weight_pruning_threshold, schedule = args.compression_schedule,
            sparsity=sparsity, separate_compression = args.separate_compression, weight_sparsity = weight_sparsity, adj_sparsity = adj_sparsity, bias_sparsity = bias_sparsity,
            optimizer=optimizer, dataset = args.data, prune_weight = args.prune_weight, prune_bias=args.prune_bias, prune_adj = args.prune_adj,
            cuda = args.cuda,
        )
      elif args.pruner == 'm2_synflow':
        synflow_pruning = m2_SynflowPruning(
          model = gcn, prune_epochs=args.prune_epochs, epoch=epoch, threshold=args.weight_pruning_threshold, schedule = args.compression_schedule,
            sparsity=sparsity, separate_compression = args.separate_compression, weight_sparsity = weight_sparsity, adj_sparsity = adj_sparsity, bias_sparsity = bias_sparsity,
            optimizer=optimizer, dataset = args.data, prune_weight = args.prune_weight, prune_bias=args.prune_bias, prune_adj = args.prune_adj,
            cuda = args.cuda,
        )
      
      elif args.pruner == 'gradient_score':
        synflow_pruning = GradientScoreBasedPruning(
          model = gcn, prune_epochs=args.prune_epochs, epoch=epoch, threshold=args.weight_pruning_threshold, schedule = args.compression_schedule,
            sparsity=sparsity, separate_compression = args.separate_compression, weight_sparsity = weight_sparsity, adj_sparsity = adj_sparsity, bias_sparsity = bias_sparsity,
            optimizer=optimizer, dataset = args.data, prune_weight = args.prune_weight, prune_bias=args.prune_bias, prune_adj = args.prune_adj,
            cuda = args.cuda,
        )
      
      elif args.pruner == 'm1_gradient_score':
        synflow_pruning = m1_GradientScoreBasedPruning(
          model = gcn, prune_epochs=args.prune_epochs, epoch=epoch, threshold=args.weight_pruning_threshold, schedule = args.compression_schedule,
            sparsity=sparsity, separate_compression = args.separate_compression, weight_sparsity = weight_sparsity, adj_sparsity = adj_sparsity, bias_sparsity = bias_sparsity,
            optimizer=optimizer, dataset = args.data, prune_weight = args.prune_weight, prune_bias=args.prune_bias, prune_adj = args.prune_adj,
            cuda = args.cuda, top_k_adj = args.top_k_adj
        )
      elif args.pruner == 'm2_gradient_score':
        synflow_pruning = m2_GradientScoreBasedPruning(
          model = gcn, prune_epochs=args.prune_epochs, epoch=epoch, threshold=args.weight_pruning_threshold, schedule = args.compression_schedule,
            sparsity=sparsity, separate_compression = args.separate_compression, weight_sparsity = weight_sparsity, adj_sparsity = adj_sparsity, bias_sparsity = bias_sparsity,
            optimizer=optimizer, dataset = args.data, prune_weight = args.prune_weight, prune_bias=args.prune_bias, prune_adj = args.prune_adj,
            cuda = args.cuda, top_k_adj = args.top_k_adj
        )
      mask_gc1_weight, mask_gc2_weight, mask_gc1_bias, mask_gc2_bias, adj_mask = synflow_pruning.compute_mask(adj, features, labels, idx_train, idx_val, idx_test)
      


  print('--------------Post training GCN--------------')
  #post-train
  best_val_acc['val_acc'] = 0
  gcn.load_state_dict(torch.load("{}/gcn_model.pt".format(args.result_dir), map_location=device), strict=False)
  optimizer.load_state_dict(torch.load("{}/optimizer.pt".format(args.result_dir), map_location=device))

  #apply_mask
  if args.prune_adj:
    gcn.apply_adj_mask = True
    gcn.set_adj_mask(adj_mask)

  apply_mask(gcn, mask_gc1_weight, mask_gc2_weight, mask_gc1_bias, mask_gc2_bias)

  t_total = time.time()
  for epoch in range(args.post_epochs):
      train(gcn, epoch, optimizer)
  
  print("Optimization Finished!")
  print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
  # Post-Testing
  
  test(gcn)


elif(args.experiment == 'multishot'):
  
  if args.separate_compression:
    compression_list_length = compression_list_check()
  else:
    compression_list_length = len(args.compression_list)

  for index in range(compression_list_length):
        for level in args.level_list:
          if args.separate_compression:
            print('{} compression weight ratio, compression adj ratio, compression bias ratio, {} train-prune levels'.format(args.compression_list_weight[index], 
              args.compression_list_adj[index], args.compression_list_bias[index], level))
          else:
            print('{} compression ratio, {} train-prune levels'.format(args.compression_list[index], level))

          #initialize masks
          mask_gc1_weight, mask_gc2_weight, mask_gc1_bias, mask_gc2_bias, adj_mask = init_mask(gcn)
          
          gcn.load_state_dict(torch.load("{}/gcn_model.pt".format(args.result_dir), map_location=device))
          optimizer.load_state_dict(torch.load("{}/optimizer.pt".format(args.result_dir), map_location=device))
          
          gcn.apply_adj_mask = False       

          for l in range(level):
            best_val_acc['val_acc'] = 0
            #apply mask
            gcn.set_adj_mask(adj_mask)

            apply_mask(gcn, mask_gc1_weight, mask_gc2_weight, mask_gc1_bias, mask_gc2_bias)

            
            # Pre-Train model
            print('--------------Pre training GCN--------------')
            for epoch in range(args.pre_epochs):
                train(gcn, epoch, optimizer)

            #pruning
            print('--------------Pruning--------------')
            weight_sparsity = 0
            adj_sparsity = 0
            bias_sparsity = 0
            sparsity = 0
            if args.separate_compression:
              weight_sparsity = calc_sparsity_multishot(args.compression_list_weight[index], l, level)
              adj_sparsity = calc_sparsity_multishot(args.compression_list_adj[index], l, level)
              bias_sparsity = calc_sparsity_multishot(args.compression_list_bias[index], l, level)
            else:
              sparsity = calc_sparsity_multishot(args.compression_list[index], l, level)
            if args.pruner == 'top_k_gradient_score':
              top_k_gradient_pruning = TopK_GradientScoreBasedPruning(
                  model = gcn, threshold=args.weight_pruning_threshold, schedule = args.compression_schedule,
                    sparsity=sparsity, separate_compression = args.separate_compression, weight_sparsity = weight_sparsity, adj_sparsity = adj_sparsity, bias_sparsity = bias_sparsity,
                    optimizer=optimizer, dataset = args.data, prune_weight = args.prune_weight, prune_bias=args.prune_bias, prune_adj = args.prune_adj,
                    cuda = args.cuda, top_k_adj = args.top_k_adj, top_k_weight = args.top_k_weight, top_k_bias = args.top_k_bias
                )
              mask_gc1_weight, mask_gc2_weight, mask_gc1_bias, mask_gc2_bias, adj_mask = top_k_gradient_pruning.compute_mask(adj, features, labels, idx_train, idx_val, idx_test)
            else:
              for epoch in range(args.prune_epochs):

                if args.pruner == 'synflow':
                  synflow_pruning = SynflowPruning(
                    model = gcn, prune_epochs=args.prune_epochs, epoch=epoch, threshold=args.weight_pruning_threshold, schedule = args.compression_schedule,
                      sparsity=sparsity, separate_compression = args.separate_compression, weight_sparsity = weight_sparsity, adj_sparsity = adj_sparsity, bias_sparsity = bias_sparsity,
                      optimizer=optimizer, dataset = args.data, prune_weight = args.prune_weight, prune_bias=args.prune_bias, prune_adj = args.prune_adj,
                      cuda = args.cuda,
                  )
                elif args.pruner == 'm1_synflow':
                  synflow_pruning = m1_SynflowPruning(
                    model = gcn, prune_epochs=args.prune_epochs, epoch=epoch, threshold=args.weight_pruning_threshold, schedule = args.compression_schedule,
                      sparsity=sparsity, separate_compression = args.separate_compression, weight_sparsity = weight_sparsity, adj_sparsity = adj_sparsity, bias_sparsity = bias_sparsity,
                      optimizer=optimizer, dataset = args.data, prune_weight = args.prune_weight, prune_bias=args.prune_bias, prune_adj = args.prune_adj,
                      cuda = args.cuda,
                  )
                elif args.pruner == 'm2_synflow':
                  synflow_pruning = m2_SynflowPruning(
                    model = gcn, prune_epochs=args.prune_epochs, epoch=epoch, threshold=args.weight_pruning_threshold, schedule = args.compression_schedule,
                      sparsity=sparsity, separate_compression = args.separate_compression, weight_sparsity = weight_sparsity, adj_sparsity = adj_sparsity, bias_sparsity = bias_sparsity,
                      optimizer=optimizer, dataset = args.data, prune_weight = args.prune_weight, prune_bias=args.prune_bias, prune_adj = args.prune_adj,
                      cuda = args.cuda,
                  )
                elif args.pruner == 'gradient_score':
                  synflow_pruning = GradientScoreBasedPruning(
                    model = gcn, prune_epochs=args.prune_epochs, epoch=epoch, threshold=args.weight_pruning_threshold, schedule = args.compression_schedule,
                      sparsity=sparsity, separate_compression = args.separate_compression, weight_sparsity = weight_sparsity, adj_sparsity = adj_sparsity, bias_sparsity = bias_sparsity,
                      optimizer=optimizer, dataset = args.data, prune_weight = args.prune_weight, prune_bias=args.prune_bias, prune_adj = args.prune_adj,
                      cuda = args.cuda,
                  )
                elif args.pruner == 'm1_gradient_score':
                  synflow_pruning = m1_GradientScoreBasedPruning(
                    model = gcn, prune_epochs=args.prune_epochs, epoch=epoch, threshold=args.weight_pruning_threshold, schedule = args.compression_schedule,
                      sparsity=sparsity, separate_compression = args.separate_compression, weight_sparsity = weight_sparsity, adj_sparsity = adj_sparsity, bias_sparsity = bias_sparsity,
                      optimizer=optimizer, dataset = args.data, prune_weight = args.prune_weight, prune_bias=args.prune_bias, prune_adj = args.prune_adj,
                      cuda = args.cuda,
                  )
                elif args.pruner == 'm2_gradient_score':
                  synflow_pruning = m2_GradientScoreBasedPruning(
                    model = gcn, prune_epochs=args.prune_epochs, epoch=epoch, threshold=args.weight_pruning_threshold, schedule = args.compression_schedule,
                      sparsity=sparsity, separate_compression = args.separate_compression, weight_sparsity = weight_sparsity, adj_sparsity = adj_sparsity, bias_sparsity = bias_sparsity,
                      optimizer=optimizer, dataset = args.data, prune_weight = args.prune_weight, prune_bias=args.prune_bias, prune_adj = args.prune_adj,
                      cuda = args.cuda, top_k_adj = args.top_k_adj
                  )
                mask_gc1_weight, mask_gc2_weight, mask_gc1_bias, mask_gc2_bias, adj_mask = synflow_pruning.compute_mask(adj, features, labels, idx_train, idx_val, idx_test)

            #reset model's weights
            original_dict = torch.load("{}/gcn_model.pt".format(args.result_dir), map_location=device)
            
            original_weights = dict(filter(lambda v: (v[0].endswith(('adj_weight', '.weight', '.bias'))), original_dict.items()))
            
            
            gcn_dict = gcn.state_dict()
            gcn_dict.update(original_weights)

            
            gcn.load_state_dict(gcn_dict)
            
            optimizer.load_state_dict(torch.load("{}/optimizer.pt".format(args.result_dir), map_location=device))

          # Post-Train model
          print('--------------Post training GCN--------------')
          #apply_mask
          if args.prune_adj:
              gcn.apply_adj_mask = True
              gcn.set_adj_mask(adj_mask)
          apply_mask(gcn, mask_gc1_weight, mask_gc2_weight, mask_gc1_bias, mask_gc2_bias)
          t_total = time.time()
          
          for epoch in range(args.post_epochs):
              train(gcn, epoch, optimizer)
          
          print("Optimization Finished!")
          print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
          # Post-Testing
          test(gcn)


