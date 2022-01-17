#random pruning
from __future__ import division
from __future__ import print_function

import sys
import time
import copy
import argparse
import numpy as np

import torch
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import torch.optim as optim

from utils import process_data, accuracy, plot_confusion, plot_tsne
from gnns.models import UGS_GCN as GCN
from pruning import UGS_pruning

TRAIN_PRINT = False

# Training settings

def parser_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument('--no-cuda', action='store_true', default=False,
                      help='Disables CUDA training.')
  parser.add_argument('--gpu', type=int, default='0',
                          help='number of GPU device to use (default: 0)')
  parser.add_argument('--fastmode', action='store_true', default=False,
                      help='Validate during training pass.')
  parser.add_argument('--seed', type=int, default=42, help='Random seed.')
  
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
  parser.add_argument('--result-dir', type=str, default='results/',
                          help='path to directory to save results (default: "results/")')
  ###### Unify pruning settings #######
  parser.add_argument('--s1', type=float, default=0.0001,help='scale sparse rate (default: 0.0001)')
  parser.add_argument('--s2', type=float, default=0.0001,help='scale sparse rate (default: 0.0001)')
  parser.add_argument('--pruning_percent_wei', type=float, default=0.5)
  parser.add_argument('--pruning_percent_adj', type=float, default=0.5)
  parser.add_argument('--compression_weight', type=float, default=0,
                      help='power of 10') 
  parser.add_argument('--compression_adj', type=float, default=1,
                      help='power of 10') 
  parser.add_argument('--pre_epochs', type=int, default=100,
                      help='Number of epochs to pre-train.')
  parser.add_argument('--post_epochs', type=int, default=100,
                      help='Number of epochs to post-train.') 
  parser.add_argument('--init_soft_mask_type', type=str, default='normal', help='all_one, kaiming, normal, uniform')

  args = parser.parse_args()
  args.cuda = not args.no_cuda and torch.cuda.is_available()
  return args

def set_arguments_compression(args, w_compression, a_compression):
  args.compression_weight = w_compression
  args.compression_adj = a_compression

def set_datasets(args, dataset):
  args.data = dataset

def initialize_gcn(args):
  device = torch.device(("cuda:" + str(args.gpu)) if args.cuda else "cpu")
  #print('device', device)
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
              dropout=args.dropout, adj=adj, cuda = args.cuda)

  optimizer = optim.Adam(gcn.parameters(),
                        lr=args.lr, weight_decay=args.weight_decay)


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
  else:
      if args.data == 'wiki':
        idx_train = idx_train[args.wiki_split_index]
        idx_val = idx_val[args.wiki_split_index]
  return gcn, optimizer, device, adj, features, idx_train, labels, idx_val, idx_test

def calc_sparsity(compression):
  sparsity = 10**(-float(compression))
  return sparsity

def train(model, epoch, optim, features, args, adj, idx_train, labels, idx_val, best_val_acc):
    t = time.time()
    model.train()
    optim.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])

    model.gc1.weight.retain_grad()
    model.gc2.weight.retain_grad()
    #model.gc1.bias.retain_grad()
    #model.gc2.bias.retain_grad()
    
    loss_train.backward()
    
    optim.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    with torch.no_grad():
      if acc_val > best_val_acc['val_acc']:
        best_val_acc['val_acc'] = acc_val
    
    if TRAIN_PRINT: 
      print('Epoch: {:04d}'.format(epoch+1),
            'loss_train: {:.4f}'.format(loss_train.item()),
            'acc_train: {:.4f}'.format(acc_train.item()),
            'loss_val: {:.4f}'.format(loss_val.item()),
            'acc_val: {:.4f}'.format(acc_val.item()),
            'time: {:.4f}s'.format(time.time() - t))
      
      print("Best val acc:", best_val_acc['val_acc'])

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

      print(
          "Sparsity in adj_matrix: {:.2f}%".format(
              100. * float(torch.sum(adj == 0))
              / float(adj.nelement())
          )
      )
      
      print(
          "Sparsity in adj_mask: {:.2f}%".format(
              100. * float(torch.sum(model.adj_mask1_train == 0))
              / float(model.adj_mask1_train.nelement())
          )
      )

      print("Prune adj percent: {:.2f}%".format(100. * (1 - (float(torch.sum(model.adj_mask1_train != 0)) / float(torch.sum(adj != 0))))))
      #added adj into global sparsity
      print(
          "Global sparsity: {:.2f}%".format(
              100. * float(
                  + torch.sum(model.gc1.weight == 0)
                  + torch.sum(model.gc2.weight == 0)
                  + torch.sum(model.gc1.bias == 0)
                  + torch.sum(model.gc2.bias == 0)
                  + torch.sum(adj != 0 )
                  - torch.sum(model.adj_mask1_train != 0)
              )
              / float(
                  + model.gc1.weight.nelement()
                  + model.gc2.weight.nelement()
                  + model.gc1.bias.nelement()
                  + model.gc2.bias.nelement()
                  + torch.sum(adj !=0)
              )
          )
      )
def test(model, features, adj, labels, idx_test):
    model.eval()
    output = model(features, adj)
    #print(labels[idx_test])
    #print(output[idx_test])
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    #plot_confusion(output[idx_test], labels[idx_test])
    #plot_tsne(output[idx_test], labels[idx_test], features[idx_test])

    #print("Test set results:",
          #"loss= {:.4f}".format(loss_test.item()),
          #"accuracy= {:.4f}".format(acc_test.item()))

def single_run(gcn, args, optimizer, device, adj, features, idx_train, labels, idx_val, idx_test):
  best_val_acc = {'val_acc': 0}

  args.pruning_percent_wei = 1 - calc_sparsity(args.compression_weight)
  args.pruning_percent_adj = 1 - calc_sparsity(args.compression_adj)
  #initialize masks
  UGS_pruning.add_mask(gcn)
  UGS_pruning.random_pruning(gcn, args.pruning_percent_adj, args.pruning_percent_wei)

  #print('--------------Post training GCN--------------')
  #post-train
  best_val_acc['val_acc'] = 0

  #adj_spar, wei_spar = UGS_pruning.print_sparsity(gcn)
  for name, param in gcn.named_parameters():
    if 'mask' in name:
        param.requires_grad = False


  t_total = time.time()
  for epoch in range(args.post_epochs):
      train(gcn, epoch, optimizer, features, args, adj, idx_train, labels, idx_val, best_val_acc)

  #print("Optimization Finished!")
  #print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
  # Post-Testing

  test(gcn, features, adj, labels, idx_test)
  return best_val_acc['val_acc']

def main():
  EXPERIMENT_NAME = 'citeseer_random_separate_adj_weight'
  NUMBER_OF_RUNS = 3
  print('EXPERIMENT_NAME: ', EXPERIMENT_NAME)
  path_to_save = 'results/separate_compression/citeseer/0.2 scale/'
  #Experiment configurations
  args = parser_arguments()

  #Experiment variables
  prune_strategy = ['random']

  #adj_compressions = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0]
  #weight_compressions = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0]
  adj_compressions = [0.2, 0.4, 0.6, 0.8, 1.0]
  weight_compressions = [0.2, 0.4, 0.6, 0.8, 1.0]
  #adj_compressions = [0.4, 0.8, 1.2, 1.6, 2.0]
  #weight_compressions = [0.4, 0.8, 1.2, 1.6, 2.0]
  #weight_compressions = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0]
  #adj_compressions = [0]

  set_datasets(args, "citeseer")
  print(f'The data set: {args.data}')

  experiment_results = {}
  experiment_results['number_of_runs'] = NUMBER_OF_RUNS
  experiment_results['adj_compressions'] = adj_compressions
  experiment_results['weight_compressions'] = weight_compressions

  for runs in range(NUMBER_OF_RUNS):
    print(f'Experiment run: {runs}')
    experiment_results['run_'+str(runs)] = {}
    args.seed = int(round(time.time()))

    for pruner in prune_strategy:
      # 'random'
      experiment_results['run_'+str(runs)][pruner] = []
      results = []
      print(f'Prune strategy: {pruner}')
      for y in adj_compressions:
          experiments = []
          for x in weight_compressions:
              
              set_arguments_compression(args, x, y)    
                
              gcn, optimizer, device, adj, features, idx_train, labels, idx_val, idx_test = initialize_gcn(args)
              
              top_val_acc = single_run(gcn, args, optimizer, device, adj, features, idx_train, labels, idx_val, idx_test)
              
              experiments.append(float('{:.4f}'.format(top_val_acc.item())))
              
              print('compression_weight: {:.2f}'.format(args.compression_weight) +  ' compression_adj: {:.2f}'.format(args.compression_adj) + ' -- ++ Result: {:.4f}'.format(top_val_acc.item()))
          results.append(experiments)        
      experiment_results['run_'+str(runs)][pruner].append(results)

  print('\n--- The results of top-1 validation accuracy: ----') 
  print(f'\nPrune strategy: {prune_strategy[0]} -- The data set: {args.data}')
  print(experiment_results)

  print(f'\n Writing results to {EXPERIMENT_NAME}.txt')
  with open(path_to_save+EXPERIMENT_NAME+'.txt', 'w') as f:
    [f.write(str(experiment_results))]

if __name__ == "__main__":
    main()