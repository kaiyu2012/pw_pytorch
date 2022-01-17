import torch
import torch.nn.utils.prune as prune
import numpy as np
import torch.nn.functional as F
from utils import process_data

class SynflowPruning_4layerTest():

  def __init__(self, model, prune_epochs, epoch, threshold, schedule, sparsity, separate_compression, weight_sparsity, adj_sparsity, bias_sparsity, optimizer, dataset, prune_weight, prune_bias, prune_adj, cuda):
          self.model = model
          self.prune_epochs = prune_epochs
          self.epoch = epoch
          self.threshold = threshold
          self.schedule = schedule
          self.sparsity  = sparsity
          self.separate_compression = separate_compression
          self.weight_sparsity = weight_sparsity
          self.adj_sparsity = adj_sparsity
          self.bias_sparsity = bias_sparsity
          self.optimizer = optimizer
          self.dataset = dataset
          self.prune_weight = prune_weight
          self.prune_bias = prune_bias
          self.prune_adj = prune_adj
          self.cuda = cuda
          self.scores = {}

  @torch.no_grad()
  def linearize(self, model):
      # model.double()
      signs = {}
      for name, param in model.state_dict().items():
          signs[name] = torch.sign(param)
          param.abs_()
      return signs
  
  @torch.no_grad()
  def nonlinearize(self, model, signs):
      # model.float()
      for name, param in model.state_dict().items():
          param.mul_(signs[name])

  def calc_threshold(self, scores, sparsity_value):

        if self.schedule == 'exponential':
          sparse = sparsity_value**((self.epoch + 1) / self.prune_epochs)
        elif self.schedule == 'linear':
          sparse = 1.0 - (1.0 - sparsity_value)*((self.epoch + 1) / self.prune_epochs)

        k = int((1.0 - sparse) * scores.numel())
        if not k < 1:
            threshold, _ = torch.kthvalue(scores, k)
        return threshold
  
  def compute_score_conservation(self):
      in_scores = []
      out_scores = []
      in_scores.append(torch.sum(self.scores['gc1_weight'].detach(), dim=0) + self.scores['gc1_bias'].detach())
      out_scores.append(torch.sum(self.scores['gc1_weight'].detach(), dim=1))
      in_scores.append(torch.sum(self.scores['gc2_weight'].detach(), dim=0) + self.scores['gc2_bias'].detach())
      out_scores.append(torch.sum(self.scores['gc2_weight'].detach(), dim=1))
      in_scores.append(torch.sum(self.scores['gc3_weight'].detach(), dim=0) + self.scores['gc3_bias'].detach())
      out_scores.append(torch.sum(self.scores['gc3_weight'].detach(), dim=1))
      in_scores.append(torch.sum(self.scores['gc4_weight'].detach(), dim=0) + self.scores['gc4_bias'].detach())
      out_scores.append(torch.sum(self.scores['gc4_weight'].detach(), dim=1))

      in_scores = torch.flatten(in_scores[:-1][0])
      out_scores = torch.flatten(out_scores[1:][0])

      print('in_scores', in_scores)
      print('out_scores', out_scores)
      print('in_scores after sum', torch.sum(in_scores))
      print('out_scores after sum', torch.sum(out_scores))
      return in_scores, out_scores

  def compute_mask(self):
        zero = torch.tensor([0.])
        one = torch.tensor([1.])
        # Load data
        adj, features, labels, idx_train, idx_val, idx_test = process_data("./data/", self.dataset) 
        adj_nonzero_count = torch.sum(adj != 0)

        #print('adj', adj)
        ones = torch.ones(features.shape)
        if self.cuda:
          zero = zero.cuda()
          one = one.cuda()

          self.model.cuda()
          features = features.cuda()
          adj = adj.cuda()
          labels = labels.cuda()
          idx_train = idx_train.cuda()
          idx_val = idx_val.cuda()
          idx_test = idx_test.cuda()
                  
          ones = ones.cuda()

        signs = self.linearize(self.model)
        self.model.train()

        output = self.model(ones, adj)

        self.model.gc1.weight.retain_grad()
        self.model.gc2.weight.retain_grad()
        self.model.gc3.weight.retain_grad()
        self.model.gc4.weight.retain_grad()
        if(self.prune_bias):
          self.model.gc1.bias.retain_grad()
          self.model.gc2.bias.retain_grad()
          self.model.gc3.bias.retain_grad()
          self.model.gc4.bias.retain_grad()
        if self.prune_adj:
          self.model.adj_weight.retain_grad()
        torch.sum(output).backward()

        #scores = {}
        if(self.prune_weight):
          self.scores['gc1_weight'] = torch.clone(self.model.gc1.weight.grad * self.model.gc1.weight).detach().abs_()
          self.scores['gc2_weight'] = torch.clone(self.model.gc2.weight.grad * self.model.gc2.weight).detach().abs_()
          self.scores['gc3_weight'] = torch.clone(self.model.gc3.weight.grad * self.model.gc3.weight).detach().abs_()
          self.scores['gc4_weight'] = torch.clone(self.model.gc4.weight.grad * self.model.gc4.weight).detach().abs_()
          weight_scores = torch.cat([torch.flatten(self.scores['gc1_weight']), torch.flatten(self.scores['gc2_weight']), torch.flatten(self.scores['gc3_weight']), torch.flatten(self.scores['gc4_weight'])])
        if(self.prune_bias):
          self.scores['gc1_bias'] = torch.clone(self.model.gc1.bias.grad * self.model.gc1.bias).detach().abs_()
          self.scores['gc2_bias'] = torch.clone(self.model.gc2.bias.grad * self.model.gc2.bias).detach().abs_()
          self.scores['gc3_bias'] = torch.clone(self.model.gc3.bias.grad * self.model.gc3.bias).detach().abs_()
          self.scores['gc4_bias'] = torch.clone(self.model.gc4.bias.grad * self.model.gc4.bias).detach().abs_()
          bias_scores = torch.cat([torch.flatten(self.scores['gc1_bias']), torch.flatten(self.scores['gc2_bias']), torch.flatten(self.scores['gc3_bias']), torch.flatten(self.scores['gc4_bias'])])

        if self.prune_adj:
          score_adj_weight = torch.clone(self.model.best_adj_grad * self.model.best_adj_weight).detach().abs_()
          temp_adj_score = torch.clone(score_adj_weight)
          temp_adj_score = torch.flatten(temp_adj_score)
          temp_adj_score, indices = torch.sort(temp_adj_score, descending=True)
          self.scores['adj_weight'] = temp_adj_score[0:adj_nonzero_count]
          adj_scores = torch.cat([torch.flatten(self.scores['adj_weight'])])

        global_scores = torch.cat([torch.flatten(v) for v in self.scores.values()])
        self.nonlinearize(self.model, signs)
