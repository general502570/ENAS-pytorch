"""Module containing the child model."""
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

import collections

class CNN(nn.Module):
    """CNN model."""
    def __init__(self, args):
        super(CNN, self).__init__()
        self.args = args

        self.conv3s = collections.defaultdict(dict)
        self.conv5s = collections.defaultdict(dict)
        self.maxpools = collections.defaultdict(dict)
        self.meanpools = collections.defaultdict(dict)
        self.pools = collections.defaultdict(dict)
        self.poollayers = []

        for i in range(self.args.layern):
            for j in range(0, self.args.noden):
                self.conv3s[i][j] = nn.Conv2d(self.args.channels[i],
                                              self.args.channels[i],
                                              3, padding = 1, bias = False)
                self.conv5s[i][j] = nn.Conv2d(self.args.channels[i],
                                              self.args.channels[i],
                                              5, padding = 2, bias = False)
                self.maxpools[i][j] = nn.MaxPool2d(3,
                                                   stride = 1,
                                                   padding = 1)
                self.meanpools[i][j] = nn.AvgPool2d(3,
                                                   stride = 1,
                                                   padding = 1)
            if i < self.args.layern - 1:
                self.poollayers.append( nn.Conv2d(self.args.channels[i],
                                                  self.args.channels[i + 1],
                                                  2, stride = 2) )

        self.fc = nn.Linear(12, 10)

        
        self._conv3s = nn.ModuleList([self.conv3s[i][j]
                                   for i in self.conv3s
                                   for j in self.conv3s[i]])
        self._conv5s = nn.ModuleList([self.conv5s[i][j]
                                   for i in self.conv5s
                                   for j in self.conv5s[i]])
        self._maxpools = nn.ModuleList([self.maxpools[i][j]
                                   for i in self.maxpools
                                   for j in self.maxpools[i]])
        self._meanpools = nn.ModuleList([self.meanpools[i][j]
                                   for i in self.meanpools
                                   for j in self.meanpools[i]])
        
        self.convs = [self.conv3s, self.conv5s, self.maxpools, self.meanpools]
        self.pools = nn.ModuleList(self.poollayers)

        self.batch_norm = []
        if self.args.mode == 'train':
            for i in range(self.args.layern):
                self.batch_norm.append( nn.BatchNorm2d(self.args.channels[i]) )
        else:
            for i in range(self.args.layern):
                self.batch_norm.append( None )
        self._batch_norm = nn.ModuleList(self.batch_norm)

        #self.reset_parameters()

    def forward(self,
                x,
                dag,
                is_train=True):
        time_steps = x.size(0)
        batch_size = x.size(1)

        is_train = is_train and self.args.mode in ['train']

        """if self.args.shared_dropouti > 0:
            embed = self.lockdrop(embed,
                                  self.args.shared_dropouti if is_train else 0)
        """

        for layer_id in range(self.args.layern):
            x = self.layer(x, dag, layer_id)
            x = self.pooling_layer(x, layer_id)

        x = self.fc(x)
        return x

    def pooling_layer(self, x, layer_id):
        """Pooling layer after each normal layer"""
        #print(x.size())
        if layer_id < self.args.layern - 1:
            ins = self.pools[layer_id](x)
            if len(self.batch_norm) != 0:
                ins = self.batch_norm[layer_id + 1](ins)
        else:
            # assuming NCHW
            ins = torch.mean(x, (2, 3))
        #print(ins.size())

        outs = F.relu(ins)
        return outs

    def layer(self, x, dag, layer_id):
        """normal layer using dag"""
        outs = [x]
        for i in range(0, dag.noden):
            ins = [outs[j] for j in dag.edges[i]]
            ins = torch.mean(torch.stack(ins).float(), 0)
            x = self.conv(ins, dag.types[i], layer_id, i)
            outs.append(x)

        ins = [outs[j + 1] for j in range(dag.noden) if dag.outs[j] == 0]
        ins = torch.mean(torch.stack(ins).float(), 0)

        x = F.relu(ins)
        return x

    def conv(self, x, conv_type, layer_id, conv_id):
        """
        One conv
        0: conv3x3
        1: conv5x5
        2: maxpool
        3: meanpool
        4: identity
        """
        #print(x.size())
        if conv_type < len(self.convs):
            conv_f = self.convs[conv_type][layer_id][conv_id]
            output = conv_f(x)
            output = F.relu(output)
        else:
            output = x

        return output

    def get_num_cell_parameters(self, dag):
        num = 0

        num += models.shared_base.size(self.w_xc)
        num += models.shared_base.size(self.w_xh)

        q = collections.deque()
        q.append(0)

        while True:
            if len(q) == 0:
                break

            node_id = q.popleft()
            nodes = dag[node_id]

            for next_node in nodes:
                next_id = next_node.id
                if next_id == self.args.num_blocks:
                    assert len(nodes) == 1, 'parent of leaf node should have only one child'
                    continue

                w_h = self.w_h[node_id][next_id]
                w_c = self.w_c[node_id][next_id]

                num += models.shared_base.size(w_h)
                num += models.shared_base.size(w_c)

                q.append(next_id)

        logger.debug(f'# of cell parameters: '
                     f'{format(self.num_parameters, ",d")}')
        return num

    def reset_parameters(self):
        init_range = 0.025 if self.args.mode == 'train' else 0.04
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.fill_(0)
