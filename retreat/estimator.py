import torch
import torchsnooper
import random
import child
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.autograd import Variable

class Estimator(torch.nn.Module):
    """
    Use GCN to examine the performance of an architecture
    """
    def __init__(self, args):
        super(Estimator, self).__init__()
        self.args = args
        #self.encoder = torch.nn.Embedding(args.nodetypen,args.embed_h1)
        self.conv1 = GCNConv(args.embed_h1, args.embed_h2, cached = True)
        self.conv2 = GCNConv(args.embed_h2, args.embed_h3, cached = True)
        self.fc = nn.Linear(args.embed_h3 * (args.noden + 2), 1)

        self.conv1.to(self.args.device)
        self.conv2.to(self.args.device)

    def forward(self, x, edges):
        x = self.conv1(x, edges)
        x = F.relu(x)
        x = F.dropout(x, training = self.training)
        x = self.conv2(x, edges)
        x = F.relu(x)
        x = x.view(-1, self.args.embed_h3 * (self.args.noden + 2))
        x = self.fc(x)
        return x

    def Add_edge(self, n1, n2, x, y, base):
        n1.append(base + x)
        n2.append(base + y)
        n1.append(base + y)
        n2.append(base + x)

    def Embedx(self, x):
        y = [float(i == x) for i in range(self.args.embed_h1)]
        return y
        print(x, type(x))
        x = [x]
        x = torch.LongTensor(x)
        print(x.device)
        x = x.to(self.args.device)
        print(x.device)
        return x
    
    def Clean_data(self, dags):
        """Make the data useful"""
        print("clean")
        feats = []
        nodes1 = []
        nodes2 = []
        base = 0

        for dag in dags:
            feats.append(self.Embedx(self.args.nodetypen))
            for i in dag.types:
                feats.append(self.Embedx(i))
            feats.append(self.Embedx(self.args.nodetypen + 1))

            for i in range(1, self.args.noden + 1):
                for j in dag.edges[i - 1]:
                    self.Add_edge(nodes1, nodes2, i, j, base)
                if dag.outs[i - 1] == 0:
                    self.Add_edge(nodes1, nodes2, i, self.args.noden + 1, base)

            base += self.args.noden + 2

        feats = torch.Tensor(feats).to(self.args.device)
        nodes = torch.LongTensor([nodes1, nodes2]).to(self.args.device)
        return feats, nodes
