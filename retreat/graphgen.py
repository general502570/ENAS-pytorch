import torch
import random
import child
import torch.nn as nn

class Dag():
    """
    Representation of a DAG
    """
    def __init__(self, noden, nodetypen):
        self.noden = noden
        self.nodetypen = nodetypen
        self.types = []
        self.edges = []
        # out edge number
        self.outs = []
        for i in range(noden):
            self.types.append(0)
        for i in range(noden):
            self.edges.append([])
        for i in range(noden):
            self.outs.append(0)

    def Action(self, action):
        if action > self.noden * (self.nodetypen - 1) + self.noden * (self.noden - 1) / 2:
            print("Illegal action!\n")
            return
        if action == 0:
            return
        if action <= self.noden * (self.nodetypen - 1):
            nod = (action - 1) // (self.nodetypen - 1)
            typ = (action - 1) % (self.nodetypen - 1)
            if typ < self.types[nod]:
                self.types[nod] = typ
            else:
                self.types[nod] = typ + 1
        else:
            action -= self.noden * (self.nodetypen - 1) + 1
            for i in range(1, self.noden):
                if action < i:
                    # switch action--i
                    if action in self.edges[i]:
                        self.edges[i].remove(action)
                        self.outs[action] -= 1
                    else:
                        self.edges[i].append(action)
                        self.outs[action] += 1
                    break
                action -= i

    def InitChain(self):
        for i in range(self.noden):
            self.edges[i] = [i]
            self.outs[i] = 1 if i < self.noden - 1 else 0

class Args():
    def __init__(self):
        self.layern = 3
        self.channels = [3, 6, 12]
        self.noden = 10
        self.stepn = 5
        self.nodetypen = 5
        self.lr = 0.8
        self.weight_decay = 0#1e-7
        self.epochn = 200
        self.train_print = 500
        self.mode = "train"

class GraphGen():
    """
    To generate random graph
    """
    def __init__(self, args, trainloader, testloader):
        self.args = args
        self.model = child.CNN(self.args)
        self.model.cuda()
        self.noden = self.args.noden
        self.stepn = self.args.stepn
        self.nodetypen = self.args.nodetypen
        self.ce = nn.CrossEntropyLoss()

        self.optimer = torch.optim.SGD(self.model.parameters(), 
            lr = self.args.lr, 
            weight_decay = self.args.weight_decay)

        self.trainloader = trainloader
        self.testloader = testloader

    def SampleDag(self):
        """Give one random dag with $stepn steps"""
        dag = Dag(self.noden, self.nodetypen)
        dag.InitChain()
        actionn = self.noden * (self.nodetypen - 1) + self.noden * (self.noden - 1) / 2
        for i in range(self.stepn):
            action = random.randint(0, actionn - 1)
            dag.Action(action)
        return dag

    def Train(self, dag):
        """Train Weight For a certain dag"""
        for epoch in range(self.args.epochn):
            running_loss = 0.0

            for i, data in enumerate(self.trainloader, 0):
                inputs, targets = data
                inputs, targets = inputs.cuda(), targets.cuda() 
                outputs = self.model(inputs, dag)
                loss = self.ce(outputs, targets)
                self.optimer.zero_grad()
                loss.backward()
                self.optimer.step()

                running_loss += loss
                if (i + 1) % self.args.train_print == 0:
                    print("epoch: %d; loss: %.3f" % (epoch, running_loss / self.args.train_print))
                    running_loss = 0.0

    def Evaluate(self, dag):
        return 0
    
    def SpaceTest(self):
        """Test search space"""
        self.samplen = 100
        for i in range(self.samplen):
            dag = self.SampleDag()
            print(dag.edges)
            self.Train(dag)
            print(acc)
