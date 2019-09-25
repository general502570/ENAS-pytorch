import torch
import random
import child
import estimator
import torch.nn as nn
import torch.nn.functional as F

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
            print("Illegal action!")
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
        self.noden = 5
        self.stepn = 5
        self.nodetypen = 5
        self.embed_h = 5
        self.lr = 0.05
        self.weight_decay = 0
        self.epochn = 1
        self.train_print = 500
        self.epoch_print = 5
        self.child_dropout = 0
        self.mode = "train"
        self.estim_train_data_graphn = 30
        self.estim_train_data_batchn = 4
        self.estim_train_data_epochn = 40
        self.estim_test_data_graphn = 30
        self.estim_test_data_batchn = 1
        self.estim_lr = 0.01
        self.estim_wd = 5e-4
        self.estim_epochn = 20
        self.embed_h1 = self.noden + 2
        self.embed_h2 = 6
        self.embed_h3 = 5
        self.device = torch.device("cuda:6")

class GraphGen():
    """
    To generate random graph
    """
    def __init__(self, args, trainloader, testloader):
        self.args = args
        self.model = child.CNN(self.args)
        print(torch.cuda.device_count())
        #self.model = nn.DataParallel(self.model)
        #self.model.cuda()
        self.model.to(self.args.device)
        self.estim = estimator.Estimator(args)
        self.estim.to(self.args.device)

        self.noden = self.args.noden
        self.stepn = self.args.stepn
        self.nodetypen = self.args.nodetypen
        self.ce = nn.CrossEntropyLoss()

        self.optimer = torch.optim.SGD(self.model.parameters(), 
            lr = self.args.lr, 
            weight_decay = self.args.weight_decay)
        self.optim_estim = torch.optim.Adam(self.estim.parameters(),
                lr = self.args.estim_lr,
                weight_decay = self.args.estim_wd)

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
        self.model.train()
        running_loss = 0.0
        running_num = 0
        for epoch in range(self.args.epochn):

            for i, data in enumerate(self.trainloader, 0):
                inputs, targets = data
                #inputs, targets = inputs.cuda(), targets.cuda() 
                #print(inputs.size())
                inputs, targets = inputs.to(self.args.device), targets.to(self.args.device) 
                outputs = self.model(inputs, dag)
                loss = self.ce(outputs, targets)
                self.optimer.zero_grad()
                loss.backward()
                self.optimer.step()

                running_loss += loss
                running_num += 1
                if i == 0 and epoch % self.args.epoch_print == 0:
                    print("epoch: %d; loss: %.3f" % (epoch, running_loss / running_num))
                    running_loss = 0.0
                    running_num = 0

    def Evaluate(self, dag):
        self.model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data
                images, labels = images.to(self.args.device), labels.to(self.args.device) 
                outputs = self.model(images, dag)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return correct / total
    
    def SpaceTest(self):
        """Test search space"""
        self.samplen = 100
        for i in range(self.samplen):
            dag = self.SampleDag()
            print(dag.edges)
            for j in range(10):
                self.Train(dag)
                acc = self.Evaluate(dag)
                print((j+1)*10, acc)
            self.model.reset_parameters()

    def Estim_data(self, inps, edgs, acs, batchn, graphn):
        for i in range(batchn):
            dags, accs = self.Estim_dags(graphn)
            inp, edg = self.estim.Clean_data(dags)
            inps.append(inp)
            edgs.append(edg)
            acs.append(accs)

    def Estim_dags(self, datan):
        """Generate training data for estimator and train"""
        print("estim dags", datan)
        dags = []
        accs = []
        for i in range(datan):
            dag = self.SampleDag()
            dags.append(dag)
            for j in range(self.args.estim_train_data_epochn):
                self.Train(dag)
            acc = self.Evaluate(dag)
            accs.append(acc)
            print(acc, dag.types, dag.edges)
            self.model.reset_parameters()
        return dags, accs

    def Train_Estim(self):
        # get train and test data
        self.train_inputs = []
        self.train_edges = []
        self.train_accss = []
        self.Estim_data(self.train_inputs, self.train_edges, self.train_accss,
                self.args.estim_train_data_batchn, self.args.estim_train_data_graphn)

        # Train
        self.estim.train()
        for i in range(self.args.estim_epochn):
            for inp, edg, acc in zip(self.train_inputs, self.train_edges, self.train_accss):
                outp = self.estim(inp, edg)
                acc = torch.Tensor(acc).to(self.args.device)
                loss_f = nn.MSELoss()
                los = loss_f(outp, acc)
                self.optim_estim.zero_grad()
                los.backward()
                self.optim_estim.step()

    def Eval_Estim(self):
        self.test_inputs = []
        self.test_edges = []
        self.test_accss = []
        self.Estim_data(self.test_inputs, self.test_edges, self.test_accss,
                self.args.estim_test_data_batchn, self.args.estim_test_data_graphn)

        self.estim.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for inp, edg, acc in zip(self.train_inputs, self.train_edges, self.train_accss):
                outp = self.estim(inp, edg)
                for target, esti in zip(acc, outp):
                    print(target, esti)
