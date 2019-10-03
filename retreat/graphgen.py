import torch
import random
import child
import estimator
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import pickle
import copy

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
        self.stepn = 7
        self.nodetypen = 5
        self.embed_h = 5
        self.lr = 0.05
        self.weight_decay = 0
        self.epochn = 200#40
        self.earlyst_min_epoch = 40
        self.train_print = 500
        self.epoch_print = 10
        self.child_dropout = 0
        self.mode = "train"
        self.estim_train_data_graphn = 30
        self.estim_train_data_batchn = 4
        self.estim_train_data_epochn = 200
        self.estim_test_data_graphn = 30
        self.estim_test_data_batchn = 1
        self.estim_lr = 0.1
        self.estim_wd = 5e-4
        self.estim_epochn = 10
        self.embed_h1 = self.noden + 2
        self.embed_h2 = 6
        self.embed_h3 = 5
        self.device = torch.device("cuda:9")

class GraphGen():
    """
    To generate random graph
    """
    def __init__(self, args, trainloader, testloader):
        self.args = args
        self.bfsdags = []
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

        '''self.optimer = torch.optim.SGD(self.model.parameters(), 
            lr = self.args.lr, 
            weight_decay = self.args.weight_decay)'''
        self.optimer = torch.optim.RMSprop(self.model.parameters())
        #self.optimer = torch.optim.Adam(self.model.parameters())
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

    def SampleDagBFS(self):
        """Give one random dag with $stepn steps"""
        if not self.bfsdags:
            dag = Dag(self.noden, self.nodetypen)
            dag.InitChain()
            actionn = self.noden * (self.nodetypen - 1) + self.noden * (self.noden - 1) / 2
            for i in range(self.stepn):
                action = random.randint(0, actionn - 1)
                dag.Action(action)
            self.bfsdags.append(dag)
        else:
            olddag = self.bfsdags[0]
            dag = copy.deepcopy(olddag)
            actionn = self.noden * (self.nodetypen - 1) + self.noden * (self.noden - 1) / 2
            action = random.randint(0, actionn - 1)
            dag.Action(action)
            del self.bfsdags[0]
            self.bfsdags.append(olddag)
            self.bfsdags.append(dag)
        return dag

    def Train(self, dag, epochn, earlyst = 0):
        """Train Weight For a certain dag"""
        self.model.train()
        running_loss = 0.0
        running_num = 0
        loses = []
        for epoch in range(epochn):

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

            aloss = running_loss / running_num
            if earlyst and epoch > self.args.earlyst_min_epoch:
                ploss = sum(loses[-earlyst:]) / earlyst
                if ploss < aloss:
                    return aloss
            loses.append(aloss)

            running_loss = 0.0
            running_num = 0

        return aloss

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
    
    def Summary(self, name, li, num):
        for i in range(num, len(li)):
            pre = round(float(sum(li[i - num : i]) / num), 4)
            print(name, i, pre, float(li[i]), "#####" if pre>li[i] else "#")
    
    def SpaceTest(self):
        """Test search space"""
        samplen = 5
        loses = []
        accs = []
        #writer = SummaryWriter()
        for i in range(samplen):
            dag = self.SampleDag()
            #print(dag.edges)
            for j in range(200):
                los = self.Train(dag, 1)
                acc = self.Evaluate(dag)
                #writer.add_scalar("scalar/los", los, j)
                #writer.add_scalar("scalar/acc", acc, j)
                loses.append(los)
                accs.append(acc)
                #print((j+1)*10, acc)
            self.Summary("loss20", loses, 20)
            self.Summary("loss10", loses, 10)
            self.Summary("acc20", accs, 20)
            self.Summary("acc10", accs, 10)
            self.model.reset_parameters()
            loses = []
            accs = []
        
        #writer.close()

    def Load_data(self, inps, edgs, acs, fname, offset, graphn, batchn):
        f = open("data_" + fname + ".txt", "rb")
        datan = pickle.load(f)
        dags = pickle.load(f)
        accs = pickle.load(f)
        for i in range(batchn):
            roffset = offset + i * graphn
            bdags = dags[roffset : roffset + graphn]
            baccs = accs[roffset : roffset + graphn]
            inp, edg = self.estim.Clean_data(bdags)
            inps.append(inp)
            edgs.append(edg)
            acs.append(baccs)
        f.close()

    def Estim_data(self, inps, edgs, acs, batchn, graphn):
        for i in range(batchn):
            dags, accs = self.Estim_dags(graphn)
            inp, edg = self.estim.Clean_data(dags)
            inps.append(inp)
            edgs.append(edg)
            acs.append(accs)

    def Estim_dags(self, datan, sampletype = "random"):
        """Generate training data for estimator and train"""
        dags = []
        accs = []
        dag = self.SampleDag()
        for i in range(datan):
            if sampletype == "random":
                dag = self.SampleDag()
            elif sampletype == "bfs":
                dag = self.SampleDagBFS()
            dags.append(dag)
            self.Train(dag, self.args.estim_train_data_epochn, 10)
            acc = self.Evaluate(dag)
            accs.append(acc)
            print(acc, dag.types, dag.edges)
            self.model.reset_parameters()

        return
            
        # pickle
        f = open("data_" + str(sampletype) + ".txt", "wb")
        pickle.dump(datan, f)
        pickle.dump(dags, f)
        pickle.dump(accs, f)
        return dags, accs

    def Train_Estim(self):
        # get train and test data
        self.train_inputs = []
        self.train_edges = []
        self.train_accss = []
        #self.Estim_data(self.train_inputs, self.train_edges, self.train_accss,
        #        self.args.estim_train_data_batchn, self.args.estim_train_data_graphn)
        self.Load_data(self.train_inputs, self.train_edges, self.train_accss, "bfs", 20, 20, 5)
        self.Load_data(self.train_inputs, self.train_edges, self.train_accss, "random", 20, 20, 5)
        self.test_inputs = []
        self.test_edges = []
        self.test_accss = []
        #self.Estim_data(self.test_inputs, self.test_edges, self.test_accss,
        #        self.args.estim_test_data_batchn, self.args.estim_test_data_graphn)
        self.Load_data(self.test_inputs, self.test_edges, self.test_accss, "bfs", 0, 20, 1)
        self.Load_data(self.test_inputs, self.test_edges, self.test_accss, "random", 0, 20, 1)

        # Train
        self.estim.train()
        for i in range(self.args.estim_epochn):
            losa = 0
            for inp, edg, acc in zip(self.train_inputs, self.train_edges, self.train_accss):
                outp = self.estim(inp, edg)
                a = outp[0].data
                a.tolist()
                acc = torch.Tensor(acc).to(self.args.device)
                loss_f = nn.MSELoss()
                los = loss_f(outp, acc)
                losa += los
                self.optim_estim.zero_grad()
                los.backward()
                self.optim_estim.step()

            print(i, losa)

    def Eval_Estim(self):

        self.estim.eval()
        total = 0
        correct = 0

        for inp, edg, acc in zip(self.train_inputs, self.train_edges, self.train_accss):
            #for111 inp, edg, acc in zip(self.test_inputs, self.test_edges, self.test_accss):
            outp = self.estim(inp, edg)
            #a = outp[0].data
            #a.tolist()
            for target, esti in zip(acc, outp):
                print(target, esti)
