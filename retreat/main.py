import torch
import torchvision
import torchvision.transforms as transforms
from graphgen import Args, GraphGen

def LoadCifar():
    #torch.cuda.set_device(0) 
    transform = transforms.Compose([transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=400,
                                              shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=80,
                                             shuffle=False, num_workers=1)

    return trainloader, testloader
    #trainiter = iter(trainloader)
    #testiter = iter(testloader)
    #images, labels = trainiter.next()
    print(labels)

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        print(i)
        print(labels)
        return

def main():
    trainloader, testloader = LoadCifar()
    args = Args()
    graphgener = GraphGen(args, trainloader, testloader)
    #graphgener.Estim_dags(120, "bfs")
    #graphgener.Estim_dags(120, "random")
    #graphgener.Estim_dags(120, "same")
    #graphgener.SpaceTest()
    graphgener.Train_Estim()
    graphgener.Eval_Estim()

main()
