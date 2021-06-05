import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
import splitnn as sn

starttime = datetime.now()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def test_data(model, testloader):
    with torch.no_grad():  # check accuracy on test set
        model.eval()
        ttotal = 0
        tcorrect = 0
        for tdata in testloader:
            timages, tlabels = tdata[0].to(device), tdata[1].to(device)
            toutputs = model(timages)
            _, predicted = torch.max(toutputs.data, 1)
            ttotal += tlabels.size(0)
            tcorrect += (predicted == tlabels).sum().item()
    acc = 100.00 * tcorrect / ttotal
    print('Accuracy of the network: ' + str(
        acc) + '%')

    model.train()

    return acc


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.fc1 = nn.Linear(20 * 5 * 5, 30)
        self.fc2 = nn.Linear(30, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, getattr(self, 'conv2').out_channels * getattr(self, 'conv2').kernel_size[0] *
                   getattr(self, 'conv2').kernel_size[1])
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


criterion = nn.CrossEntropyLoss()
net = Net()
optimizer = optim.Adam(net.parameters(), lr=0.001)

net = sn.roll_dice(net, criterion, optim=optimizer, batches=10, rolls=30, trainloader=trainloader)
# Optional: gets best random starting parameters - avoid bad starts
# batches specifies the number of batches to test, set to an amount 1% or less of total samples, shuffling data is recommended
# rolls is how many rolls to try, set between 10 and 100. Each additional roll decreases the probability of finding better starting parameters
# optimizer type should be specified

net.to(device)

for epoch in range(15):  # loop over the dataset multiple times

    running_loss = 0.0
    accumulated_grad = []
    # accumulated_grad must be defined before training loop begins
    net.to(device)
    totali = 0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        accumulated_grad = sn.accumulate_gradients(net.parameters(), accumulated_grad)
        # accumulate_gradients must be called in the training loop
        totali = totali + 1
    print('Epoch ' + str(epoch) + ' loss: ' + str(running_loss / totali))

    test_data(net, testloader)

    accumulated_grad = sn.gradients_average(accumulated_grad, len(trainset))
    # gradients_average must be called before split_neurons and between epochs

    sn.split_neurons(net, accumulated_grad, device=device, cutoffadd=0.3, cutoffrem=0.8, max_params=12000000)
    # cutoffadd: range=(0 to inf) the higher the cutoff, the less neurons will split; just be careful you don't set this too low or you might explode the network too quickly
    # cutoffadd: should be adjusted based on loss function, batch size, etc. if the network is NOT growing, try decreasing it. If the network is exploding, try raising it.
    # cutoffrem 1: range=(<1 to >0), this will remove neurons based on a log normal distribution of the gradients delta for each neuron, the closer to 0, the more neurons removed. setting to 0 or 1 turns off this feature
    # cutoffrem 2(Experimental): range=(<0 to -inf), this is experimental and will remove neurons based on any in which the neuron's bias < cutoffrem.
    # A max of 30-50% parameter growth rate seems ideal. Set the cutoff as high as you can without causing the network to explode or max_params to be hit
    # max_params: this should be set to the most number of parameters your gpu can handle. This does not mean it will reach this amount, but it will stop splitting neurons if it does.
    # device: set this to device or leave empty if running on CPU

    optimizer = optim.Adam(net.parameters(), lr=0.001)
finishtime = datetime.now()
ttime = finishtime - starttime
print('Finished Training in ' + str(ttime) + '.')
