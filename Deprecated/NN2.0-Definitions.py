#DO NOT REMOVE - If you copy or use any part of this file, please note the license and warranty conditions found here: https://github.com/CerebralSeed/Neural-Splitting-with-Dice-Roll-in-Pytorch

import torch
import torch.optim as optim
import torch.nn as nn

def model_reboot(model):
    if isinstance(model, nn.Conv2d) or isinstance(model, nn.Linear): model.reset_parameters()


def roll_dice(model_class, criterion, batches, rolls, trainloader):
    best_params = []
    best_loss = 0
    model = model_class
    model.to(device)
    for epoch in range(rolls):  # test "roll" number of randomized parameter vectors

        model.apply(model_reboot)
        opts = optim.Adam(model.parameters(), lr=0.001)
        sumloss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
            # zero the parameter gradients
            opts.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            opts.step()

            sumloss += loss.item()
            if i % batches == batches - 1:
                print(str(epoch) + ' - ' + ' Checking for better parameters...')
                if best_loss == 0:
                    for layer in model.parameters():
                        best_params.append(layer)
                    print('Initializing loss from ' + str(best_loss / batches) + ' -----> ' + str(sumloss / batches))
                    best_loss = sumloss
                if best_loss > sumloss:
                    for layer, layer2 in zip(model.parameters(), best_params):
                        layer2 = layer
                    print('Updating loss from ' + str(best_loss / batches) + ' -----> ' + str(sumloss / batches))
                    best_loss = sumloss
                break
    for layer, layer2 in zip(best_params, model.parameters()):
        layer2 = layer
    return model


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
    timeelapsed=datetime.now()-starttime
    print('Accuracy of the network: ' + str(
        acc) + '%  | Time Passed: '+str(timeelapsed))

    model.train()

    return acc


def split(p, b1, w2, grad, cutoff):
    print(p.size(), b1.size(), w2.size())
    modfy = 0
    w2out = 0
    if len(p.size()) == 4:
        if len(w2.size()) == 2:
            w2out = w2.size()[0]
            w2 = w2.view(-1, p.size()[0], p.size()[2], p.size()[3])
            modfy = 1
    w2t = torch.transpose(w2, 0, 1)
    w1temp = p
    b1temp = b1
    w2ttemp = w2t
    j = 0
    neuron_total = 0
    for i, m in enumerate(grad):
        if m > torch.mean(grad)*(1+cutoff):
            k = i + j
            b = p[i].unsqueeze(0)
            c = b1[i].unsqueeze(0) * 1.0001
            d = w2t[i].unsqueeze(0)
            d = d * 0.5

            w1temp = torch.cat([w1temp[:k], b, w1temp[k:]], 0)
            b1temp = torch.cat([b1temp[:k], c, b1temp[k:]], 0)
            w2ttemp = torch.cat([w2ttemp[:k], d, w2ttemp[k:]], 0)
            w2ttemp[k + 1] = w2t[i] * 0.5
            j += 1
            neuron_total = neuron_total + 1
    w2ttemp = torch.transpose(w2ttemp, 0, 1)
    if modfy == 1:
        w2ttemp = w2ttemp.reshape(w2out, w1temp.size()[0] * w1temp.size()[2] * w1temp.size()[3])
    print(w1temp.size(), b1temp.size(), w2ttemp.size())
    print(str(neuron_total) + " neurons were split.")
    return w1temp, b1temp, w2ttemp


def sum_params(model):
    net_size = 0
    for idx in model.parameters():  # get total parameters
        net_size = net_size + torch.numel(idx)
    return net_size


def split_neurons(model, accumulated_grad, cutoff, max_params):
    twos = 0
    temp_model = []
    net_size = sum_params(model)
    if max_params < net_size:  # compare total params with max
        print("Max network size reached: " + str(net_size))
        return
    for index, layer in enumerate(model.parameters()):
        temp_model.append(layer)
    for idx, p in enumerate(temp_model):
        if idx != twos:  # use function only on weights
            continue
        if idx == len(list(model.parameters())) - 2:  # if on output layer, don't split neurons
            continue
        if temp_model[idx + 1].grad is None:  # if there are no gradients to evaluate, skip
            continue
        b1 = temp_model[idx + 1]
        w2 = temp_model[idx + 2]
        grad = accumulated_grad[idx + 1]
        new_w1, new_b1, new_w2 = split(p, b1, w2, grad, cutoff)
        temp_model[idx] = new_w1
        temp_model[idx + 1] = new_b1
        temp_model[idx + 2] = new_w2
        twos = twos + 2
    c = 0
    for attr in dir(model):
        targ = getattr(model, attr)
        if type(targ) == nn.Linear:
            setattr(model, attr, type(targ)(temp_model[c].size()[1], temp_model[c].size()[0]))
            c = c + 2
        if type(targ) == nn.Conv2d:
            setattr(model, attr, type(targ)(temp_model[c].size()[1], temp_model[c].size()[0],
                                            targ.kernel_size, targ.stride, targ.padding, targ.dilation,
                                            targ.groups, padding_mode=targ.padding_mode))
            c = c + 2
    with torch.no_grad():
        for idx, p in enumerate(model.parameters()):
            p.copy_(temp_model[idx])
    net_size2 = sum_params(model)
    print("Parameters: " + str(net_size) + " -----> " + str(net_size2))


def accumulate_gradients(model_params, accumulated_grad):
    if not accumulated_grad:  # if empty, fill with gradients
        for tensor in model_params:
            accumulated_grad.append(torch.abs(tensor.grad))
    else:  # if values present, add gradients to them
        for idx, tensor in enumerate(model_params):
            accumulated_grad[idx] = accumulated_grad[idx] + torch.abs(tensor.grad)
    return accumulated_grad


def gradients_average(accumulated_grad, len_trainset):
    inv_len = len_trainset ** -1
    # divide accumulated gradients by total number of samples
    for tensor in accumulated_grad:
        tensor = tensor * inv_len
    return accumulated_grad
