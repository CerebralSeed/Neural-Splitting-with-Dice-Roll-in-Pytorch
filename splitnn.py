# v1.034
#DO NOT REMOVE - If you copy or use any part of this file, please note the license and warranty conditions found here: https://github.com/CerebralSeed/Neural-Splitting-with-Dice-Roll-in-Pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime


def model_reboot(model):
    if isinstance(model, nn.Conv2d) or isinstance(model, nn.Linear): model.reset_parameters()


def roll_dice(model_class, criterion, optim, batches, trainloader, rolls=10):
    best_params = []
    best_loss = 0
    model = model_class
    for epoch in range(rolls):  # test "roll" number of randomized parameter vectors

        model.apply(model_reboot)
        opts = optim
        sumloss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0], data[1]
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


def split(w1, b1, w2, grad, cutoffadd, cutoffrem):
    modfy = 0
    w2out = 0
    if len(w1.size()) == 4:
        if len(w2.size()) == 2:
            w2out = w2.size()[0]
            w2 = w2.view(-1, w1.size()[0], w1.size()[2], w1.size()[3])
            modfy = 1
    w2t = torch.transpose(w2, 0, 1)
    w1temp = w1
    b1temp = b1
    w2ttemp = w2t
    neuron_added = neuron_rem = j = 0
    for i, m in enumerate(grad):
        if cutoffrem > 0:
            if m < torch.mean(grad) * (1 - cutoffrem):
                k = i + j

                w1temp = torch.cat([w1temp[:k], w1temp[k + 1:]])
                b1temp = torch.cat([b1temp[:k], b1temp[k + 1:]])
                w2ttemp = torch.cat([w2ttemp[:k], w2ttemp[k + 1:]])
                j -= 1
                neuron_rem += 1
        if cutoffrem < 0:
            k = i + j
            if b1temp[k] < cutoffrem:
                w1temp = torch.cat([w1temp[:k], w1temp[k + 1:]])
                b1temp = torch.cat([b1temp[:k], b1temp[k + 1:]])
                w2ttemp = torch.cat([w2ttemp[:k], w2ttemp[k + 1:]])
                j -= 1
                neuron_rem += 1
        if m > torch.mean(grad) * (1 + cutoffadd):
            k = i + j
            w1temp = torch.cat([w1temp[:k], w1temp[k].unsqueeze(0), w1temp[k:]], 0)
            b1temp = torch.cat([b1temp[:k], b1temp[k].unsqueeze(0) * 1.0001, b1temp[k:]], 0)
            w2ttemp = torch.cat([w2ttemp[:k], w2ttemp[k].unsqueeze(0) * 0.5, w2ttemp[k:]], 0)
            w2ttemp[k + 1] = w2t[i] * 0.5
            j += 1
            neuron_added += 1

    w2ttemp = torch.transpose(w2ttemp, 0, 1)
    if modfy == 1:
        w2ttemp = w2ttemp.reshape(w2out, w1temp.size()[0] * w1temp.size()[2] * w1temp.size()[3])
    return w1temp, b1temp, w2ttemp, neuron_added, neuron_rem


def gru_remove(k, hidden_size, linbtemp, linwtemp, has_embed, embedt, w1itemp, w1htemp, b1itemp, b1htemp, w2ttemp, grad_fin):
    # removes previous linear layer corresponding row
    linbtemp = torch.cat([linbtemp[:k], linbtemp[k + 1:]], 0)
    linwtemp = torch.cat([linwtemp[:k], linwtemp[k + 1:]], 0)
    # remove embedding layer, if exists before linear layer
    if has_embed:
        embedt = torch.cat([embedt[:k], embedt[k + 1:]], 0)
    # remove row-wise corresponding elements in weights and biases
    for iter in range(3):
        w1itemp = torch.cat(
            [w1itemp[:k + hidden_size * iter - iter], w1itemp[k + hidden_size * iter - iter + 1:]], 0)
        w1htemp = torch.cat(
            [w1htemp[:k + hidden_size * iter - iter], w1htemp[k + hidden_size * iter - iter + 1:]], 0)
        b1itemp = torch.cat(
            [b1itemp[:k + hidden_size * iter - iter], b1itemp[k + hidden_size * iter - iter + 1:]], 0)
        b1htemp = torch.cat(
            [b1htemp[:k + hidden_size * iter - iter], b1htemp[k + hidden_size * iter - iter + 1:]], 0)
    # remove column-wise weight corresponding elements
    w1itemp = torch.cat((w1itemp[:, :k], w1itemp[:, k + 1:]), 1)
    w1htemp = torch.cat((w1htemp[:, :k], w1htemp[:, k + 1:]), 1)
    # remove next layer's corresponding weights
    w2ttemp = torch.cat([w2ttemp[:k], w2ttemp[k + 1:]], 0)
    # remove corresponding k-th gradient
    grad_fin = torch.cat([grad_fin[:k], grad_fin[k + 1:]], 0)


    return linbtemp, linwtemp, embedt, w1itemp, w1htemp, b1itemp, b1htemp, w2ttemp, grad_fin


def splitgru(embed, linw, linb, w1i, w1h, b1i, b1h, w2, hidden_size, grad, cutoffadd, cutoffrem, has_embed, device):
    w2t = torch.transpose(w2, 0, 1)
    embedt=[]
    if has_embed:
        embedt = torch.transpose(embed, 0, 1)
    # split vectors, a copy for appending and a reference vector
    linwtemp = linw
    linbtemp = linb
    w1itemp = w1i
    w1htemp = w1h
    b1itemp = b1i
    b1htemp = b1h
    w2ttemp = w2t
    # average z+r biases gradients
    grad_fin = torch.empty(hidden_size).to(device)
    for idx, bias in enumerate(grad):
        if idx < hidden_size:
            grad_fin[idx] = (bias + grad[hidden_size + idx]) * 0.5
    neuron_added = neuron_rem = j = 0
    for i, m in enumerate(grad_fin):
        if cutoffrem > 0:
            if m < torch.median(grad_fin) * (1 - cutoffrem):
                k = i + j
                linbtemp, linwtemp, embedt, w1itemp, w1htemp, b1itemp, b1htemp, w2ttemp, grad_fin=\
                    gru_remove(k, hidden_size, linbtemp, linwtemp, has_embed, embedt, w1itemp, w1htemp, b1itemp,
                               b1htemp, w2ttemp, grad_fin)
                j -= 1
                neuron_rem += 1
                hidden_size -= 1
        if cutoffrem < 0:
            k = i + j
            if torch.median(w1htemp[k]) + torch.median(b1itemp[k]) < cutoffrem:
                linbtemp, linwtemp, embedt, w1itemp, w1htemp, b1itemp, b1htemp, w2ttemp, grad_fin = \
                    gru_remove(k, hidden_size, linbtemp, linwtemp, has_embed, embedt, w1itemp, w1htemp, b1itemp,
                               b1htemp, w2ttemp, grad_fin)
                j -= 1
                neuron_rem += 1
                hidden_size -= 1
    for i, m in enumerate(grad_fin):
        if m > torch.median(grad_fin) * (1 + cutoffadd):
            k = i + j
            if has_embed == True:
                embedt = torch.cat([embedt[:k], embedt[k].unsqueeze(0), embedt[k:]], 0)
            linbtemp = torch.cat([linbtemp[:k], linbtemp[k].unsqueeze(0), linbtemp[k:]], 0)
            linwtemp = torch.cat([linwtemp[:k], linwtemp[k].unsqueeze(0), linwtemp[k:]], 0)
            for iter in range(3):
                w1itemp = torch.cat(
                    [w1itemp[:k + hidden_size * iter + iter], w1itemp[k + hidden_size * iter].unsqueeze(0),
                     w1itemp[k + hidden_size * iter + iter:]], 0)
                w1htemp = torch.cat(
                    [w1htemp[:k + hidden_size * iter + iter], w1htemp[k + hidden_size * iter].unsqueeze(0),
                     w1htemp[k + hidden_size * iter + iter:]], 0)
                b1itemp = torch.cat(
                    [b1itemp[:k + hidden_size * iter + iter], b1itemp[k + hidden_size * iter].unsqueeze(0) * 1.0001,
                     b1itemp[k + hidden_size * iter + iter:]], 0)
                b1htemp = torch.cat(
                    [b1htemp[:k + hidden_size * iter + iter], b1htemp[k + hidden_size * iter].unsqueeze(0) * 1.0001,
                     b1htemp[k + hidden_size * iter + iter:]], 0)
            zeros = torch.zeros(w1htemp.size()[0], 1, device=device)
            w1itemp = torch.cat([w1itemp, zeros], 1)
            w1htemp = torch.cat([w1htemp, zeros], 1)
            f = w2t[i].unsqueeze(0)
            f = f * 0.5
            w2ttemp = torch.cat([w2ttemp[:k], f, w2ttemp[k:]], 0)
            w2ttemp[k + 1] = w2ttemp[k + 1] * 0.5
            j += 1
            neuron_added += 1
            hidden_size += 1

    if has_embed == True:
        embed = torch.transpose(embedt, 0, 1)
    w2ttemp = torch.transpose(w2ttemp, 0, 1)
    return embed, linwtemp, linbtemp, w1itemp, w1htemp, b1itemp, b1htemp, w2ttemp, neuron_added, neuron_rem


def sum_params(model):
    net_size = 0
    for idx in model.parameters():  # get total parameters
        net_size = net_size + torch.numel(idx)
    return net_size


def split_neurons(model, accumulated_grad, device=None, cutoffadd=1.5, cutoffrem=1.0, max_params=12000000):
    # cutoffadd: range=(0 to inf) the higher the cutoff, the less neurons will split; just be careful you don't set this too low or you might explode the network too quickly
    # cutoffadd: should be adjusted based on loss function, batch size, etc. if the network is NOT growing, try reducing it. If the network is exploding, try raising it.
    # cutoffrem 1: range=(<1 to >0), this will remove neurons based on a log normal distribution of the gradients delta for each neuron, the closer to 0, the less neurons removed. setting to 0 or 1 turns off this feature
    # cutoffrem 2(Experimental): range=(<0 to -inf), this is experimental and will remove neurons based on any in which the neuron's bias < cutoffrem.
    # A max of 30-50% parameter growth rate seems ideal. Set the cutoff as high as you can without causing the network to explode or max_params to be hit
    # max_params: this should be set to the most number of parameters your gpu can handle. This does not mean it will reach this amount, but it will stop splitting neurons if it does.
    # device: set this to device or leave empty if running on CPU
    if not device:
        device = 'cpu'
    incrm = 0
    temp_model = []
    net_size = sum_params(model)
    if max_params < net_size:  # compare total params with max
        print("Max network size reached: " + str(net_size))
        return
    for layer in model.parameters():
        temp_model.append(layer)
    has_embed = False
    layer_var = 0
    for layer, p in zip(model.children(), model.named_children()):  # for each layer in the model
        print(p)
        if temp_model[incrm + 1].grad is None:  # if there are no gradients to evaluate, skip
            continue
        if p[0].startswith('last'):
            continue
        if type(layer) == nn.Embedding:
            if len(list(model.parameters())) - 1 == incrm:
                continue
            has_embed = True
            incrm += 1
            layer_var += 1
        if type(layer) == nn.GRU:
            if len(list(model.parameters())) -4 == incrm:
                continue
            incrm += 4
            layer_var += 1
        if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
            try:
                if type(list(model.children())[layer_var + 1]) == nn.GRU:
                    incrm += 2
                    layer_var += 1
                    continue
            except:
                pass
            if len(list(model.parameters())) - 2 == incrm:
                continue
            print(temp_model[incrm].size(), temp_model[incrm + 1].size(), temp_model[incrm + 2].size())
            w1 = temp_model[incrm]
            b1 = temp_model[incrm + 1]
            w2 = temp_model[incrm + 2]
            grad = accumulated_grad[incrm + 1]
            new_w1, new_b1, new_w2, nadd, nrem = split(w1, b1, w2, grad, cutoffadd, cutoffrem)
            temp_model[incrm] = new_w1
            temp_model[incrm + 1] = new_b1
            temp_model[incrm + 2] = new_w2
            print(temp_model[incrm].size(), temp_model[incrm + 1].size(), temp_model[incrm + 2].size())
            print(str(nadd) + ' neuron(s) were split. ' + str(nrem) + ' neuron(s) were removed.')
            layer_var += 1
            incrm += 2
    incrm = 0
    with torch.cuda.device_of(temp_model[0]):

        for layer, p in zip(model.children(), model.named_children()):
            if type(layer) == nn.Embedding:
                continue
            if type(layer) == nn.GRU:
                continue
            if type(layer) == nn.Linear:
                setattr(model, p[0], type(layer)(temp_model[incrm].size()[1], temp_model[incrm].size()[0]).cuda(device))
                incrm += 2
            if type(layer) == nn.Conv2d:
                setattr(model, p[0], type(layer)(temp_model[incrm].size()[1], temp_model[incrm].size()[0],
                                                 layer.kernel_size, layer.stride, layer.padding, layer.dilation,
                                                 layer.groups, padding_mode=layer.padding_mode).cuda(device))
                incrm += 2
    model.cuda(device)

    with torch.no_grad():
        for idx, p in enumerate(model.parameters()):
            p.copy_(temp_model[idx])

    print("Parameters: " + str(net_size) + " -----> " + str(sum_params(model)))


def accumulate_gradients(model_params, accumulated_grad):
    if not accumulated_grad:  # if empty, fill with gradients
        for tensor in model_params:
            if tensor.grad == None:
                continue
            accumulated_grad.append(torch.abs(tensor.grad))
    else:  # if values present, add gradients to them
        for idx, tensor in enumerate(model_params):
            if tensor.grad == None:
                continue
            accumulated_grad[idx] = accumulated_grad[idx] + torch.abs(tensor.grad)
    return accumulated_grad


def gradients_average(accumulated_grad, len_trainset):
    inv_len = len_trainset ** -1
    # divide accumulated gradients by total number of samples
    for tensor in accumulated_grad:
        tensor = tensor * inv_len
    return accumulated_grad


def weight_dec(epoch):
    dec = (epoch + 3) ** -1
    return dec
