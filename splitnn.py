# v1.04
#DO NOT REMOVE - If you copy or use any part of this file, please note the license and warranty conditions found here: https://github.com/CerebralSeed/Neural-Splitting-with-Dice-Roll-in-Pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime
from collections import OrderedDict


def model_reboot(model):
    if isinstance(model, nn.Conv2d) or isinstance(model, nn.Linear): model.reset_parameters()


def roll_dice(model_class, criterion, batches, trainloader, rolls=10,device=None):
    best_params = []
    best_loss = 0
    model = model_class
    for epoch in range(rolls):  # test "roll" number of randomized parameter vectors

        model.apply(model_reboot)
        sumloss = 0.0
        for i, (data,targ,delt) in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            data_input = data.to(device)
            targets = targ.to(device)

            # forward + backward + optimize
            outputs = model(data_input)

            loss = criterion(outputs, targets).to(device)

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
    del best_params, sumloss, best_loss, loss
    return model


def split(w1, b1, w2, grad, splits, cutoffadd, cutoffrem):
    modfy = 0
    w2out = 0
    if len(w1.size()) == 4: #is Conv2D layer
        if len(w2.size()) == 2: #is followed by Linear layer
            w2out = w2.size()[0] #size of Linear layer
            w2 = w2.view(-1, w1.size()[0], w1.size()[2], w1.size()[3]) #reshape linear layer
            modfy = 1
    if len(w1.size()) == 3: #is Conv1D layer
        if len(w2.size()) == 2: #is followed by linear layer
            w2out = w2.size()[0]
            w2 = w2.view(-1, w1.size()[0], w1.size()[2])
            modfy = 2
    w2ttemp = torch.transpose(w2, 0, 1)
    w1temp = w1
    b1temp = b1
    neuron_added = neuron_rem = j = 0
    mean=torch.mean(grad)
    for i, m in enumerate(grad):
        if cutoffrem > 0:
            if m < mean * (1 - cutoffrem):
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
        if m > mean * (1 + cutoffadd):
            k = i + j

            w1temp = torch.cat([w1temp[:k], torch.cat(splits*[w1temp[k].unsqueeze(0)]), w1temp[k:]], 0)
            b1temp = torch.cat([b1temp[:k], torch.cat(splits*[b1temp[k].unsqueeze(0) * (1+splits/1000)]), b1temp[k:]], 0)
            w2ttemp = torch.cat([w2ttemp[:k], torch.cat(splits*[w2ttemp[k].unsqueeze(0) * 1/(splits+1)]), w2ttemp[k:]], 0)

            w2ttemp[k + splits] = w2ttemp[k+splits] * 1/(splits+1)
            j += splits
            neuron_added += splits

    w2ttemp = torch.transpose(w2ttemp, 0, 1)
    if modfy == 1:
        w2ttemp = w2ttemp.reshape(w2out, w1temp.size()[0] * w1temp.size()[2] * w1temp.size()[3])
    if modfy == 2:
        w2ttemp = w2ttemp.reshape(w2out, w1temp.size()[0] * w1temp.size()[2])
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
    return sum(p.numel() for p in model.parameters())


def split_neurons(model, accumulated_grad, optimizer, splits=1, cutoffadd=1.5, cutoffrem=1.0, max_params=12000000, verbose=0):
    # cutoffadd: range=(0 to inf) the higher the cutoff, the less neurons will split; just be careful you don't set this too low or you might explode the network too quickly
    # cutoffadd: should be adjusted based on loss function, batch size, etc. if the network is NOT growing, try reducing it. If the network is exploding, try raising it.
    # cutoffrem 1: range=(<1 to >0), this will remove neurons based on a log normal distribution of the gradients delta for each neuron, the closer to 0, the less neurons removed. setting to 0 or 1 turns off this feature
    # cutoffrem 2(Experimental): range=(<0 to -inf), this is experimental and will remove neurons based on any in which the neuron's bias < cutoffrem.
    # A max of 30-50% parameter growth rate seems ideal. Set the cutoff as high as you can without causing the network to explode or max_params to be hit
    # max_params: this should be set to the most number of parameters your gpu can handle. This does not mean it will reach this amount, but it will stop splitting neurons if it does.
    # device: set this to device or leave empty if running on CPU
    # for Conv1D/Conv2D, make sure to use pooling layers after your final Conv before Linear to ensure the output dimensions are (out_channels, in_channels, kernel_size) (3 dims for conv1d, 4 dims for conv2d)
    # Optimizer: when resetting the optimizer, be sure to adjust the new learning rate. I.e. if it is too high for ADA, the model may struggle in the first couple of batches to find the correct rate, losing some inference
    # It's better to run this function every >=2 epochs so that initial neuron split resettling doesn't get used as a basis for later splitting. This prevents neurons getting stuck in a splitting loop.

    incrm = 0
    temp_model = []
    net_size = sum_params(model)
    if max_params < net_size:  # compare total params with max
        print("Max network size reached: " + str(net_size))
        return
    torch.cuda.empty_cache()
    for layer in model.parameters():
        temp_model.append(layer)
    has_embed = False
    layer_var = 0

    for layer, p in zip(model.children(), model.named_children()):  # for each layer in the model
        if accumulated_grad[incrm + 1] is None:  # if there are no gradients to evaluate, skip
            print(p[0]+" has no accumulated gradients to evaluate for splitting.")
            continue
        if p[0].startswith('last'):
            if verbose > 1:
                print(p[0]+" begins with 'last' and will be ignored.")
            continue

        if type(layer) == nn.Embedding:
            if len(list(model.parameters())) - 1 == incrm:
                if verbose > 1:
                    print(p[0]+" is final embedding layer and will be ignored.")
                continue
            has_embed = True
            incrm += 1
            layer_var += 1
        if type(layer) == nn.GRU:
            if len(list(model.parameters())) -4 == incrm:
                if verbose > 1:
                    print(p[0]+" is final GRU and will be ignored.")
                continue
            incrm += 4
            layer_var += 1
        if type(layer) == nn.Linear or type(layer) == nn.Conv2d or type(layer)==nn.Conv1d:
            try:
                if type(list(model.children())[layer_var + 1]) == nn.GRU:
                    if verbose > 1:
                        print(p[0]+" follows a GRU layer and will be ignored.")
                    incrm += 2
                    layer_var += 1
                    continue
            except:
                pass
            if len(list(model.parameters())) - 2 == incrm:
                if verbose>1:
                    print(p[0]+" is out layer and not evaluated.")
                continue

            if verbose>1:
                print(temp_model[incrm].size(), temp_model[incrm + 1].size(), temp_model[incrm + 2].size())
            w1 = temp_model[incrm]
            b1 = temp_model[incrm + 1]
            w2 = temp_model[incrm + 2]
            grad = accumulated_grad[incrm + 1]
            new_w1, new_b1, new_w2, nadd, nrem = split(w1, b1, w2, grad, splits, cutoffadd, cutoffrem)
            temp_model[incrm] = new_w1
            temp_model[incrm + 1] = new_b1
            temp_model[incrm + 2] = new_w2

            if verbose > 1:
                print(temp_model[incrm].size(), temp_model[incrm + 1].size(), temp_model[incrm + 2].size())
                print(p[0]+" - "+str(nadd) + ' neuron(s) were split. ' + str(nrem) + ' neuron(s) were removed.')
            layer_var += 1
            incrm += 2
    incrm = 0

    for layer, p in zip(model.children(), model.named_children()):
        try:
            dev=getattr(next(layer.parameters()),'device')
        except:
            continue
        if type(layer) == nn.Embedding:
            continue
        if type(layer) == nn.GRU:
            continue
        if type(layer) == nn.Linear:
            if str(dev) =='cpu':
                setattr(model, p[0], type(layer)(temp_model[incrm].size()[1], temp_model[incrm].size()[0]))
            else:
                setattr(model, p[0], type(layer)(temp_model[incrm].size()[1], temp_model[incrm].size()[0]).to(dev))
            incrm += 2
        if type(layer) == nn.Conv2d or type(layer)==nn.Conv1d:
            if str(dev) == 'cpu':
                setattr(model, p[0], type(layer)(temp_model[incrm].size()[1], temp_model[incrm].size()[0],
                                             layer.kernel_size, layer.stride, layer.padding, layer.dilation,
                                             layer.groups, padding_mode=layer.padding_mode))
            else:
                setattr(model, p[0], type(layer)(temp_model[incrm].size()[1], temp_model[incrm].size()[0],
                                             layer.kernel_size, layer.stride, layer.padding, layer.dilation,
                                             layer.groups, padding_mode=layer.padding_mode).to(dev))

            incrm += 2

    with torch.no_grad():
        for idx, p in enumerate(model.parameters()):
            p.copy_(temp_model[idx])
    del temp_model
    torch.cuda.empty_cache()
    if verbose>0:
        print("Parameters: " + str(net_size) + " -----> " + str(sum_params(model)))
    optimizer=reset_optimizer(model, optimizer)
    return optimizer


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

def reset_optimizer(model, optimizer):

    if optimizer.__class__ == torch.optim.Adagrad:
        steps = []
        for group in optimizer.param_groups:
            [steps.append(optimizer.state[p]['step']) for p in group['params']]

    if optimizer.__class__==torch.optim.Adadelta:
        optimizer=optimizer.__class__(model.parameters(),rho=optimizer.defaults['rho'], eps=optimizer.defaults['eps'], lr=optimizer.defaults['lr'], weight_decay=optimizer.defaults['weight_decay'])
    if optimizer.__class__==torch.optim.Adagrad:
        optimizer=optimizer.__class__(model.parameters(), lr_decay=optimizer.defaults['lr_decay'], eps=optimizer.defaults['eps'], lr=optimizer.defaults['lr'], weight_decay=optimizer.defaults['weight_decay'])
    if optimizer.__class__==torch.optim.Adam:
        optimizer=optimizer.__class__(model.parameters(),amsgrad=optimizer.defaults['amsgrad'],betas=optimizer.defaults['betas'], eps=optimizer.defaults['eps'], lr=optimizer.defaults['lr'], weight_decay=optimizer.defaults['weight_decay'])
    if optimizer.__class__==torch.optim.AdamW:
        optimizer=optimizer.__class__(model.parameters(),amsgrad=optimizer.defaults['amsgrad'],betas=optimizer.defaults['betas'], eps=optimizer.defaults['eps'], lr=optimizer.defaults['lr'], weight_decay=optimizer.defaults['weight_decay'])
    if optimizer.__class__==torch.optim.SparseAdam:
        optimizer=optimizer.__class__(model.parameters(),betas=optimizer.defaults['betas'], eps=optimizer.defaults['eps'], lr=optimizer.defaults['lr'])
    if optimizer.__class__==torch.optim.Adamax:
        optimizer=optimizer.__class__(model.parameters(),betas=optimizer.defaults['betas'], eps=optimizer.defaults['eps'], lr=optimizer.defaults['lr'], weight_decay=optimizer.defaults['weight_decay'])
    if optimizer.__class__==torch.optim.ASGD:
        optimizer=optimizer.__class__(model.parameters(),alpha=optimizer.defaults['alpha'], lambd=optimizer.defaults['lambd'], t0=optimizer.defaults['t0'], lr=optimizer.defaults['lr'], weight_decay=optimizer.defaults['weight_decay'])
    if optimizer.__class__==torch.optim.LBFGS:
        optimizer=optimizer.__class__(model.parameters(),max_iter=optimizer.defaults['max_iter'], max_eval=optimizer.defaults['max_eval'],tolerance_grad=optimizer.defaults['tolerance_grad'],tolerance_change=optimizer.defaults['tolerance_change'],history_size=optimizer.defaults['history_size'], lr=optimizer.defaults['lr'], line_search_fn=optimizer.defaults['line_search_fn'])
    if optimizer.__class__==torch.optim.RMSprop:
        optimizer=optimizer.__class__(model.parameters(),momentum=optimizer.defaults['momentum'],alpha=optimizer.defaults['alpha'],centered=optimizer.defaults['centered'], eps=optimizer.defaults['eps'], lr=optimizer.defaults['lr'], weight_decay=optimizer.defaults['weight_decay'])
    if optimizer.__class__==torch.optim.Rprop:
        optimizer=optimizer.__class__(model.parameters(),etas=optimizer.defaults['etas'], step_sizes=optimizer.defaults['step_sizes'], lr=optimizer.defaults['lr'])
    if optimizer.__class__==torch.optim.SGD:
        optimizer=optimizer.__class__(model.parameters(),momentum=optimizer.defaults['momentum'],dampening=optimizer.defaults['dampening'], nesterov=optimizer.defaults['nesterov'], lr=optimizer.defaults['lr'], weight_decay=optimizer.defaults['weight_decay'])

    optimizer.zero_grad()
    if optimizer.__class__ == torch.optim.Adagrad:
        for group in optimizer.param_groups:

            for p in group['params']:
                for i, p in enumerate(group['params']):
                    optimizer.state[p]['step'] = steps[i]

    return optimizer

