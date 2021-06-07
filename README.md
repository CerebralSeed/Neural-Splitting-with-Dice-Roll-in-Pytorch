# Neural Splitting with Dice Roll in Pytorch
## Introduction
This demonstrates a dynamic neural splitting module implemented in Pytorch* and combined with a "best of n" starting parameters. 

## License and Warranties
Use of this code for educational research or projects of a personal nature is permitted without warranty.

You can modify this code for educational or private use. Modifications or derivatives of this code for commercial use are prohibited.

Any display of this code should reference at the top an "https" link to this Github repository.

If you are interested in a license of this code for commercial - use, research and/or development, please contact me at therealjjj77"@"yahoo"."com (remove the quotes).

## Summary of Files
Example files implementing Neural Splitting with Dice Roll for CPU and GPU are in the main directory. 

NN-Definitions.py is a copy of the custom code that makes up the Neural Splitting with Dice Roll. 

Research folder contains research done on the Neural Splitting and Dice Roll algorithms. 

*I am not in any way affiliated with Pytorch or Facebook.

## Motivation for Neuron Splitting
Consider the Island Problem: we find ourselves on an island in the ocean and the only way off the island is to find the lowest point. However, there is a dense fog so we can't see very far. But we have a flat XY map of the island boundaries and can instantly teleport anywhere we desire within those boundaries. How could we quickly find the lowest point on the island? 

We could divide the island into four equal partitions and teleport to the center of each of these. Once there, we could measure altitude(loss) and slope and find the best combination of these before we begin on foot. 

However, in the problem of machine learning, we aren't just dealing with the X and Y axes. Each parameter in the network is an axis. For each axis added, the number of partitions increase by two times. Thus 2^n are the total number of partitions with n number of parameters. In order to increase our likelihood of finding a good set of starting parameters and avoiding the local minima problem, we would need a small number of total parameters. 

If we started with a smaller set of parameters, how could we go about increasing the number of neurons without changing the output? Consider the following:
![Vector Dot Product Identity](https://github.com/CerebralSeed/Neural-Splitting-with-Dice-Roll-in-Pytorch/blob/main/matrix-dot-identity2.png)
(Feel free to check this for yourself in a spreadsheet.)

Matrix multiplication, as used in neural networks, is just a multiplication of many individual vectors. If we want to split neurons without disrupting the networks progress, we need to simply clone the target biases, clone the weights before those biases, and take the weights after the target biases and clone and divide these by two. I've attached a simple example in a spreadsheet, so one can easily verify that this method works in preserving the output. Please find that here:

https://github.com/CerebralSeed/Neural-Splitting-with-Dice-Roll-in-Pytorch/blob/main/neural-network-granularity.xls

Now that we have a way to split neurons, how can we go about intelligently choosing which neurons to split? Each time we run data through our network, we accumulate losses for backpropagation. Loss tells us how far a parameter is away from the data we are feeding the network. Parameters which have seen all of the data and have a high amount of loss would seem like the perfect candidates for splitting. And that is what this algorithm does. 

## Motivation for Dice Roll
Secondly, now that we can begin with a smaller network size, how can we go about testing out random starting parameters for an ideal fit? And this is what Dice Roll does. It takes the smaller specified number of batches of training data and runs it through a random parameter set candidate, calculating the loss. These parameters get saved and it begins with a new random parameter set candidate. If the new candidate exhibits lower loss, the old candidate is tossed in favor of the new. This helps to automate avoiding a bad starting set of parameters. And since the network size is relatively small, the process can take place very quickly. 

## Getting Started
Download the file splitnn.py to your working folder. Then use import splitnn in your imports. A working example on a Conv2D net on CIFAR10 is provided in the working-example folder.  

Steps:
1. (Line 75 in sample_network_CIFAR10.py) Optional: After defining your model, call splitnn.roll_dice. This will test random starting parameters for a best fit, before training commences. You'll need to define the following: splitnn.roll_dice(model, criterion, optimizer, batches, rolls, device, trainloader). Model, criterion, optimizer, device and trainloader do not need further explanation(see Pytorch documentation). "batches" here means how many batches to run per roll(it is not batch_size). For example, if you set this to 10, then it will take 10 batches to test each roll. You'll want to set this to <1% of your total data/batch_size for quick testing. Roll is the number of attempts taken to find better parameters.
2. (Line 65 in sample_network_CIFAR10.py) Set any views inside your forward pass to dynamically get the size of the previous layer's output. i.e. getattr(self, 'ftc1').out_features
3. (Line 86 in sample_network_CIFAR10.py) Define a variable(accumulated_grad) as an empty list []. This will be used for accumulating gradients. 
4. (Line 103 in sample_network_CIFAR10.py) Define the variable from 2 with splitnn.accumulate_gradients. I.e accumulated_grad=splitnn.accumulated_gradients(model.parameters(), accumulated_grad)
5. (Line 110 in sample_network_CIFAR10.py) Define a variable with splitnn.gradients_average, calling in the variable from step 3. Pass in the number of samples in your training set. I.e. accumulated_grad = splitnn.gradients_average(accumulated_grad, len(trainset)). 
6. (Line 113 in sample_network_CIFAR10.py) Call the splitnn.split_neurons function on your model. You will call it as: splitnn.split_neurons(model, device, cutoffadd, cutoffrem, max_params). Model and device do not need further explanation. If you plotted the sum of the gradients during an epoch using a histogram, you'll see it makes a log normal distribution, to where the median is at the peak value. cutoffadd is added to 1 and multiplied by that peak value. This sets the cutoff for neurons that should split being any above this value. cutoffrem combines two functions in one, depending upon the range in which it is set. If cutoffrem is between 0 and 1, it will be subtracted from 1 and multiplied by the peak value. This means the closer to 1 it is, the less neurons removed, while the closer to 0, the more neurons removed. If cutoffrem is defined as less than 0, it will instead remove any neurons whose biases are below the value set. The latter method is highly experimental and did not show any benefit thus far. max_params defines the number of total parameters you'd like the model to stop adding any more neurons. For example, if your gpu can only handle around 350 million parameters, you may want to set this to around 320 million. It is always good, though, to save your model between epochs, just in case.
7. (Line 122 in sample_network_CIFAR10.py) Redefine your optimizer. This is necessary due to how the Dynamic Computation Graph is handled in Pytorch. Doing this between epochs has little to no impact on performance.
