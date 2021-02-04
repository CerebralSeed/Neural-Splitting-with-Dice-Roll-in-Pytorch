This contains files related to the Limited Time Trial.
# Limited Time Trial
## Abstract
We tested the Neural Splitting with Dice Roll algorithms, allowing the network 7 minutes to acheive best results. We find that Neural Splitting with Dice Roll acheived an average of 11.93% increase in accuracy and a 17.86% decrease in loss over the control group on the CIFAR 10 dataset.

## Introduction
Often in a commercial setting, a budget for an AI training project may limit how long a network can train. By splitting neurons, we can start with a smaller/faster network and test candidate initialization parameters. And then later grow the network via the Splitting Neurons algorithm. 

## Method
1. Created a network with 3 hidden layers, two Conv2D, and a linear layer(Maxpooling2D between the two Conv2Ds). 
2. Started the network with the following neuron sizes:
Conv1 = 300 neurons
Conv2 = 25 neurons
Fc1 = 15 neurons

3. Instantiate the network with the "roll_dice" algorithm. This simply runs n number batches of the training data through the network started with random parameters and calculates total loss. It loops through this n number rolls. It then saves over the parameter set that is has the lowest loss after each "roll". This then sets your network parameters to the best candidate starting parameters found.

4. Commence training.

5. After each epoch, run the network through the neuron splitting algorithm. This algorithm targets any neurons whos bias is above the specified "cutoff" attribute. So set this lower if you want more splitting and higher for less. But be careful for setting it too high as the network might "explode" and grow faster than inference. The algorithm stops running once the number of network parameters exceeds the "max_neurons" attribute. 

6. A stop watch is set to start at the same time the network is "Run". This goes for 7 minutes. Then the training is stopped at the 7 minute marker. The last epoch that completed is then recorded for loss, accuracy, and the number of neurons in each that produced those values.

7. The above steps 1-6 were run 10 times and are the Test Group. Next we create a Control Group. The Control Group starts at the number of parameters each run in the Test Group finished at, as recorded in step 6. But the control group does not run the dice roll or neuron splitting algorithm. The Control Group runs 7 minutes and is stopped at the 7 minute marker and the last printed loss and accuracy are recorded.

Notes: In all cases, loss and accuracy were still improving at the 7 minute mark. The number of dice rolls allowed in the Test Group are recorded in each run. This amount was varied to see if there were any significant trends. At 100 rolls, the benefit was too small while the time it consumed was over 1 minute of the total time. So this was reduced to 30 rolls for the remainder of runs.  

## Results
The average accuracy of the Neural Splitting with Dice Roll test group was 11.93% higher while the average loss was 17.86% lower(lower loss corresponds to better fitting of the data). In every instance, the Test Group performed signficantly higher than the Control Group. Occasionally when training, a network starts off with a less than ideal set of parameters and either takes a lot longer to train to peak accuracy or never acheives anywhere close to the best training outcome, getting stuck in local minima. The Roll Dice algorithm solves this problem by choosing the best starting parameters from a set n, as specified by the data scientist. This reduces the likelihood of a bad start, as demonstrated in the Control Group on run #8.
Neural Splitting and Dice Roll algorithms can help to greatly increase the results given a limited budget for training. 

(See figure for complete results)

## Additional Comments
While this demonstrates how these algorithms may help researchers and companies with limited budgets obtain better results, this does not demonstrate whether these algorithms are better for someone who has an unlimited budget and time to train a network. So the next research experiment will be to demonstrate how these algorithms perform when allowed to train to a best fit vs. a control group of the same parameter size.
