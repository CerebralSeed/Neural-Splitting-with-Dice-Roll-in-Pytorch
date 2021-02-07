This contains files related to the Unlimited Time Trial.
# Limited Time Trial
## Abstract
We tested the Neural Splitting with Dice Roll algorithms, allowing the network acheive peak validation accuracy. We find that Neural Splitting with Dice Roll acheived an average of 4.79% increase in accuracy and a 12.01% decrease in loss over the control group on the CIFAR 10 dataset. The best results were acheived on average in 47.25% less time than the control group. 

## Introduction
In some settings, a company or organization may desire the best outcome at any cost. This study seeks to demonstrate benefits, if any, to splitting neurons with dice roll.  

## Method
1. Create a network with 3 hidden layers, two Conv2D, and a linear layer(Maxpooling2D between the two Conv2Ds). 
2. Start the network with the following neuron sizes:
Conv1 = 300 neurons
Conv2 = 25 neurons
Fc1 = 15 neurons

3. Instantiate the network with the "roll_dice" algorithm. This simply runs n number batches of the training data through the network started with random parameters and calculates total loss. It loops through this n number rolls. It then saves over the parameter set that is has the lowest loss after each "roll". This then sets your network parameters to the best candidate starting parameters found.

4. Commence training.

5. After each epoch, run the network through the neuron splitting algorithm. This algorithm targets any neurons whose bias is above the specified "cutoff" attribute. So set this lower if you want more splitting and higher for less. But be careful for setting it too high as the network might "explode" and grow faster than inference. The algorithm stops running once the number of network parameters exceeds the "max_neurons" attribute. 

6. Run training until a maximum validation accuracy is reached. Keep running the training for twice the time to see if a better validation accuracy develops. The epoch with the best validation accuracy is then recorded for time, loss, accuracy, and the number of neurons in each that produced those values.

7. The above steps 1-6 were run 10 times and are the Test Group. Next we create a Control Group. The Control Group starts at the number of parameters each run in the Test Group finished at, as recorded in step 6. But the control group does not run the dice roll or neuron splitting algorithm. The Control Group runs until acheiving the best validation accuracy and has the time, loss and accuracy recorded.

## Results
The average accuracy of the Neural Splitting with Dice Roll test group was 4.79% higher while the average loss was 12.01% lower(lower loss corresponds to better fitting of the data). This was acheived in 47.25% less time, on average - nearly half. In every instance, the Test Group performed higher than the Control Group. Neural Splitting and Dice Roll algorithms can help to greatly increase the results given a limited budget for training. 

(See Results Table below for complete results)


![Results Table](https://github.com/CerebralSeed/Neural-Splitting-with-Dice-Roll-in-Pytorch/blob/main/Research/Unlimited%20Time%20Trial/results.jpg) 

Test Group Code: https://github.com/CerebralSeed/Neural-Splitting-with-Dice-Roll-in-Pytorch/blob/main/Research/Unlimited%20Time%20Trial/main-wgpu.py

Control Group Code: https://github.com/CerebralSeed/Neural-Splitting-with-Dice-Roll-in-Pytorch/blob/main/Research/Unlimited%20Time%20Trial/main-wgpu-cont.py


## Additional Comments
While this demonstrates how these algorithms may help researchers and companies acheive better results with neuron splitting with dice roll, tests need to be performed on smaller network sizes more appropriate to the CIFAR10 dataset. Those will be next.
During the training, we noticed that the number of neurons in the first hidden layer were much more likely to split. Given this, we reviewed the accumulated gradient function and found that the magnitude of the losses should be accumulated and not their raw value. After making this adjustment, we performed a histogram of the adjusted values and found this conforms closely to a log normal distribution. Due to this, we adjusted the function targeting which neurons to split. Going forward, neurons will be split based on the mean of the average of the magnitudes, multiplied by (1+cutoff). This change results in much quicker convergence on best validation accuracy and lowest loss. This updated function will be used in the next test evaluating smaller networks.  


