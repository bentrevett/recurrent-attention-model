# Recurrent Models of Visual Attention

Convolutional neural networks (CNNs) have their computation scale linearly with the number of pixels in the input. What if we could come up with a model that only looks at a sequence of small regions (patches) within the input image? The amount of computation is then independent on the size of the image, and dependent on the size and number of the patches extracted. This also reduces the task complexity as the model can focus on the object of interest, ignoring any surrounding clutter. This work is related to three branches of research: reducing computation in computer vision, "saliency detectors", and computer vision as a sequential decision task.

Biological inspirations: Humans do not perceive a whole scene at once, instead they focus attention on parts of the visual space to acquire information and then combine it to build an internal representation of the scene. Locations at which humans fixate have been shown to be task specific. 

### Recurrent Attention Model (RAM)

Consider the reinforcement learning (RL) scenario of a goal-directed agent interacting with a (visual) environment. At each time-step, t, the agent receives an observation - a small snapshot of the state. In this work, the state is the whole image and the observation, called a glimpse, is formed of k patches of the image, centered on location l, (x,y coordinates), which have been scaled to g x g pixels. The agent then outputs an action, a, and a location, l, where the action the predicted class of the image and the location are the x,y coordinates to use in the next glimpse at time-step, t+1. The agent's goal is to maximize the total sum of rewards over the T time-steps, noting that the reward is only receved at time-step T, and is 1 when the image is correctly classified, 0 otherwise.

There are three main components: the glimpse sensor, the glimpse network and the overall model architecture.

The glimpse sensor takes in the image, x, along with a location, l, and outputs the glimpse - k patches of the image, centered on l, where each patch is twice as big as the previous, but all are scaled to g x g pixels.

The glimpse network encapsulates the glimpse sensor. It takes in the image and location, passes them to the glimpse sensor, then feeds the output of the glimpse sensor to a linear layer, the input location to a linear layer, and then sums the output of each of those linear layers, before passing their combination through another linear layer to get g. All of these linear layers use the ReLU activation function.

The overall model takes the output of the glimpse network, g, and combines it with the recurrent hidden state from the previous time-step, h_t-1. They mention an LSTM, but the equations state the recurrent state is calculate as just summing h_t-1 and g after they have both passed through a linear layer, and then using ReLU on the output of this sum. The next location is the recurrent hidden state passed through a linear layer with 2 outputs. The next location is not just the outputs of this layer, but the outputs are used as the mean for a Gaussian distribution with a fixed variance. Note: they do not state what the fixed variance is. The action is the recurrent hidden state passed through a linear layer with C outputs, with C = number of classes.

Note: the very first glimpse location is chosen randomly, assume it's uniform between [0,1] for x,y.

We have multiple losses within our model. The first loss is the REINFORCE loss, the sum of log pobability of an action multiplied by the rewards. To reduce variance, the rewards have a baseline subtracted from them. This baseline is the output of a linear layer on h. The second loss is the baseline loss, the MSE of the output of the baseline linear layer on h and the actual rewards. The third loss is the classification loss: the cross entropy loss between the prediction and the label at time-step T.

Their experiments used between 2-8 glimpses, with patches ranging from 8x8 to 12x12 with between 1-4 "scales", which are the number of patches returned by the glimpse sensor(?). The final linear layer in the glimpse network has 256 dimensions and the 2 in the previous layer have 128. The RNN has a hidden state with 256 dimensions. They use SGD with a momentum of 0.9. 

### Experiments

They test their experiments on 4 "datasets". The first is the standard 28x28 MNIST dataset. The second is the 60x60 Translated MNIST dataset, where the 28x28 MNIST image is randomly placed in a 60x60 larger black image. The third is the 60x60 Cluttered Translated MNIST, which is the same as the Translated MNIST, but with 4 pieces of "clutter" - where clutter is a random 8x8 section of another MNIST digit. The final is the 100x100 Cluttered Translated MNIST, which is the same as the 60x60, just on a larger image patch. 

For 28x28 MNIST experiments, they find increasing the number of glimpses reduces the validation error until it saturates at 6 glimpses, where it beats a 2-layer MLP w/ 256 hidden units. For this each glimpse contains a single 8x8 patch, showing that the RAM can learn to combine information from each of the glimpses.

For the 60x60 Translated MNIST experiments, the number of glimpses reduces error until 8 glimpses. They do not show results for >8 glimpses. 6 and 8 glimpses both beat a 2-layer MLP w/ 256 hidden units and a CNN, detailed in table 1. For this, the RAM has 3 scaled 12x12 patches. This experiment shows the RAM is translation invariant and can find/focus on the digit within the image.

For both of the Cluttered Translated MNIST experiments, the RAM beats an MLP and CNN model with 4 glimpses. Error continues to decrease for up to 8 glimpses. Again, they do not show results for >8 glimpses. For this, the RAM has 3-4 scaled 12x12 patches. This shows the RAM can ignore the presence of clutter within images.