## Neural Networks ##

# A Neuron is a basic unit of a neural network. It holds a number between 0 and 1. This 0 to 1 value is called an activation. 0 means not activated and
# 1 means activated think of it like a light switch 0 is off and as you go up until 1 it fully lights up.

# A Neural network is a system of neurons organized in a layer example vertical line next to each other.


## Layers of neurons ##

# Input layer - is the first layer that takes in the raw data. for e.g an image of a handwritten digit where each pixel is a neuron w/ value of 0-1.
# Hidden layer - is the middle layer. They find patterns in the data. The more hidden layers the more complex patterns the network can learn.
# Output layer - is the last layer. Gives the final answer. so for e.g each layer is a number. the one with the highest value between 0 and 1 is the ans.
# Output e.g -> Neuron 0: 0.02, Neuron 1: 0.01, Neuron 2:0.95. So out answer is 2 because the network is 95% sure it is 2.


## What is weight ##

# Every connection between a neuron has weight. A number that says how imp that connection is. High Weight = big influence, low weight= small influence.
# Negative weight shows that connection suppresses activation. Network learns by adjusting the weight.

# A bias is an extra number added to each neuron before activation. It shifts the activation up or down. Think of it as as the neuron's threshold. The
# neuron will only fire if the threshold is for e.g above 5. How the process works is you add the signal + bias and plug it into the sigmoid formula
# σ(x) = 1 / (1 + e^(-x)) -> where x is our signal+bias and then value generated will be in between 0 and 1. leaning to 0 means not firing fully and
# towards 1 means firing actively.


## Sigmoid fxn ##

# Sigmoid fxn takes any function and squished it in between 0 and 1 using the formula "σ(x) = 1 / (1 + e^(-x))". If value is very negative then
# it will be close to zero and if number is very positive it will be very close to 1. If it is zero then it will halfway activated. Think of it like a
# light dimmer switch.

## Cost function ##

# This fxn measures how wrong the network is. So the network makes the prediction, we compare it to the correct answer. The difference is the cost.


## Gradient descent ##

# Gradient descent is how the network learns. Think of cost function as the as the hilly landscape. The network starts at a random point on the hill.
# Gradient descent is the ball rolling down the hill and finding the lowest point which is known as the minimum cost.
# Steps are: - Make a prediction, -Calculate the cost aka how wrong it was, - Adjusts the weights slightly in the direction that reduced the cost. Repeat.


## Backpropagation ##

# Backpropagation is like a feedback loop. This is how the network figures out which weights to adjust and by how much. After each prediction the error
# travels backwards through the network from the output layer to the input layer adjusting the weight along the way.


## Learning rate ##

# It is a hyperparameter that we tune. Think of it like the size of your steps while walking downhill. It controls how big your steps are
# when rolling down the hill for e.g. 0.1 is a large step, 0.01 is a medium step and 0.001 is a small step. There is no max value for learning rate
# but in general it anything above 0.1 is usually too large and the ball overshoots the minimum, the model will diverge instead of converge.


## SUMMARY STEPS ##

# 1- Feed data into the input layer.
# 2- Activations flow forward through the hidden layers.
# 3- Output layer makes a prediction.
# 4- Cost fxn measures how wrong it was
# 5- Backpropagation sends the error backwards.
# 6- Gradient descent adjusts the weight
# 7- This process is repeated thousands of times and the network gets better.
