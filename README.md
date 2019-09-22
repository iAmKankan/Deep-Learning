# Deep-Learning
Deep learning is a machine learning technique that teaches computers to do what comes naturally to humans: learn by example

# Activation functions and what are it uses in a Neural Network Model?
Activation functions are really important for a Artificial Neural Network to learn and make sense of something really complicated and **Non-linear complex functional mappings between the inputs and response variable**.
* Their main purpose is to convert a input signal of a node in a A-NN to an output signal.
* A Neural Network without Activation function would simply be a Linear regression Model which has limited power and does not performs good most of the times.
*  Also without activation function our Neural network would not be able to learn and model other complicated kinds of data such as images, videos , audio , speech etc

## Why do we need Non-Linearities?
* Non-linear functions are those which have degree more than one and they have a curvature when we plot a Non-Linear function. Now we need a Neural Network Model to learn and represent almost anything and any arbitrary complex function which maps inputs to outputs.
* Also another important feature of a Activation function is that it should be differentiable. We need it to be this way so as to perform backpropogation optimization strategy while propogating backwards in the network to compute gradients of Error(loss) with respect to Weights and then accordingly optimize weights using Gradient descend or any other Optimization technique to reduce Error.

## Most popular types of Activation functions -
Sigmoid or Logistic
Tanh — Hyperbolic tangent
ReLu -Rectified linear units


## [Back-propagation](https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/) 
<img src="https://github.com/iAmKankan/Deep-Learning/blob/master/Pictures/b1.png?raw=true">


### What is error?
* Error or lost is the difference between the actual value and the expected value.
* In deep Neural Net we adding up the Waights in every layer and at the end (or between-ReLu for hidden layer)we calculate all the waightes with a Activation function.
* The main perpose of back propagation is to go back in the Neural Network and modify the weights 
<img src="https://github.com/iAmKankan/Deep-Learning/blob/master/Pictures/neural_network-9.png?raw=true">

* Backpropagation is a technique used to train certain classes of neural networks – it is essentially a principal that allows the machine learning program to adjust itself according to looking at its past function.
* Backpropagation is sometimes called the “backpropagation of errors.”
* Backpropagation as a technique uses gradient descent: It calculates the gradient of the loss function at output, and distributes it back through the layers of a deep neural network. The result is adjusted weights for neurons.



