## Neural Network Basics

* Neural networks were one of the first machine learning models.
* Deep learning implies the use of neural networks.  
* The **"deep"** in deep learning refers to a neural network with many hidden layers.  

* Neural networks accept input and produce output.  
    * The input to a neural network is called the feature vector.  
    * The size of this vector is always a fixed length.  
    * Changing the size of the feature vector means recreating the entire neural network.  
    * A vector implies a 1D array.  Historically the input to a neural network was always 1D.  
    * However, with modern neural networks you might see inputs, such as:-

* **1D Vector** - Classic input to a neural network, similar to rows in a spreadsheet.  Common in predictive modeling.
* **2D Matrix** - Grayscale image input to a convolutional neural network (CNN).
* **3D Matrix** - Color image input to a convolutional neural network (CNN).
* **nD Matrix** - Higher order input to a CNN.

Prior to CNN's, the image input was sent to a neural network simply by squashing the image matrix into a long array by placing the image's rows side-by-side.  CNNs are different, as the nD matrix literally passes through the neural network layers.


**Dimensions** The term dimension can be confusing in neural networks.  In the sense of a 1D input vector, dimension refers to how many elements are in that 1D array.  
* For example a neural network with 10 input neurons has 10 dimensions.  
* However, now that we have CNN's, the input has dimensions too.  
* The input to the neural network will *usually* have 1, 2 or 3 dimensions.  4 or more dimensions is unusual.  
----------
* You might have a 2D input to a neural network that has 64x64 pixels. 
* This would result in 4,096 input neurons.  
* This network is either** 2D or 4,096D, **depending on which set of dimensions you are talking about!

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
* Error or loss is the difference between the actual value and the expected value.
* In deep Neural Net we adding up the Waights in every layer and at the end (or between-ReLu for hidden layer)we calculate all the waightes with a Activation function.
* The main perpose of back propagation is to go back in the Neural Network and modify the weights 
<img src="https://github.com/iAmKankan/Deep-Learning/blob/master/Pictures/neural_network-9.png?raw=true">

* Backpropagation is a technique used to train certain classes of neural networks – it is essentially a principal that allows the machine learning program to adjust itself according to looking at its past function.
* Backpropagation is sometimes called the “backpropagation of errors.”
* Backpropagation as a technique uses gradient descent: It calculates the gradient of the loss function at output, and distributes it back through the layers of a deep neural network. The result is adjusted weights for neurons.



