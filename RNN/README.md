# Recurrent Neural Network
* The idea behind RNNs is to make use of sequential information. 
* In a traditional neural network we assume that all inputs (and outputs) are independent of each other. 
* But for many tasks that’s a very bad idea. 
    * If you want to predict the next word in a sentence you better know which words came before it. 
* **RNNs are called recurrent because they perform the same task for every element of a sequence, with the output being depended on the previous computations and you already know that they have a “memory” which captures information about what has been calculated so far.**

* Processing sequences and time series requires some sort of memory since dynamic temporal behaviour is also adding information to the whole picture. So by
introducing loopback connections between neurons such a Recurrent Neural Network can remember past events.


<img src="https://github.com/iAmKankan/Deep-Learning/blob/master/RNN/rnn2.png?raw=true">

* Note that any Recurrent Neural Network can be unfolded through time into a Deep Feed Forward Neural Network. So again, this whole exercise is only there since
training can be improved by changing the neural network topology from a single hidden layer feed forward network to something else.

<img src="https://github.com/iAmKankan/Deep-Learning/blob/master/RNN/rnn3.png?raw=true">
 
## Types of RNNS
<img src="https://github.com/iAmKankan/Deep-Learning/blob/master/RNN/types%20of%20rnn.png?raw=true">



## Backpropogate Through Time:
