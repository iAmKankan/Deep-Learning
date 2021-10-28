![light](https://user-images.githubusercontent.com/12748752/136802581-e8e0607f-3472-44f7-a8b2-8ba82a0f8070.png)
![dark](https://user-images.githubusercontent.com/12748752/136802585-2ef5b7ff-ddbc-417f-b963-ca233db3ded1.png)
## Index
![dark](https://user-images.githubusercontent.com/12748752/136802585-2ef5b7ff-ddbc-417f-b963-ca233db3ded1.png)
* [Optimization](#optimization)
* [Types of Optimizer](#types-of-optimizer)
   * [Gradient Descent](https://github.com/iAmKankan/Deep-Learning/blob/master/gradient-descent.md)
   * [Stochastic Gradient Descent]
   * [Minibatch Stochastic Gradient Descent]
   * [Momentum]
   * [Adagrad]
   * [RMSProp]
   * [Adadelta]
   * [Adam]
   * [Learning Rate Scheduling]
* [Optimization Challenges in Deep Learning](#challenges)
   * [Local Minima](#)
   * [Saddle Points](#)
   * [Vanishing Gradients](#)
## Optimization 
![dark](https://user-images.githubusercontent.com/12748752/136802585-2ef5b7ff-ddbc-417f-b963-ca233db3ded1.png)
* Optimizers are algorithms or methods used to change the attributes of the neural network such as **weights**, **Bias** and **learning rate** to reduce the losses. 
> <img src="https://latex.codecogs.com/svg.image?\\&space;W\&space;=\&space;W\&space;&plus;&space;\&space;\Delta&space;W&space;\\b\&space;=\&space;W\&space;&plus;&space;\&space;\Delta&space;b&space;\\&space;\Delta&space;W\&space;=\&space;-\eta&space;\nabla&space;c\&space;;\&space;\mathrm{[\eta=\&space;Learning\&space;rate,&space;\nabla&space;c=\&space;minimizing\&space;error&space;]}&space;\\&space;\\\Delta&space;W\&space;=\&space;-\eta&space;\frac{\partial&space;c}{\partial&space;w}&space;\&space;\mathrm{[Gredient\&space;Descent&space;]}" title="\\ W\ =\ W\ + \ \Delta W \\b\ =\ W\ + \ \Delta b \\ \Delta W\ =\ -\eta \nabla c\ ;\ \mathrm{[\eta=\ Learning\ rate, \nabla c=\ minimizing\ error ]} \\ \\\Delta W\ =\ -\eta \frac{\partial c}{\partial w} \ \mathrm{[Gredient\ Descent ]}" />

* Optimizers are used to solve optimization problems by minimizing the function.

* For a deep learning problem, we usually define a *loss function* first.
* Once we have the loss function, we can use an optimization algorithm in attempt to minimize the loss.
* In optimization, a loss function is often referred to as the *objective function* of the optimization problem. 
* In Deep Learning after calculationg weights  
