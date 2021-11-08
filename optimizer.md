## Index
![dark](https://user-images.githubusercontent.com/12748752/136802585-2ef5b7ff-ddbc-417f-b963-ca233db3ded1.png)
* [Optimization](#optimization)
* [### Goal of Optimization](#goal-of-optimization)
* [Training Error Optimization](#training-error-optimization)
* [Types of Optimizer](#types-of-optimizer)
   * [Gradient Descent](https://github.com/iAmKankan/Deep-Learning/blob/master/gradient-descent.md)
   * [Stochastic Gradient Descent](#)
   * [Minibatch Stochastic Gradient Descent](#)
   * [Momentum](#)
   * [Adagrad](#)
   * [RMSProp](#)
   * [Adadelta](#)
   * [Adam](#)
   * [Learning Rate Scheduling](#)
* [Optimization Challenges in Deep Learning](#challenges)
   * [Local Minima](#)
   * [Saddle Points](#)
   * [Vanishing Gradients](#)
## Optimization 
![dark](https://user-images.githubusercontent.com/12748752/136802585-2ef5b7ff-ddbc-417f-b963-ca233db3ded1.png)
* In Statistics, Machine Learning and other Data Science fields, we optimize a lot of stuff.
> #### Linear Regression we optimize
> *  _**Intercept**_ 
> *  _**Slope**_
<img src="https://user-images.githubusercontent.com/12748752/139344656-8e5f34a2-608d-45d5-90a9-0dc4676692e9.png" width=30%>

> #### When we use Logistic Regression we optimize 
> * _**A Squiggle**_
<img src="https://user-images.githubusercontent.com/12748752/139344662-2edb7ae2-2ee9-43d0-9bec-5d099e62bce5.png" width=30%>

> #### When we use t-SNE we optimize 
> * _**Clusters**_
### Goal of Optimization
![light](https://user-images.githubusercontent.com/12748752/136802581-e8e0607f-3472-44f7-a8b2-8ba82a0f8070.png)

> #### **The goal of Optimization is primarily concerned with minimizing an objective(loss function)**
> * The goal of optimization is to reduce the _**Training Error**_.


> #### **The goal of Deep Learning is finding a suitable model, given a finite amount of data.**
> * The goal of deep learning (or more broadly, statistical inference) is to reduce the _**Generalization Error**_. 
> * To accomplish so we need to pay attention to **overfitting** in addition to using the **optimization algorithm** to reduce the training error.

* **Empirical Risk:** The empirical risk is an average loss on the training dataset.
*  **Risk:**: The risk is the expected loss on the entire population of data. 

### Training Error Optimization
![light](https://user-images.githubusercontent.com/12748752/136802581-e8e0607f-3472-44f7-a8b2-8ba82a0f8070.png)

* In Deep Learning optimizers are algorithms or methods used to change the attributes of the neural network such as **weights**, **Bias** and **learning rate** to reduce the losses. 
> <img src="https://latex.codecogs.com/svg.image?\\&space;W\&space;=\&space;W\&space;&plus;&space;\&space;\Delta&space;W&space;\\b\&space;=\&space;W\&space;&plus;&space;\&space;\Delta&space;b&space;\\&space;\Delta&space;W\&space;=\&space;-\eta&space;\nabla&space;c\&space;;\&space;\mathrm{[\eta=\&space;Learning\&space;rate,&space;\nabla&space;c=\&space;minimizing\&space;error&space;]}&space;\\&space;\\\Delta&space;W\&space;=\&space;-\eta&space;\frac{\partial&space;c}{\partial&space;w}&space;\&space;\mathrm{[Gredient\&space;Descent&space;]}" title="\\ W\ =\ W\ + \ \Delta W \\b\ =\ W\ + \ \Delta b \\ \Delta W\ =\ -\eta \nabla c\ ;\ \mathrm{[\eta=\ Learning\ rate, \nabla c=\ minimizing\ error ]} \\ \\\Delta W\ =\ -\eta \frac{\partial c}{\partial w} \ \mathrm{[Gredient\ Descent ]}" />

* Optimizers are used to solve optimization problems by minimizing the function.

* For a deep learning problem, we usually define a *loss function* first.
* Once we have the loss function, we can use an optimization algorithm in attempt to minimize the loss.
* In optimization, a loss function is often referred to as the *objective function* of the optimization problem. 
* In Deep Learning after calculationg weights  




