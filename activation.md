## Index
![dark](https://user-images.githubusercontent.com/12748752/136802585-2ef5b7ff-ddbc-417f-b963-ca233db3ded1.png)
* [Activation Function](#activation-function)
   * [Sigmoid](#sigmoid)
   * [TanH](#tanh)






## Activation Function
![dark](https://user-images.githubusercontent.com/12748752/136802585-2ef5b7ff-ddbc-417f-b963-ca233db3ded1.png)

### Why do we need activation functions in the first place
![light](https://user-images.githubusercontent.com/12748752/136802581-e8e0607f-3472-44f7-a8b2-8ba82a0f8070.png)
* If you chain several linear transformations, all you get is a linear transformation.
> **For example: Say we have f(x) and g(x) then Then chaining these two linear functions gives you another linear function f(g(x)).**
>> f(x) = 2 x + 3 
>> 
>> g(x) = 5 x - 1 
>> 
>> f(g(x)) = 2(5 x - 1) + 3 = 10 x + 1.
>
> 
> 
* So, if you don’t have some non-linearity between layers, then even a deep stack of layers is equivalent to a single layer.
* You cannot solve very complex problems with that.

### Sigmoid
![light](https://user-images.githubusercontent.com/12748752/136802581-e8e0607f-3472-44f7-a8b2-8ba82a0f8070.png)
* A sigmoid function is a mathematical function having a characteristic "S"-shaped curve or sigmoid curve.
* A common example of a sigmoid function is the logistic function shown in the first figure and defined by the formula- 

> <img src="https://latex.codecogs.com/svg.image?\mathit{S(x)}\&space;=&space;\&space;\frac{1}{1&plus;e^{-x}}" title="\mathit{S(x)}\ = \ \frac{1}{1+e^{-x}}" width=20% />

#### Derivative of the Sigmoid Activation function 
![light](https://user-images.githubusercontent.com/12748752/136802581-e8e0607f-3472-44f7-a8b2-8ba82a0f8070.png)
* Inorder to do so we need to introduce **Quotient rule formula in Differentiation**
 > 
