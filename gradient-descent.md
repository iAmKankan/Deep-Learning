## Index
![dark](https://user-images.githubusercontent.com/12748752/136802585-2ef5b7ff-ddbc-417f-b963-ca233db3ded1.png)


## Gradient Descent
![dark](https://user-images.githubusercontent.com/12748752/136802585-2ef5b7ff-ddbc-417f-b963-ca233db3ded1.png)
* It is an optimization algorithm to find the minimum of a function.
> #### Start with a random point on the function and move in the _**negative direction**_ of the _**gradient of the function**_ to reach the _**local/global minima**_.

> ### Example 1
> * <img src="https://latex.codecogs.com/svg.image?y=(x&plus;5)^2" title="y=(x+5)^2" />
>
> ### Step1
> * Let's say random point **X=-3**
> * Then, find the gradient of the function 
> ><img src="https://latex.codecogs.com/svg.image?\frac{\mathrm{d}&space;y}{\mathrm{d}&space;x}\&space;=\&space;\mathrm{2&space;\ast&space;&space;(x&plus;5)}&space;" title="\frac{\mathrm{d} y}{\mathrm{d} x}\ =\ \mathrm{2 \ast (x+5)} " />
>
> ### Step2
> * Move in the direction of the negative of the gradient.
> * **But: HOW MUCH to move? For that, we define a learning rate: learning_rate= 0.01**
>
> ### Step3
> * Perform 2 iterations of gradient descent.
> 
> * **Initialize Parameters**
> 
> <img src="https://latex.codecogs.com/svg.image?X_0&space;=&space;-3,\&space;\&space;learning\&space;rate&space;=&space;0.01,\&space;\&space;\frac{\mathrm{d}&space;y}{\mathrm{d}&space;x}=\&space;2\ast&space;(x&plus;5)" title="X_0 = -3,\ \ learning\ rate = 0.01,\ \ \frac{\mathrm{d} y}{\mathrm{d} x}=\ 2\ast (x+5)" />
> 
> * **Iteration 1**
>> <img src="https://latex.codecogs.com/svg.image?\\X_1&space;=\&space;X_0&space;-\&space;\&space;(learning\&space;rate)\&space;\ast&space;\&space;\&space;\frac{\mathrm{d}&space;y}{\mathrm{d}&space;x}&space;\\X_1&space;=\&space;(-3)&space;-\&space;\&space;(0.01)\&space;\ast&space;\&space;\&space;(2&space;\ast(x&plus;5)&space;)\\X_1&space;=\&space;(-3)&space;-\&space;\&space;(0.01)\&space;\ast&space;\&space;\&space;(2&space;\ast(-3&plus;5)&space;)\\&space;X_1&space;=\&space;-3.04&space;" title="\\X_1 =\ X_0 -\ \ (learning\ rate)\ \ast \ \ \frac{\mathrm{d} y}{\mathrm{d} x} \\X_1 =\ (-3) -\ \ (0.01)\ \ast \ \ (2 \ast(x+5) )\\X_1 =\ (-3) -\ \ (0.01)\ \ast \ \ (2 \ast(-3+5) )\\ X_1 =\ -3.04 " />
>
> * **Iteration 2**
>><img src="https://latex.codecogs.com/svg.image?\\X_2&space;=\&space;X_1&space;-\&space;\&space;(learning\&space;rate)\&space;\ast&space;\&space;\&space;\frac{\mathrm{d}&space;y}{\mathrm{d}&space;x}&space;\\X_2&space;=\&space;(-3.04)&space;-\&space;\&space;(0.01)\&space;\ast&space;\&space;\&space;(2&space;\ast(--3.04&plus;5)&space;)\\&space;X_2&space;=\&space;-3.0792&space;" title="\\X_2 =\ X_1 -\ \ (learning\ rate)\ \ast \ \ \frac{\mathrm{d} y}{\mathrm{d} x} \\X_2 =\ (-3.04) -\ \ (0.01)\ \ast \ \ (2 \ast(--3.04+5) )\\ X_2 =\ -3.0792 " />
>
> * ...
> >........
> 
> * **Iteration n**
>> <img src="https://latex.codecogs.com/svg.image?X_n&space;=\&space;-5.0" title="X_n =\ -5.0" />

* The itaration would be continued till the result reached the global minima.
* For the above example the Global Minima is **X=-5**

### One-Dimensional Gradient Descent
![light](https://user-images.githubusercontent.com/12748752/136802581-e8e0607f-3472-44f7-a8b2-8ba82a0f8070.png)
* Lets explain why the gradient descent algorithm may reduce the value of the objective function. 
* Consider some continuously differentiable real-valued function  f:R→R . Using a Taylor expansion we obtain
