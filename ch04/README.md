# Activation Functions

Activation functions make it possible for neural networks to map nonlinear functions.
There are a lot of different activation functions and they all have different use cases.
Activation functions take a neurons output as input and outputs a modified version of it.
In general a neural network have two types of activation functions. First type is used in
the hidden layer, second type is used in the output layer. The function used on the hidden
neurons does not have to be the same but usually are.

## Step Activation Function

Simulates a neuron "firing" or "not firing" based on the input to the activation function.

$$
f(x) =
\begin{cases}
    1 & x \gt 0 \\
    0 & x \leq 0
\end{cases}
$$

This has historically been used in hidden layers but is nowadays rarely used.

## Linear Activation Function

Maps a straight line and the function is $f(x) = x$. This function is often applied to the last layer in regression models (Models that output a scalar instead of a classification).

## Sigmoid Activation Function

When training and optimizing a neural network you assess the impact that weights and biases have on a network's output. Therefore it is good to have an activation function that gives more information about how "close" the neuron was to activating.

The original activation function used for neural networks was the Sigmoid function:

$$
y = \frac{1}{1+e^{-x}}
$$

## ReLu Activation Function

## Softmax Activation Function
