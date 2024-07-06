# Neurons, Layers and Batches

A neural network consists of multiple layers that contains neurons.

## Neurons

A neuron takes in a number of inputs and gives an output.

### Input

If it is a dense layer neural net the number of inputs are all the outputs from the previous layer.

### Output

The output is computed by multiplying each input with a unique **weight** and adding them together.
Afterwards a **bias** is added to the result.

## Layers

A neural network can consist of multiple layers. The first layer or the input of a neural network
is called the **input layer**. The last layer in the network or the output is called the
**output layer**. The layers between the input and output layers are called the **hidden layers**.

## Batches

When training neural networks batches of input data are used to help with generalization during training.
If batches are not used the network will likely fit to an individual sample rather than the entire dataset.
Batches of data also enable faster training using parallel processing.
