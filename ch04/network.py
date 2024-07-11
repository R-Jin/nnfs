import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from Layer_Dense import Layer_Dense
from Activation_ReLU import Activation_ReLU
from softmax import Activation_Softmax

nnfs.init()

classes = 3

# Create dataset
X, y = spiral_data(samples=100, classes=classes)

n_inputs = X.shape[1]
print(n_inputs)

# Create dense layer with 2 input features and 3 outputs corresponding to the number of classes in the dataset
dense1 = Layer_Dense(n_inputs, classes)
dense1.forward(X)

# Activation 
activation1 = Activation_ReLU()
activation1.forward(dense1.output)

dense2 = Layer_Dense(3, 3)
dense2.forward(activation1.output)

# Softmax activation function as last layer
activation2 = Activation_Softmax()
activation2.forward(dense2.output)

print(activation2.output[:5])