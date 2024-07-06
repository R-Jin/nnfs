import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from Layer_Dense import Layer_Dense

nnfs.init()

classes = 3

# Create dataset
X, y = spiral_data(samples=100, classes=classes)

n_inputs = X.shape[1]
print(n_inputs)

# Create dense layer with 2 input features and 3 outputs corresponding to the number of classes in the dataset
dense1 = Layer_Dense(n_inputs, classes)
dense1.forward(X)

# Show outputs of the first 5 samples
print(dense1.output[:5])