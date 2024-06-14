"""Math library"""

import numpy as np

# Neural network one layer of three neurons

# One batch with three samples
inputs = [
    [1, 2, 3, 2.5],
    [2, 5, -1, 2],
    [-1.5, 2.7, 3.3, -0.8],
]

# Weights for each neuron
weights = [
    [0.2, 0.8, -0.5, 1],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87],
]

# Bias for each neuron
biases = [2, 3, 0.5]

outputs = np.dot(inputs, np.array(weights).T) + biases

print(outputs)
