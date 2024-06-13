inputs = [1, 2, 3, 2.5]
weights1 = [0.2, 0.8, -0.5, 1]
weights2 = [0.5, -0.91, 0.26, -0.5]
weights3 = [-0.26, -0.27, 0.17, 0.87]

bias1 = 2
bias2 = 3
bias3 = 0.5

weights = [weights1, weights2, weights3]
biases = [bias1, bias2, bias3]

outputs = []

for neuron_weights, bias in zip(weights, biases):
    output = 0
    for n_input, weight in zip(inputs, neuron_weights):
        output += n_input * weight
    output += bias
    outputs.append(output)

print(outputs)
