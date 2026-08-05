import numpy as np

# Input Dataset (XOR)
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

# Output Dataset
y = np.array([[0],
              [1],
              [1],
              [0]])

# Sigmoid Activation Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of Sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# Initialize weights and biases
np.random.seed(1)

input_neurons = 2
hidden_neurons = 2
output_neurons = 1

weights_input_hidden = np.random.uniform(size=(input_neurons, hidden_neurons))
weights_hidden_output = np.random.uniform(size=(hidden_neurons, output_neurons))

bias_hidden = np.random.uniform(size=(1, hidden_neurons))
bias_output = np.random.uniform(size=(1, output_neurons))

learning_rate = 0.5

# Training using Backpropagation
epochs = 10000

for epoch in range(epochs):

    # Forward Propagation
    hidden_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, weights_hidden_output) + bias_output
    predicted_output = sigmoid(final_input)

    # Calculate Error
    error = y - predicted_output

    # Backpropagation
    d_output = error * sigmoid_derivative(predicted_output)

    error_hidden = d_output.dot(weights_hidden_output.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_output)

    # Update Weights and Biases
    weights_hidden_output += hidden_output.T.dot(d_output) * learning_rate
    weights_input_hidden += X.T.dot(d_hidden) * learning_rate

    bias_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    bias_hidden += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

# Display Results
print("Predicted Output after Training:\n")

for i in range(len(X)):
    print("Input:", X[i], "Predicted:", round(predicted_output[i][0], 3))

print("\nActual Output:")
print(y)