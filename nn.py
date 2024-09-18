import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load and preprocess data
def load_data(train_path, test_path):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    train_data = np.array(train_data)
    test_data = np.array(test_data)
    np.random.shuffle(train_data)
    train_data = train_data.T
    test_data = test_data.T
    return train_data, test_data

def preprocess_data(data):
    X = data[1:] / 255.0
    Y = data[0]
    return X, Y

# Activation functions
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

def softmax(z):
    z -= np.max(z, axis=0)
    return np.exp(z) / np.sum(np.exp(z), axis=0)

# Neural network functions
def init_params(input_dim, hidden_dim, output_dim):
    W1 = np.random.normal(size=(hidden_dim, input_dim)) * np.sqrt(1. / input_dim)
    b1 = np.random.normal(size=(hidden_dim, 1)) * np.sqrt(1. / hidden_dim)
    W2 = np.random.normal(size=(output_dim, hidden_dim)) * np.sqrt(1. / (hidden_dim + output_dim))
    b2 = np.random.normal(size=(output_dim, 1)) * np.sqrt(1. / output_dim)
    return W1, b1, W2, b2

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = sigmoid(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / X.shape[1] * dZ2.dot(A1.T)
    db2 = 1 / X.shape[1] * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * sigmoid_derivative(Z1)
    dW1 = 1 / X.shape[1] * dZ1.dot(X.T)
    db1 = 1 / X.shape[1] * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 -= alpha * dW1
    b1 -= alpha * db1
    W2 -= alpha * dW2
    b2 -= alpha * db2
    return W1, b1, W2, b2

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

# Training and testing functions
def train(X, Y, alpha, iterations):
    input_dim = X.shape[0]
    hidden_dim = 20
    output_dim = 10
    W1, b1, W2, b2 = init_params(input_dim, hidden_dim, output_dim)
    for i in range(iterations + 1):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration:", i)
            predictions = get_predictions(A2)
            print("Accuracy:", get_accuracy(predictions, Y) * 100, "%")
    return W1, b1, W2, b2

def test(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def visualize_prediction(index, X, W1, b1, W2, b2):
    current_image = X[:, index, None]
    prediction = test(current_image, W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction:", prediction)
    print("Label:", label)
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

# Load and preprocess data
train_data, test_data = load_data('task 13/mnist_train.csv', 'task 13/mnist_test.csv')
X_train, Y_train = preprocess_data(train_data)
X_test, Y_test = preprocess_data(test_data)

# Train neural network
W1, b1, W2, b2 = train(X_train, Y_train, 3, 100)

# Test neural network
test_predictions = test(X_test, W1, b1, W2, b2)
print("Test accuracy:", get_accuracy(test_predictions, Y_test) * 100, "%")

# Visualize prediction
visualize_prediction(0, X_train, W1, b1, W2, b2)