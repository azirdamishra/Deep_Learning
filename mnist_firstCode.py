
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils

from dense import Dense
from convolution import Convolution
from reshape import Reshape
from activationFuncs import Sigmoid
from losses import binary_cross_entropy,binary_cross_entropy_prime

#data import problems
import requests
requests.packages.urllib3.disable_warnings()
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context

def preprocess_data(x, y, limit):
    zero_index = np.where(y == 0)[0][:limit]
    one_index = np.where(y == 1)[0][:limit]
    all_indices = np.hstack((zero_index, one_index))
    all_indices = np.random.permutation(all_indices)
    x,y = x[all_indices], y[all_indices]
    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype("float32") / 255
    y = np_utils.to_categorical(y)
    y = y.reshape(len(y), 2, 1)
    return x, y

#load MSNIST from server, limit to 100 images per class since we're not training on GPU
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 100)
x_test, y_test = preprocess_data(x_test, y_test, 100)

#neural network
network = [
    Convolution((1,28,28), 3, 5),
    Sigmoid(),
    Reshape((5,26,26), (5 * 26 *26,1)),
    Dense(5 * 26 * 26, 100),
    Sigmoid(),
    Dense(100, 2),
    Sigmoid()
]

epochs = 20
learning_rate = 0.1

#train 
for e in range(epochs):
    error = 0
    for x, y in zip(x_train, y_train):
        #forward
        output = x
        for layer in network:
            output = layer.forward(output)

        #error
        error += binary_cross_entropy(y, output)

        #backward 
        gradientt = binary_cross_entropy_prime(y, output)
        for layer in reversed(network):
            gradientt = layer.backward(gradientt, learning_rate)

    error /= len(x_train)
    print(f"{e + 1}/{epochs}, error={error}")

    #test
    for x, y in zip(x_test, y_test):
        output = x
        for layer in network:
            output = layer.forward(output)
        print(f"pred: {np.argmax(output)}, true: {np.argmax(y)}")
