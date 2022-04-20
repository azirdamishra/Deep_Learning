
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from activationFuncs import Tanh
from losses import mse, mse_prime
from network import train, predict
from dense import Dense


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
    #reshape and normalise the input data
    x = x.reshape(x.shape[0], 28 * 28, 1)
    x = x.astype("float32") / 255

    #encode output which is a number in the range [0,9] into a vector of size 10
    y = np_utils.to_categorical(y)
    y = y.reshape(y.shape[0], 10, 1)
    return x[:limit], y[:limit]

#load MNIST from the server 
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 1000)
x_test, y_test = preprocess_data(x_test, y_test, 20)

network = [
    Dense(28 * 28, 40),
    Tanh(),
    Dense(40, 10),
    Tanh()
]

#train
train(network, mse, mse_prime, x_train, y_train, epochs=100, learning_rate=0.1)

#test
for x, y in zip(x_test, y_test):
    output = predict(network, x)
    print("predicted: ", np.argmax(output), " \ttrue_value: ", np.argmax(y))
