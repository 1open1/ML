import math
def sigmoid(x):
  return 1 / (1 + math.exp(-x))

sigmoid(1)

sigmoid(100)  #So for the input greater than 10 the output would be 1 (S shape) (for less than 10 it is 0)

import numpy as np
import matplotlib.pyplot as plt

def plot_sigmoid():
    x = np.linspace(-10, 10, 100)  # Generate 100 equally spaced values from -10 to 10
    y = 1 / (1 + np.exp(-x))  # Compute the sigmoid function values

    plt.plot(x, y)
    plt.xlabel('Input')
    plt.ylabel('Sigmoid Output')
    plt.title('Sigmoid Activation Function')
    plt.grid(True)
    plt.show()
plot_sigmoid()

"""**Tanh Function**"""

def tanh(x): #Range of -1 to 1
  return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))

tanh(-10)

tanh(-100)

tanh(100)

import numpy as np
import matplotlib.pyplot as plt

def plot_tanh():
    # Generate values for x
    x = np.linspace(-10, 10, 100)

    # Compute tanh values for corresponding x
    tanh = np.tanh(x)

    # Plot the tanh function
    plt.plot(x, tanh)
    plt.title("Hyperbolic Tangent (tanh) Activation Function")
    plt.xlabel("x")
    plt.ylabel("tanh(x)")
    plt.grid(True)
    plt.show()
plot_tanh()

"""**ReLu Function**"""

def relu(x):
    return max(0,x)

relu(-22)

import numpy as np
import matplotlib.pyplot as plt

def plot_relu():
    # Generate values for x
    x = np.linspace(-10, 10, 100)

    # Compute ReLU values for corresponding x
    relu = np.maximum(0, x)

    # Plot the ReLU function
    plt.plot(x, relu)
    plt.title("ReLU Activation Function")
    plt.xlabel("x")
    plt.ylabel("ReLU(x)")
    plt.grid(True)
    plt.show()
plot_relu()

"""**Leaky ReLu**"""

def leaky_relu(x, alpha=0.1):
    return max(x, alpha * x)

leaky_relu(-125)

def plot_leaky_relu():
    # Generate values for x
    x = np.linspace(-10, 10, 100)

    # Define the leaky ReLU function
    def leaky_relu(x, alpha=0.1):
        return np.where(x >= 0, x, alpha * x)

    # Compute leaky ReLU values for corresponding x
    leaky_relu_values = leaky_relu(x)

    # Plot the leaky ReLU function
    plt.plot(x, leaky_relu_values)
    plt.title("Leaky ReLU Activation Function")
    plt.xlabel("x")
    plt.ylabel("Leaky ReLU(x)")
    plt.grid(True)
    plt.show()
plot_leaky_relu()

"""**Soft Max**"""

def softmax(x):  #It takes an input array and returns a probability array, on of the array summation results to 1
    # Subtracting the maximum value for numerical stability
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x, axis=0)

softmax([4,5,6])

sum(softmax([4,5,6]))

def plot_softmax(probabilities, class_labels):
    plt.bar(class_labels, probabilities)
    plt.xlabel("Class")
    plt.ylabel("Probability")
    plt.title("Softmax Output")
    plt.show()

# Example usage:
class_labels = ["Class A", "Class B", "Class C"]
probabilities = softmax([4, 5, 6])
plot_softmax(probabilities, class_labels)
