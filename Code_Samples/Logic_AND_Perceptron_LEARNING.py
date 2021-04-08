'''
'''

import numpy as np
import matplotlib.pyplot as plt # to plot error during training

class Perceptron(object):

    def __init__(self, no_of_inputs, no_of_epochs=10, learning_rate=0.01):
        self.no_of_epochs = no_of_epochs
        self.learning_rate = learning_rate
        self.weights = np.random.normal(0, 0.5, no_of_inputs + 1)
        self.output_bias = self.weights[0]

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.output_bias
        if summation > 0:
          activation = 1
        else:
          activation = 0
        return activation

    def train(self, training_inputs, labels):
        interation_number = 0
        epoch_number = 0
        for _ in range(self.no_of_epochs):
            interation_number += 1
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.output_bias += self.learning_rate * (label - prediction)
                epoch_number += 1
                print(interation_number,epoch_number,label, prediction)
############################
############################
############################
############################
############################
############################
training_inputs = []
training_inputs.append(np.array([1, 1]))
training_inputs.append(np.array([1, 0]))
training_inputs.append(np.array([0, 1]))
training_inputs.append(np.array([0, 0]))

teacher = np.array([1, 0, 0, 0])


NN=perceptron = Perceptron(2,10,0.1)
NN.train(training_inputs, teacher)
m = -(NN.weights[1] / NN.weights[2])
c = -(NN.output_bias / NN.weights[2])

plt.figure(figsize=(200,1))
fig, ax = plt.subplots()
xmin, xmax = -0.2, 1.4
X = np.arange(xmin, xmax, 0.1)
ax.scatter(0, 0, color="r")
ax.scatter(0, 1, color="r")
ax.scatter(1, 0, color="r")
ax.scatter(1, 1, color="g")
ax.set_xlim([xmin, xmax])
ax.set_ylim([-0.1, 1.1])


print(m, c)
ax.plot(X, m * X + c )
plt.plot()
