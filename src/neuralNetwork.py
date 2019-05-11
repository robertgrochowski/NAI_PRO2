import numpy as np
from numpy import matlib as m
import math

class NeuralNetwork:

    trainingSet = m.mat([
        [1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1],  # 0
        [0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0],  # 0
        [0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],  # 0
        [1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0],  # 0
        [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],  # 1
        [0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],  # 1
        [0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],  # 1
        [0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0],  # 1
        [1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1],  # 2
        [1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1],  # 2
        [1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1],  # 2 td
        [1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1],  # 2 td
        [1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1],  # 3
        [1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1],  # 3
        [1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1],  # 3
        [1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1],  # 3
        [1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1],  # 4
        [1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],  # 4
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1],  # 4
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]   # 4
    ], dtype=float)

    answerSet = m.mat([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0]
    ],dtype=float)

    def __init__(self, hidden_layer_neurons, alpha, _lambda):
        self.alpha = alpha
        self.hiddenLayerNeurons = hidden_layer_neurons
        self.outputLayerNeurons = 4
        self.hiddenLayerWeights = m.rand(hidden_layer_neurons, 24) # todo 24
        self.outputLayerWeights = m.rand(self.outputLayerNeurons, hidden_layer_neurons) # todo 4

        self.hiddenLayerBias = m.rand(hidden_layer_neurons, 1)
        self.outputLayerBias = m.rand(4, 1)

    def teach(self):

        for epoch in range(400):
            error = 0
            for i in range(0, self.trainingSet.shape[0]):
                # Xn:
                Xn = self.trainingSet[i]
                Dn = self.answerSet[i]

                # Compute NET and F(x) values
                # Hidden layer
                NET1 = self.hiddenLayerWeights * np.transpose(Xn) + self.hiddenLayerBias
                Y = self.get_sigmoid_bipolar_value(NET1)

                # Output layer
                NET2 = np.dot(self.outputLayerWeights, Y) + self.outputLayerBias
                Z = self.get_sigmoid_bipolar_value(NET2)

                # Compute error for each output neuron
                E = []
                for k in range(self.outputLayerNeurons):
                    net = NET2.item(k, 0)
                    d = Dn.item(0, k)
                    z = Z.item(k, 0)

                    E.insert(k, self.sigmoid_bipolar_derivative(z) * (d - z))

                # compute error for each hidden neuron
                P = []
                for j in range(self.hiddenLayerNeurons):
                    s = 0
                    for k in range(self.outputLayerNeurons):
                        s += self.outputLayerWeights.item(k, j) * E[k]
                    y = Y.item(j, 0)
                    P.insert(j, self.sigmoid_bipolar_derivative(y) * s)

                # Improve output weights
                for k in range(self.outputLayerNeurons):
                    self.outputLayerWeights[k] = self.outputLayerWeights[k] + np.dot(self.alpha * E[k], np.transpose(Y))
                    self.outputLayerBias[k] = self.outputLayerBias[k] + (self.alpha * E[k])

                # Improve hidden layer weights
                for j in range(self.hiddenLayerNeurons):
                    new = np.dot(self.alpha * P[j], Xn)
                    self.hiddenLayerWeights[j] = self.hiddenLayerWeights[j] + new
                    self.hiddenLayerBias[j] = self.hiddenLayerBias[j] + (self.alpha * P[j])

                # Kumuluj blad
                for k in range(self.outputLayerNeurons):
                    error += pow((Dn.item(0, k) - Z.item(k, 0)), 2)

            error /= 2
            print(error)

    def sigmoid_bipolar_derivative(self, y, _lambda=1):
        return _lambda/2 * (1 - pow(y, 2))

    def get_sigmoid_bipolar_value(self, NETValue, _lambda=1):
        output = np.mat(NETValue, dtype=float)

        for i in range(output.shape[0]):
            output[i][0] = self.sigmoid(output.item(i, 0))

        return output

    def sigmoid(self, val):
        return (2 / (1 + pow(math.e, val*-1*1))) - 1

if __name__ == '__main__':
    n = NeuralNetwork(17, 0.3, 1)
    n.teach()

    #v = n.sigmoid(-1)
    #print(v)
    pass

