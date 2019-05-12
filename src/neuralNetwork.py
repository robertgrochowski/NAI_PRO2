import numpy as np
import src.trainingSet as trainingSet
import math

from numpy import matlib as m


class NeuralNetwork:

    def __init__(self, hidden_layer_neurons, alpha, max_error, max_epoch_amount,  _lambda=1):
        self.alpha = alpha
        self._lambda = _lambda
        self.maxError = max_error
        self.maxEpochAmount = max_epoch_amount
        self.hiddenLayerNeurons = hidden_layer_neurons
        self.outputLayerNeurons = 4
        self.hiddenLayerWeights = m.rand(hidden_layer_neurons, 24)
        self.outputLayerWeights = m.rand(self.outputLayerNeurons, hidden_layer_neurons)

        self.trainingSet = trainingSet.trainingSet
        self.answerSet = trainingSet.answerSet

        self.hiddenLayerBias = m.rand(self.hiddenLayerNeurons, 1)
        self.outputLayerBias = m.rand(self.outputLayerNeurons, 1)

    def teach(self):

        error = 100
        epoch = 0
        errorList = []
        while error >= self.maxError and epoch < self.maxEpochAmount:
            error = 0
            for i in range(0, self.trainingSet.shape[0]):
                # Xn:
                Xn = self.trainingSet[i]
                Dn = self.answerSet[i]

                # Compute NET and F(x) values
                # Hidden layer
                NET1 = self.hiddenLayerWeights * np.transpose(Xn) + self.hiddenLayerBias
                Y = self.get_sigmoid_bipolar_value(NET1, self._lambda)

                # Output layer
                NET2 = np.dot(self.outputLayerWeights, Y) + self.outputLayerBias
                Z = self.get_sigmoid_bipolar_value(NET2, self._lambda)

                # Back propagation
                # Compute error for each output neuron
                E = []
                for k in range(self.outputLayerNeurons):
                    net = NET2.item(k, 0)
                    d = Dn.item(0, k)
                    z = Z.item(k, 0)
                    E.insert(k, self.sigmoid_bipolar_derivative(z, self._lambda) * (d - z))

                # compute error for each hidden neuron
                P = []
                for j in range(self.hiddenLayerNeurons):
                    s = 0
                    for k in range(self.outputLayerNeurons):
                        s += self.outputLayerWeights.item(k, j) * E[k]
                    P.insert(j, self.sigmoid_bipolar_derivative(Y.item(j, 0), self._lambda) * s)

                # Improve output weights
                for k in range(self.outputLayerNeurons):
                    self.outputLayerWeights[k] = self.outputLayerWeights[k] + np.dot(self.alpha * E[k], np.transpose(Y))
                    self.outputLayerBias[k] = self.outputLayerBias[k] + (self.alpha * E[k])

                # Improve hidden layer weights
                for j in range(self.hiddenLayerNeurons):
                    self.hiddenLayerWeights[j] = self.hiddenLayerWeights[j] + np.dot(self.alpha * P[j], Xn)
                    self.hiddenLayerBias[j] = self.hiddenLayerBias[j] + (self.alpha * P[j])

                # Cumulate error
                for k in range(self.outputLayerNeurons):
                    error += pow((Dn.item(0, k) - Z.item(k, 0)), 2)

            error /= 2
            epoch += 1
            errorList.append(error)
            print(str(error))

        return errorList, epoch

    def classify_input(self, _input):
        # Hidden layer
        NET1 = self.hiddenLayerWeights * np.transpose(_input) + self.hiddenLayerBias
        Y = self.get_sigmoid_bipolar_value(NET1, self._lambda)

        # Output layer
        NET2 = np.dot(self.outputLayerWeights, Y) + self.outputLayerBias
        Z = self.get_sigmoid_bipolar_value(NET2, self._lambda)

        print(Z)
        out = m.zeros((4, 1))

        for k in range(4):
            if Z.item(k, 0) >= 0.75:
                out[k][0] = 1

        result = 0
        for k in range(3, -1, -1):
            if out.item(k, 0) == 1:
                result += 2**(3-k)

        return result if result <= 9 else -1

    # Compute bipolar derivative (F'(x))
    def sigmoid_bipolar_derivative(self, y, _lambda=1):
        return _lambda/2 * (1 - pow(y, 2))

    # Compute bipolar value (F(x))
    def get_sigmoid_bipolar_value(self, NETValue, _lambda=1):
        output = np.mat(NETValue, dtype=float)

        for i in range(output.shape[0]):
            output[i][0] = (2 / (1 + pow(math.e, -output.item(i, 0)*_lambda))) - 1

        return output

