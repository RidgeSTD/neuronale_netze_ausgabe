# -*- coding: utf-8 -*-

"""
Activation functions which can be used within neurons.
"""

import numpy as np


class Activation:
    """
    Containing various activation functions and their derivatives
    """

    @staticmethod
    def sign(netOutput, threshold=0):
        return 1 if netOutput >= threshold else 0

    @staticmethod
    def sigmoid(netOutput):
        return(1 / (1 + np.exp(-netOutput)))

    @staticmethod
    def sigmoidPrime(netOutput):
        return Activation.sigmoid(netOutput) * Activation.sigmoid(1 - netOutput)

    @staticmethod
    def tanh(netOutput):
        return np.tanh(netOutput)

    @staticmethod
    def tanhPrime(netOutput):
        return 1 - np.sqrt(Activation.tanh(netOutput))

    @staticmethod
    def rectified(netOutput):
        return lambda x: max(0.0, x)

    @staticmethod
    def rectifiedPrime(netOutput):
        # Here you have to code the derivative of rectified linear function
        pass

    @staticmethod
    def identity(netOutput):
        return lambda x: x

    @staticmethod
    def identityPrime(netOutput):
        # Here you have to code the derivative of identity function
        pass

    @staticmethod
    def softmax(netOutput):
        # Here you have to code the softmax function
        pass

    @staticmethod
    def getActivation(str):
        """
        Returns the activation function corresponding to the given string
        """

        if str == 'sigmoid':
            return Activation.sigmoid
        elif str == 'softmax':
            return Activation.softmax
        elif str == 'tanh':
            return Activation.tanh
        elif str == 'relu':
            return Activation.rectified
        elif str == 'linear':
            return Activation.identity
        else:
            raise ValueError('Unknown activation function: ' + str)

    @staticmethod
    def getDerivative(str):
        """
        Returns the derivative function corresponding to a given string which
        specify the activation function
        """

        if str == 'sigmoid':
            return Activation.sigmoidPrime
        elif str == 'tanh':
            return Activation.tanhPrime
        elif str == 'relu':
            return Activation.rectifiedPrime
        elif str == 'linear':
            return Activation.identityPrime
        else:
            raise ValueError('Cannot get the derivative of'
                             ' the activation function: ' + str)
