# -*- coding: utf-8 -*-

import sys
import logging

import numpy as np

from src.util.activation_functions import Activation
from src.model.classifier import Classifier

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


class Perceptron(Classifier):
    """
    A digit-7 recognizer based on perceptron algorithm

    Parameters
    ----------
    train : list
    valid : list
    test : list
    learningRate : float
    epochs : positive int

    Attributes
    ----------
    learningRate : float
    epochs : int
    trainingSet : list
    validationSet : list
    testSet : list
    weight : list
    """

    def __init__(self, train, valid, test,
                 learningRate=0.01, epochs=50):

        self.learningRate = learningRate
        self.epochs = epochs

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test

        # Initialize the weight vector with small random values
        # around 0 and0.1
        self.weight = np.random.rand(self.trainingSet.input.shape[1]) / 100
        # TODO add the bias here

    def train(self, verbose=True):
        """Train the perceptron with the perceptron learning algorithm.

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """
        train_n = self.trainingSet.input.shape[0]
        if verbose:
            print 'enter inters...'
        for i in range(0, self.epochs):
            evl_result = np.array(self.evaluate(test = self.trainingSet))
            # the following itemwise time can be also replaced from logical and, which maybe faster
            # iter_err = np.logical_and(evl_result, self.trainingSet.label)
            iter_err = (evl_result.astype(int) != self.trainingSet.label).astype(int)
            correct_num = train_n - np.sum(iter_err)
            if verbose:
                print('iter ' + str(i) + ' finished with ' + str(correct_num) + '/' + str(train_n) + 'correctness')
            if correct_num < train_n:
                self.updateWeights(self.trainingSet.input, iter_err)
        if verbose:
            print('finish iterating!')

    def classify(self, testInstance):
        """Classify a single instance.

        Parameters
        ----------
        testInstance : list of floats

        Returns
        -------
        bool :
            True if the testInstance is recognized as a 7, False otherwise.
        """
        return self.fire(testInstance)

    def evaluate(self, test=None):
        """Evaluate a whole dataset.

        Parameters
        ----------
        test : the dataset to be classified
        if no test data, the test set associated to the classifier will be used

        Returns
        -------
        List:
            List of classified decisions for the dataset's entries.
        """
        if test is None:
            test = self.testSet.input
        # Once you can classify an instance, just use map for all of the test set.
        return list(map(self.classify, test))

    def updateWeights(self, input, error):
        # TODO there is for now no operation over the bias/x0 term according to the original design of Framework
        self.weight = self.weight + np.sum(input.transpose() * error, axis=1)

    def fire(self, input):
        """Fire the output of the perceptron corresponding to the input """
        return Activation.sign(np.dot(np.array(input), self.weight))
