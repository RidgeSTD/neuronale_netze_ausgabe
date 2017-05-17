# -*- coding: utf-8 -*-

from random import random
from src.model.classifier import Classifier

__author__ = "Wenlan Hua"  # Adjust this when you copy the file
__email__ = "uvlty@student.kit.edu"  # Adjust this when you copy the file


class StupidRecognizer(Classifier):
    """
    This class demonstrates how to follow an OOP approach to build a digit
    recognizer.

    It also serves as a baseline to compare with other
    recognizing method later on.
    """

    def __init__(self, train, valid, test, byChance=0.5):

        self.byChance = byChance

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test

    def train(self):
        # Do nothing, really nothing, because it's stupid
        pass

    def classify(self, testInstance):
        # byChance is the probability of being correctly recognized
        # This one is really stupid, stupid by it's name. It only gives a blind guess
        return random() < self.byChance

    def evaluate(self):
        return list(map(self.classify, self.testSet.input))
