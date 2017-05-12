# sentimentAnalysis.py

# imports
import util
from featureExtractor import BaseFeatureExtractor
from classificationMethod import ClassificationMethod
import numpy as np


vocabularyList = {} # vocabulary list
vocabularyCounts = {} # The number of occurrences of a word in SSD sentences
with open('data/SST/vocabulary.txt', 'r') as f:
    i = 0
    for line in f:
        v, c = line.strip().split()
        vocabularyCounts[v] = int(c)
        vocabularyList[v] = i
        i += 1


def loadTextData():
    rawTrainingData = []
    rawValidationData = []
    rawTestData = []
    with open('data/SST/trainingData.txt', 'r') as f:
        for line in f:
            rawTrainingData.append(line.strip())
    with open('data/SST/validationData.txt', 'r') as f:
        for line in f:
            rawValidationData.append(line.strip())
    with open('data/SST/testData.txt', 'r') as f:
        for line in f:
            rawTestData.append(line.strip())
    with open('data/SST/trainingLabels.txt', 'r') as f:
        rawTrainingLabels = [int(t) for t in f.read().strip().split()]
    with open('data/SST/validationLabels.txt', 'r') as f:
        rawValidationLabels = [int(t) for t in f.read().strip().split()]
    with open('data/SST/testLabels.txt', 'r') as f:
        rawTestLabels = [int(t) for t in f.read().strip().split()]
    return rawTrainingData, rawTrainingLabels, rawValidationData, rawValidationLabels, rawTestData, rawTestLabels


class FeatureExtractorText(BaseFeatureExtractor):
    """
    Extract text data given a list of sentences.
    """
    def __init__(self):
        super(FeatureExtractorText, self).__init__()

    def fit(self, trainingData):
        """
        Train feature extractor given the text training Data (not in numpy format)
        :param trainingData: a list of sentences
        :return:
        """

        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def extract(self, data):
        """
        Extract the feature of text data
        :param data: a list of sentences (not in numpy format)
        :return: features, in numpy format and len(features)==len(data)
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def visualize(self, data):
        """
        May be used for ease of visualize the text data
        :param data:
        :return:
        """
        pass


class ClassifierText(ClassificationMethod):
    """
    Perform classification to text data set
    """
    def __init__(self, legalLabels):
        super(ClassifierText, self).__init__(legalLabels)
        # You may use the completed classification methods
        # or directly use sklearn for learning
        # e.g.
        # import classifiers
        # self.classifier = classifiers.PerceptronClassifier(legalLabels, 50)
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """
        Train the text classifier with text features
        :param trainingData: in numpy format
        :param trainingLabels: in numpy format
        :param validationData: in numpy format
        :param validationLabels: in numpy format
        """

        # You may use the completed classification methods
        # e.g.
        # self.classifier.train(trainingData, trainingLabels, validationData, validationLabels)
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def classify(self, data):
        """
        Classify the text classifier with text features
        :param data: in numpy format
        :return:
        """

        # You may use the completed classification methods
        # e.g.
        # return self.classifier.classify(data)
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()
