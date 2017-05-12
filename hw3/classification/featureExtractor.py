# featureExtractor.py

import sys
import util
import numpy as np
import display


class BaseFeatureExtractor(object):
    def __init__(self):
        pass
    
    def fit(self, trainingData):
        """
        Train feature extractor given the training Data
        :param trainingData: in numpy format
        :return:
        """
        pass
    
    def extract(self, data):
        """
        Extract the feature of data
        :param data: in numpy format
        :return: features, in numpy format and len(features)==len(data)
        """
        pass
    
    def visualize(self, data):
        pass


class BasicFeatureExtractorDigit(BaseFeatureExtractor):
    """
    Just regard the value of the pixels as features (in 784 dimensions)
    """
    def __init__(self):
        super(BasicFeatureExtractorDigit, self).__init__()

    def fit(self, trainingData):
        pass
    
    def extract(self, data):
        return data
    
    def visualize(self, data):
        # reconstruction and visualize
        display.displayDigit(data, outfile='visualize/original_digits.png')

    
class PCAFeatureExtractorDigit(BaseFeatureExtractor):
    """
    Principle Component Analysis(PCA)
    """
    def __init__(self, dimension):
        """
        self.weights: weight to learn in PCA, in numpy format and shape=(dimension, 784)
        self.mean: mean of training data, in numpy format

        :param dimension: dimension to reduction
        """
        super(PCAFeatureExtractorDigit, self).__init__()
        self.dimension = dimension
        self.weights = None
        self.mean = None
    
    def fit(self, trainingData):
        """
        Train PCA given the training Data

        Some numpy functions that may be of use (we consider np as short of numpy)
        np.mean(a, axis): mean value of array elements over a given axis
        np.linalg.svd(X, full_matrices=False): perform SVD decomposition to X
        np.dot(A, B): dot product of two arrays, or matrix multiplication between A and B.

        :param trainingData: in numpy format
        :return:
        """
        self.mean = np.mean(trainingData, axis=0)
        data = trainingData - self.mean
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()
    
    def extract(self, data):
        """

        :param data: in numpy format
        :return: features, in numpy format, features.shape = (len(data), self.dimension)
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()
    
    def reconstruct(self, pcaData):
        """
        Perform reconstruction of data given PCA features

        :param pcaData: in numpy format, features.shape[1] = self.dimension
        :return: originalData, in numpy format, originalData.shape[1] = 784
        """
        assert pcaData.shape[1]==self.dimension
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def visualize(self, data):
        """
        Visualize data with both PCA and reconstruction
        :param data: in numpy format
        :return:
        """
        # extract features
        pcaData = self.extract(data)
        # reconstruction and visualize
        reconstructImg = self.reconstruct(pcaData)
        display.displayDigit(np.clip(reconstructImg, 0, 1), outfile='visualize/pca_digits.png')


class KMeansClusterDigit(BaseFeatureExtractor):
    """
    K-means clustering
    """
    def __init__(self, num_cluster, num_iterations):
        """
        :param num_cluster: number of clusters
        :param num_iterations: number of iterations
        """
        super(KMeansClusterDigit, self).__init__()
        self.num_cluster = num_cluster
        self.num_iterations = num_iterations
        self.clusters = None
    
    def fit(self, trainingData):
        cluster_no = np.random.randint(self.num_cluster, size=(len(trainingData)))
        self.clusters = np.zeros((self.num_cluster, trainingData.shape[1]))
        for iteration in range(self.num_iterations):
            print 'iteration', iteration, '...'
            c = [[] for t in range(self.num_cluster)]
            for i, n in enumerate(cluster_no):
                c[n].append(i)
            for i in range(self.num_cluster):
                self.clusters[i] = np.mean(trainingData[c[i]], axis=0)
            Cls2 = np.sum(self.clusters*self.clusters, axis=1)
            D = -2*np.dot(trainingData, self.clusters.T) + Cls2
            new_cluster_no = np.argmin(D, axis=1)
            if np.sum(new_cluster_no==cluster_no) == len(new_cluster_no):
                break
            cluster_no = new_cluster_no
        
    def visualize(self, data):
        Cls2 = np.sum(self.clusters*self.clusters, axis=1)
        clusters = np.zeros(self.clusters.shape)
        occupy = np.zeros(len(self.clusters), dtype=np.int32)
        D = -2*np.dot(data, self.clusters.T) + Cls2
        ind = np.argsort(D, axis=1)
        ind2 = np.argsort(np.min(D, axis=1))
        for i in ind2:
            for t in ind[i]:
                if occupy[t] == 0:
                    print i, t
                    occupy[t] = 1
                    clusters[i] = self.clusters[t]
                    break
        return np.array(clusters)
