# classifiers.py
# -------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import util
from classificationMethod import ClassificationMethod
import numpy as np
from scipy.optimize import fmin_slsqp, fmin_l_bfgs_b


class LinearRegressionClassifier(ClassificationMethod):
    """
    Classifier with Linear Regression.
    """
    def __init__(self, legalLabels):
        """

        :param legalLabels: Labels to predict (for digit data, legalLabels = range(10))
        """
        super(LinearRegressionClassifier, self).__init__(legalLabels)
        self.legalLabels = legalLabels
        self.type = 'lr'
        self.lambda_ = 1e-4
        self.weights = None

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """
        Train the Linear Regression Classifier.

        For digit data, trainingData/validationData are all in numpy format with size ([number of data], 784)
        For doc data, trainingData/validationData should also be in numpy format.
        """
        n, dim = trainingData.shape
        X = trainingData
        Y = np.zeros((n, len(self.legalLabels)))
        Y[np.arange(n), trainingLabels] = 1
        self.weights = np.dot(np.linalg.inv(np.dot(X.T, X) + self.lambda_*np.eye(dim)), np.dot(X.T, Y))
    
    def classify(self, data):
        """
        Predict which class is in.
        :param data: data to classify which class is in. (in numpy format)
        :return list or numpy array
        """
        return np.argmax(np.dot(data, self.weights), axis=1)


class KNNClassifier(ClassificationMethod):
    """
    KNN Classifier.
    """
    
    def __init__(self, legalLabels, num_neighbors):
        """

        :param legalLabels: Labels to predict (for digit data, legalLabels = range(10))
        :param num_neighbors: number of nearest neighbors.
        """
        super(KNNClassifier, self).__init__(legalLabels)
        self.legalLabels = legalLabels
        self.type = 'knn'
        self.num_neighbors = num_neighbors
    
    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """
        Train the Linear Regression Classifier by just storing the trainingData and trainingLabels.

        For digit data, trainingData/validationData are all in numpy format with size ([number of data], 784)
        For doc data, trainingData/validationData should also be in numpy format.
        """

        # trainingData is normalized
        self.trainingData = trainingData / np.linalg.norm(trainingData, axis=1).reshape((len(trainingData), 1))
        self.trainingLabels = trainingLabels
    
    def classify(self, data):
        """
        Predict which class is in.

        Some numpy functions that may be of use (we consider np as short of numpy)
        np.sum(a, axis): sum of array elements over a given axis.
        np.dot(A, B): dot product of two arrays, or matrix multiplication between A and B.
        np.sort, np.argsort: return a sorted copy (or indices) of an array.

        :param data: Data to classify which class is in. (in numpy format)
        :return Determine the class of the given data (list or numpy array)
        """

        data = data / np.linalg.norm(data, axis=1).reshape((len(data), 1))

        "*** YOUR CODE HERE ***"
        # should compute sim(data[i], data[j]) = dot(data[i], data[j])
        ans = []
        for k in range(0, data.shape[0]):
            sim = []
            for i in range(0, len(self.trainingData)):
                sim.append( (np.dot(data[k], self.trainingData[i]), self.trainingLabels[i]) )
            # sim[n][0]: the similarity; sim[k][1]: the number

            sim = sorted(sim, key = lambda x: x[0])
            sim.reverse()
            stat = {} # stat[label]: the similarity between data[k] and label
            for i in range(0, 5):
                if stat.has_key(sim[i][1]):
                    stat[sim[i][1]] += 1
                else:
                    stat[sim[i][1]] = 1
            maxima = 0
            for label in self.trainingLabels:
                if stat.has_key(label) and stat[label] > maxima:
                    maxima = stat[label]
                    tmp = label
            ans.append(tmp)
        return ans

        util.raiseNotDefined()


class PerceptronClassifier(ClassificationMethod):
    """
    Perceptron classifier.
    """
    def __init__( self, legalLabels, max_iterations):
        """
        self.weights/self.bias: parameters to train, can be considered as parameter W and b in a perception.
        self.batchSize: batch size in a mini-batch, used in SGD method
        self.weight_decay: weight decay parameters.
        self.learningRate: learning rate parameters.

        :param legalLabels: Labels to predict (for digit data, legalLabels = range(10))
        :param max_iterations: maximum epoches
        """
        super(PerceptronClassifier, self).__init__(legalLabels)
        self.legalLabels = legalLabels
        self.type = "perceptron"
        self.max_iterations = max_iterations
        self.weights = None
        self.bias = None
        self.batchSize = 100
        self.weight_decay = 1e-3 # lambda, L-2 penalize
        self.learningRate = 1 # the length of each step
        
    def setWeights(self, input_dim):
        self.weights = np.random.randn(input_dim, len(self.legalLabels))/np.sqrt(input_dim)
        self.bias = np.zeros(len(self.legalLabels))
    
    def prepareDataBatches(self, traindata, trainlabel):
        """
        Generate data batches with given batch size(self.batchsize)

        :return a list in which each element are in format (batch_data, batch_label). E.g.:
            [(batch_data_1, batch_label_1), (batch_data_2, batch_label_2), ..., (batch_data_n, batch_label_n)]

        """
        index = np.random.permutation(len(traindata))
        traindata = traindata[index]
        trainlabel = trainlabel[index]
        split_no = int(len(traindata) / self.batchSize)
        return zip(np.split(traindata[:split_no*self.batchSize], split_no), np.split(trainlabel[:split_no*self.batchSize], split_no))

    def train(self, trainingData, trainingLabels, validationData, validationLabels ):
        """
        The training loop for the perceptron passes through the training data several
        times and updates the weight vector for each label based on classification errors.
        See the project description for details.

        For digit data, trainingData/validationData are all in numpy format with size ([number of data], 784)
        For doc data, trainingData/validationData should also be in numpy format.

        Some data structures that may be in use:
        self.weights/self.bias (numpy format): parameters to train,
            can be considered as parameter W and b in a perception.
        self.batchSize (scalar): batch size in a mini-batch, used in SGD method
        self.weight_decay (scalar): weight decay parameters.
        self.learningRate (scalar): learning rate parameters.

        Some numpy functions that may be of use (we consider np as short of numpy)
        np.sum(a, axis): sum of array elements over a given axis.
        np.dot(A, B): dot product of two arrays, or matrix multiplication between A and B.
        np.mean(a, axis): mean value of array elements over a given axis
        np.exp(a)
        """

        self.setWeights(trainingData.shape[1]) # bias and weight have both been initialized
        # DO NOT ZERO OUT YOUR WEIGHTS BEFORE STARTING TRAINING, OR
        # THE AUTOGRADER WILL LIKELY DEDUCT POINTS.
        
        # Hyper-parameters. Your can reset them. Default batchSize = 100, weight_decay = 1e-3, learningRate = 1
        "*** YOU CODE HERE ***"
        self.batchSize = 100.0
        self.weight_decay = 1e-3
        self.learningRate = 1
        k = self.batchSize
        l = len(self.legalLabels)

        for iteration in range(self.max_iterations):
            print "Starting iteration ", iteration, "..."
            dataBatches = self.prepareDataBatches(trainingData, trainingLabels)
            for batchData, batchLabel in dataBatches: # 100 instances
                "*** YOUR CODE HERE ***"
                W = self.weights
                b = self.bias
                numerators = np.exp(np.dot(W.T, batchData.T) + b[:, None]) # an array of numerators in formula exp(***)
                denominator = np.sum(numerators, axis = 0)
                probablity = numerators / denominator # the probablity list 0..9 for xi
                for i in range(0, len(batchLabel)):
                    probablity [batchLabel[i], i] -= 1
                pb = np.sum(probablity, axis = 1)
                self.bias = (1 - self.weight_decay*self.learningRate) * b - self.learningRate / k * pb
                pw = np.dot(probablity, batchData)
                self.weights = (1 - self.weight_decay*self.learningRate) * W - self.learningRate / k * pw.T

    def classify(self, data):
        """
        :param data: Data to classify which class is in. (in numpy format)
        :return Determine the class of the given data (list or numpy array)
        """
        
        return np.argmax(np.dot(data, self.weights) + self.bias, axis=1)

    def visualize(self):
        sort_weights = np.sort(self.weights, axis=0)
        _min = 0
        _max = sort_weights[-10]
        return np.clip(((self.weights-_min) / (_max-_min)).T, 0, 1)


class SVMClassifier(ClassificationMethod):
    """
    SVM Classifier
    """
    def __init__(self, legalLabels, max_iterations=3000, C=1.0, kernelType='rbf'):
        """
        self.sigma: \sigma value in Gaussian RBF kernel.
        self.support/sulf.support_vectors: support vectors and support(y*\alpha). May be in list or dict format.

        :param legalLabels: Labels to predict (for digit data, legalLabels = range(10))
        :param max_iterations: maximum iterations in optimizing constrained QP problem
        :param C: value C in SVM
        :param kernelType: kernel type. Only 'rbf' or 'linear' are valid
        """
        super(SVMClassifier, self).__init__(legalLabels)
        self.type = 'svm'
        self.legalLabels = legalLabels
        self.max_iterations = max_iterations
        self.C = C
        self.kernelType = kernelType
        self.sigma = 1.0
        
        # may be used in training
        self.support = None
        self.support_vectors = None
        self.biases = None

        # DO NOT change self.testing, or you MAY NOT PASS THE AUTOGRADER
        self.testing = False
    
    def optimizeConstrainedQuad(self, x, A, b, bounds, E, e, debug=False):
        """
        min 1/2 x^T A x + b^T x
        s.t. bounds[i][0] <= x_i <= bounds[i][1]
             E x = e

        :param x : vector of dimension n;
        :param A : matrix of dimension n*n;
        :param: bounds: list of vector, each vector with size 2, length of list is n.
        :param: E: matrix of size m*n
        :param e: vector of size m
        :param debug: whether to output the intermediate results during optimization
        :return optimized x
        """

        if len(E.shape) == 1:
            E = E.reshape((1, E.shape[0])) # to make E a matrix instead of a line vector
            e = np.array(e).reshape(1) # to make e a matrix instead of a single number
            
        assert x.shape[0]==A.shape[0] and x.shape[0]==A.shape[1]
        assert x.shape[0]==E.shape[1]
        assert x.shape[0]==len(bounds)
        assert sum(len(bnd)==2 for bnd in bounds)==len(bounds)
        assert E.shape[0]==e.shape[0]

        if self.testing:
            np.savez('test_cases/student_test_cqp.npz', A=np.array(A), b=np.array(b), bounds=np.array(bounds), E=np.array(E), e=np.array(e))
        
        n = x.shape[0]
        func = lambda x: 0.5*np.dot(x, np.dot(A, x))  + np.dot(b, x)
        f_eqcons = lambda x: np.dot(E, x) - e
        bounds = bounds
        fprime = lambda x: np.dot(A, x) + b
        fprime_eqcons = lambda x: E
        func_w_eqcon = lambda x: func(x) + n*5e-5*np.sum(f_eqcons(x)**2)
        fprime_w_eqcon = lambda x: fprime(x) + n*1e-4*np.dot(f_eqcons(x), fprime_eqcons(x))
        max_iters = self.max_iterations
        iprint = 90 if debug else 0
        res = fmin_l_bfgs_b(func_w_eqcon, x, fprime=fprime_w_eqcon, bounds=bounds, maxfun=max_iters, maxiter=max_iters, factr=1e10, iprint=iprint)
        print 'F = %.4f Eqcons Panelty = %.4f' % (func(res[0]), np.sum(f_eqcons(res[0])**2))
        return res[0]

    def generateKernelMatrix(self, data1, data2=None):
        """
        Generate a kernel. Linear Kernel and Gaussian RBF Kernel is provided.

        :param data1: in numpy format
        :param data2: in numpy format
        :return:
        """
        if data2 is None:
            data2 = data1
        if self.kernelType == 'rbf':
            X12 = np.sum(data1*data1, axis=1, keepdims=True)
            X22 = np.sum(data2*data2, axis=1, keepdims=True)
            XX = 2*np.dot(data1, data2.T) - X12 - X22.T
            XX = np.exp(XX / (2*self.sigma*self.sigma))
        elif self.kernelType == 'linear':
            XX = np.dot(data1, data2.T)
        else:
            raise Exception('Unknown kernel type: ' + str(self.kernelType))
        return XX
    
    def trainSVM(self, trainData, trainLabels):
        """
        Train SVM with just two labels: 1 and -1

        :param traindata: in numpy format
        :param trainLabels: in numpy format

        Some functions that may be of use:
        self.optimizeConstrainedQuad: solve constrained quadratic programming problem
        self.generateKernelMatrix: Get kernel matrix given specific type of kernel
        """
        assert len(trainData) == len(trainLabels)
        assert (np.sum(trainLabels==1) + np.sum(trainLabels==-1)) == len(trainLabels)
        alpha = np.zeros(len(trainData))
        KernelMat = self.generateKernelMatrix(trainData)
        y = trainLabels
        A = np.dot(np.array([y]).T, np.array([y])) * KernelMat
        b = np.linspace(-1.0, -1.0, len(trainData))
        bounds = np.zeros((len(trainData), 2))
        bounds[:] = np.array([0, self.C])
        # print 'alpha:', alpha # a1;a2;a3;...
        # print 'A:', A # yi*yj*K(xi, xj)
        # print 'b:', b # 1;1;1;1;1...
        # print 'bounds:', bounds # (0, C)
        # print 'y:', y # 1, -1, ...
        # optimizeConstrainedQuad(x, A, b, bounds, E, e):
        alpha = self.optimizeConstrainedQuad(alpha + 0.0, A, b, bounds, y + 0.0, 0.0)
        "*** YOUR CODE HERE ***"
        tot = 0
        SUM = 0.0
        # print alpha
        for i in range(0, len(alpha)):
            if alpha[i] > 1e-6 and alpha[i] < self.C - 1e-6: # avoid precision issue
                tot += 1.0
                deduction = 0
                # for j in range(0, len(alpha)):
                #     deduction += alpha[j]*y[j]*KernelMat[j, i]
                SUM += ( y[i] - np.sum(alpha * y * KernelMat[:, i].T) )
        bias = SUM / tot
        # print 'bias:',bias
        return alpha, bias

        # util.raiseNotDefined()
    
    def train( self, trainingData, trainingLabels, validationData, validationLabels ):
        """
        ovr(one vs. the rest) training with SVM

        For digit data, trainingData/validationData are all in numpy format with size ([number of data], 784)
        For doc data, trainingData/validationData should also be in numpy format.

        Some numpy functions that may be of use (we consider np as short of numpy)
        np.where(condition, x, y): Return elements, either from x or y, depending on condition.
        np.mean(a, axis): mean value of array elements over a given axis
        indexing, slicing in numpy may be important
        """

        # determine sigma with training data
        # import sklearn # sklearn is not allowed!
        X = trainingData[:1000]
        X2 = np.sum(X*X, axis=1, keepdims=True)
        self.sigma = np.sqrt(np.mean(-2*np.dot(X,X.T)+X2+X2.T))
        # DID NOT USE VALIDATION DATA?
        self.support = {}
        self.support_vectors = {}
        self.biases = {}
        for t in self.legalLabels:
            # for each class, use ovr to train SVM classifier
            print 'classify label', t, '...'
            traindata = trainingData
            trainlabels = np.where(trainingLabels==t, 1, -1) # [1,-1,1,...], 1 is the place where label is this one
            # To avoid the precision loss underlying the floating point,
            # we recommend use (alpha > 1e-6 + beta) to determine whether alpha is greater than beta,
            # and (alpha < beta - 1e-6) to determine whether alpha is smaller than beta,
            # and (abs(alpha - beta) < 1e-6) to determine wheter alpha is equal to beta.
            "*** YOUR CODE HERE ***"
            self.support_vectors = traindata #
            self.support[t] = trainlabels #
            self.biases[t] = self.trainSVM(traindata, trainlabels) # alpha, bias
            # util.raiseNotDefined()
    
    def classify(self, data): # data is a bunch of tests
        """
        ovr(one vs. the rest) classification with SVM
        """
        "*** YOUR CODE HERE ***"
        # x = data[i, :] # x is a single test case
        # KernelMat = self.generateKernelMatrix(self.support_vectors, x) # K(x_i, x)
        # for t in self.legalLabels:
        #     f = np.sum(self.support[t] * self.biases[t][0] * KernelMat.T) + self.biases[t][1] # the f_value f_t(x)
        #     # maxima = -999999
        #     # if f > maxima:
        #     #     maxima = f
        #     #     ans = t
        #     print f
        #     maxima = np.linspace(-99999, -99999, data.shape[1])
        #     ans = np.linspace(0, 0, data.shape[1])
        #     for i in range(0, data.shape[1]):
        #         if f[i] > maxima[i]:
        #             maxima[i] = f[i]
        #             ans[i] = t
        # return ans
        ans = []
        # print data.shape
        for k in range(0, data.shape[0]):

            x = data[k, :] # x is a single test case
            # print x
            KernelMat = self.generateKernelMatrix(self.support_vectors, np.array([x])) # K(x_i, x)
            # print KernelMat
            maxima = -999999
            for t in self.legalLabels:
                f = np.sum(self.support[t] * self.biases[t][0] * KernelMat.T) + self.biases[t][1] # the f_value f_t(x)
                # print self.support[t]
                # print self.biases[t][0]
                # print self.biases[t][1]
                # print f
                if f > maxima:
                    maxima = f
                    tmp = t
            ans.append(tmp)
        # print ans
        # print len(ans)
        return ans

        util.raiseNotDefined()
