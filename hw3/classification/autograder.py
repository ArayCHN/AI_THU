#!/usr/bin/python2.7

import sys, time, traceback
import os

import classifiers
from featureExtractor import *
import answers
from test_cases import solution_classifiers, solution_featureExtractor

## code to handle timeouts
import signal


# check whether sklearn is used
def checkSklearnPackage(grade, q):
    if 'sklearn' in sys.modules:
        grade[q] = 0
        raise Exception("Python packages sklearn is abandoned in this project (expect Question 6)")


class TimeoutFunctionException(Exception):
    """Exception to raise on a timeout"""
    pass


class TimeoutFunction:

    def __init__(self, function, timeout):
        "timeout must be at least 1 second. WHY??"
        self.timeout = timeout
        self.function = function

    def handle_timeout(self, signum, frame):
        raise TimeoutFunctionException()

    def __call__(self, *args):
        if not 'SIGALRM' in dir(signal):
            return self.function(*args)
        old = signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.timeout)
        # try:
        result = self.function(*args)
        # finally:
        # signal.signal(signal.SIGALRM, old)
        signal.alarm(0)
        return result


def getExceptionTraceBack():
    return ''.join(traceback.format_exception(*sys.exc_info())[-2:]).strip()     # .replace('\n',': ')


def display(str, minor = False):
    global allPassed
    if (not minor):
        allPassed = False
    str = str.replace("\n"," &&&\n&&& ")
    print "&&& %s &&&" % str


numTraining = 5000
numTest = 1000
DIGIT_DATUM_WIDTH = 28
DIGIT_DATUM_HEIGHT = 28


def readDigitData(trainingSize=5000, testSize=1000):
    rootdata = 'data/digitdata/'
    # loading digits data
    rawDigitData = np.load(rootdata+'mnist.npz')
    rawTrainingData = rawDigitData['x_train'][:trainingSize]
    trainingLabels = rawDigitData['y_train'][:trainingSize]
    rawValidationData = rawDigitData['x_valid'][:testSize]
    validationLabels = rawDigitData['y_valid'][:testSize]
    rawTestData = rawDigitData['x_test'][:testSize]
    testLabels = rawDigitData['y_test'][:testSize]
    return (rawTrainingData, trainingLabels, rawValidationData, validationLabels, rawTestData, testLabels)


def getAccuracy(data, classifier, featureExtractor=BasicFeatureExtractorDigit()):
    rawTrainingData, trainingLabels, rawValidationData, validationLabels, rawTestData, testLabels = data
    featureExtractor.fit(rawTrainingData)
    trainingData = featureExtractor.extract(rawTrainingData)
    validationData = featureExtractor.extract(rawValidationData)
    testData = featureExtractor.extract(rawTestData)
    classifier.train(trainingData, trainingLabels, validationData, validationLabels)
    guesses = classifier.classify(testData)
    correct = [guesses[i] == testLabels[i] for i in range(len(testLabels))].count(True)
    acc = 100.0 * correct / len(testLabels)
    print str(correct), ("correct out of " + str(len(testLabels)) + " (%.1f%%).") % (acc)
    return acc

digitData = readDigitData(numTraining, numTest)


def q1(grades):
    print "\n===== Grading Q1 ===== "

    grades[1] = 0

    print "1. checking python dataClassifier.py -d digits -c knn -t 5000 -s 1000 -f basic -n 5"
    try:
        legalLabels = range(10)
        classifier = classifiers.KNNClassifier(legalLabels, 5)
        accDigits = getAccuracy(digitData, classifier) # our solution: 91.0%
        accThres = [50, 70, 80, 87]
        score = [accDigits >= t for t in accThres].count(True)
        grades[1] = score
        checkSklearnPackage(grades, 1)
    except:
        display("An exception was raised:\n%s" % getExceptionTraceBack())


def q2(grades):
    print "\n===== Grading Q2 ===== "

    grades[2] = 0

    print "2. checking python dataClassifier.py -d digits -c perceptron -t 5000 -s 1000 -f basic -i 50"
    try:
        legalLabels = range(10)
        classifier = classifiers.PerceptronClassifier(legalLabels, 50)
        accDigits = getAccuracy(digitData, classifier)  # our solution: 87.4%
        accThres = [40, 60, 70, 78, 83]
        score = [accDigits >= tt for tt in accThres].count(True)
        grades[2] = score

        checkSklearnPackage(grades, 2)
    except:
        display("An exception was raised:\n%s" % getExceptionTraceBack())


def q3(grades):
    print "\n===== Grading Q3 ===== "
    rawTrainingData, trainingLabels, _, _, _, _ = digitData

    grades[3] = 0

    print "3. checking python dataClassifier.py -d digits -c svm -t 5000 -s 1000 -f basic -i 50 -k rbf -C 10"
    try:
        legalLabels = range(10)

        # test SVMClassifier.trainSVM (3 points)
        classifier = classifiers.SVMClassifier(legalLabels, 1, 10.0, 'rbf')
        classifier.sigma = 10.0
        classifier.testing = True
        solution_classifier = solution_classifiers.SVMClassifier(legalLabels, 1, 10.0, 'rbf')
        solution_classifier.sigma = 10.0
        solution_classifier.testing = True
        # synthetic data
        testTrainingLabels = np.where(trainingLabels==0, 1, -1)
        testTrainingData = rawTrainingData
        classifier.trainSVM(testTrainingData, testTrainingLabels)
        solution_classifier.trainSVM(testTrainingData, testTrainingLabels)
        student_answer = np.load('test_cases/student_test_cqp.npz')
        solution_answer = np.load('test_cases/solution_test_cqp.npz')
        for key in student_answer.files:
            student_ans = student_answer[key]
            solution_ans = solution_answer[key]
            if key != 'E':
                diff = np.mean(np.abs(student_ans - solution_ans))
            else:
                diff = min(np.mean(np.abs(student_ans + solution_ans)), np.mean(np.abs(student_ans - solution_ans)))
            if diff > 1e-4:
                raise Exception("""TEST FAILED: error computing array {} in SVMClassifier.trainSVM :\n
                                    Student answer: \n {} \n
                                    Correct answer: \n {} \n
                                    Mean Difference: {} \n
                                """
                                .format(key, student_ans, solution_ans, diff))

        grades[3] += 3

        student_answer.close()
        solution_answer.close()
        os.remove('test_cases/student_test_cqp.npz')
        os.remove('test_cases/solution_test_cqp.npz')

        # test classification (3 points)
        classifier = classifiers.SVMClassifier(legalLabels, 5000, 10.0, 'rbf')
        accDigits = getAccuracy(digitData, classifier)  # our solution: 94.2%
        accThres = [60, 80, 90]
        score = [accDigits >= tt for tt in accThres].count(True)
        grades[3] += score
        checkSklearnPackage(grades,3)
    except:
        display("An exception was raised:\n%s" % getExceptionTraceBack())


def q4(grades):
    print "\n===== Grading Q4 ===== "
    rawTrainingData, trainingLabels, rawValidationData, validationLabels, rawTestData, testLabels = digitData

    print "4. checking python dataClassifier.py -d digits -f pca -m 32"
    grades[4] = 0
    try:
        featureExt = PCAFeatureExtractorDigit(32)
        solution_featureExt = solution_featureExtractor.PCAFeatureExtractorDigit(32)
        featureExt.fit(rawTrainingData)
        solution_featureExt.fit(rawTrainingData)
        feature = featureExt.extract(rawTestData)
        solution_feature = solution_featureExt.extract(rawTestData)
        diff = np.mean(np.abs(feature - solution_feature))
        if np.mean(np.abs(feature - solution_feature)) > 1e-3:
            raise Exception("""TEST FAILED: error performing PCA in PCAFeatureExtractorDigit.extract: \n
                                Student answer: \n {} \n,
                                Correct answer: \n {} \n
                                Mean difference: {} \n
                            """
                            .format(feature, solution_feature, diff))
        grades[4] += 3
        reconstructed = featureExt.reconstruct(feature)
        solution_reconstructed = solution_featureExt.reconstruct(solution_feature)
        diff = np.mean(np.abs(reconstructed - solution_reconstructed))
        if np.mean(np.abs(feature - solution_feature)) > 1e-3:
            raise Exception("""TEST FAILED: error performing PCA in PCAFeatureExtractorDigit.reconstruct: \n
                                Student extracted PCA feature: \n {} \n,
                                Correct answer: \n {} \n
                                Mean difference: {} \n
                            """
                            .format(feature, solution_feature, diff))
        grades[4] += 1
        checkSklearnPackage(grades, 4)
    except:
        display("An exception was raised:\n%s" % getExceptionTraceBack())


if __name__ == '__main__':
    np.random.seed(0)
    NUM_TESTS = 4
    USAGE_STRING = """
      USAGE:      python classificationAutograder.py <options>
      EXAMPLES:   python classificationAutograder.py -q 1,2 -o score
                      - would test question 1,2, and the output would be in file score
                  python classificationAutograder.py
                      - would test all questions and print the output on stdout
                     """

    from optparse import OptionParser

    parser = OptionParser(USAGE_STRING)

    parser.add_option('-o', '--output', help='Save the autograder results to file output', default='stdout')
    parser.add_option('-q', '--q', help='Test No. of question. If 0, then test all questions', default='0')

    options, otherjunk = parser.parse_args(sys.argv[1:])
    if len(otherjunk) != 0: raise Exception('Command line input not understood: ' + str(otherjunk))

    if options.output != 'stdout':
        sys.stdout = file(sys.argv[1], 'w')
        sys.stderr = sys.stdout
    print 'Autograder transcript for project 3\n', 'Starting on %d-%d at %d:%02d:%02d' % time.localtime()[1:6]
    grades = {}
    print '\n+++++++++++++++++++++++++++++++++++++++'
    print '+++ AUTOGRADING FOR THIS SUBMISSION +++'
    print '+++++++++++++++++++++++++++++++++++++++'
    if options.q == '0':
        tests = range(1,NUM_TESTS+1)
    else:
        tests = [int(t) for t in options.q.strip().split(',')]
        for t in tests:
            if t <= 0 or t > NUM_TESTS:
                raise Exception('Test No. of question illegal.')
    for t in tests:
        TimeoutFunction(getattr(sys.modules[__name__], 'q'+str(t)), 1200)(grades)
    print "\n"
    full_credits = {1:4, 2:5, 3:6, 4:4}
    for t in tests:
      print "GRADE FOR QUESTION ",t,": ", grades[t], '/', full_credits[t]
    print '+++++++++++++++++++++++++++++++++++++++'
    print 'TOTAL: %d/%d' % (sum([grades[g] for g in tests]), sum([full_credits[g] for g in tests]))
    print '+++++++++++++++++++++++++++++++++++++++'
    print '@@@PASSED SANITY CHECKS@@@'
    print 'Done with Autograder on  %d-%d at %d:%02d:%02d' % time.localtime()[1:6]
    sys.stdout.close
