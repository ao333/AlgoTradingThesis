import numpy as np
import Learners.LinReg as lrl
from Learners import Bag as bl


class InsaneLearner(object):
    def __init__(self, verbose=False):
        self.learners = []
        self.bags = 20
        for i in range(self.bags):
            # Create a learner containing 20*20 linear regression learners
            self.learners.append(bl.BagLearner(learner=lrl.LinRegLearner, kwargs={}, bags=20, boost=False, verbose=False))

    def addEvidence(self, dataX, dataY):
        for i in range(self.bags):
            self.learners[i].addEvidence(dataX, dataY)

    def query(self, points):
        Y = []  # initialize pred Y
        for i in range(self.bags):  # rows = no. of test examples, columns = no. of individual bag learners
            Y.append(self.learners[i].query(points))  # predictions for each baglearner in rows of pred Y
        return np.mean(Y, axis=0)  # return mean of all learners in 1D-array


if __name__ == '__main__':
    data = np.genfromtxt('data/Istanbul.csv', delimiter=',')
    data = data[1:, 1:]
    # np.random.shuffle(data) #  Don't shuffle to compare output with single linreglearner
    split = int(0.6 * data.shape[0])  # 60-40 break into train-test sets
    trainX = data[:split, :-1]
    trainY = data[:split, -1]  # last column is labels
    testX = data[split:, :-1]
    testY = data[split:, -1]  # last column is labels

    learner = InsaneLearner(verbose=False)  # constructor for InsaneLearner
    learner.addEvidence(trainX, trainY)

    Y = learner.query(trainX)  # get the predictions
    rmse1 = np.sqrt(((Y - trainY)**2).sum() / trainY.shape[0])
    corr = np.corrcoef(Y, trainY)
    print("In sample results")
    print("RMSE: ", rmse1)
    print("corr: ", corr[0, 1])

    Y = learner.query(testX)  # get the predictions
    rmse2 = np.sqrt(((Y - testY)**2).sum() / testY.shape[0])
    corr = np.corrcoef(Y, testY)
    print
    print("Out of sample results")
    print("RMSE: ", rmse2)
    print("corr: ", corr[0, 1])
