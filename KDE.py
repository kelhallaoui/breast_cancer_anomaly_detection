import numpy as np
from sklearn.neighbors.kde import KernelDensity
from ImportData import getData, separateData, separateDataOut
import matplotlib.pyplot as plt

class KDE(object):
    initial = 0
    def __init__(self):
        self._initial =  1
        self._scores_model = None
        self._scores_all = None

    @property
    def scores_model(self):
        return self._scores_model
    @property
    def scored_all(self):
        return self._scores_all
    @property
    def mean_std(self):
        return np.mean(self._scores_model), np.std(self._scores_model)


    def trainKDE(self, X_healthy, TX1RX2):
        # Makes a model that fits a gaussian to the data set only using the healthy cases
        kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(X_healthy)
        # Score is the log likelihood that the data comes from the fitted data
        self._scores_model = kde.score_samples(X_healthy)
        self._scores_all = kde.score_samples(TX1RX2)

        mean_healthy, standev_healthy = self.mean_std
        return self._scores_model, self._scores_all, mean_healthy, standev_healthy

    def getTheta(self, TXRX, labels):
        inter = 100
        thetas = np.zeros((inter))

        for j in range(0, inter):
            X_healthy, X_healthy_index, patient_out = separateData(2, TXRX, labels) # 2 indicates that I want to leave_out 2 patients

            kde = KDE()
            scores_model, scores_all, mean_healthy, standev_healthy = KDE.trainKDE(kde, X_healthy, TXRX)

            precision_outs = np.zeros((500,1))
            precision_totals = np.zeros((500,1))

            for theta in range(0,500):
                classify = np.zeros((96,1))


                for i in range(0, 96):
                    if(scores_all[i] < mean_healthy+theta*standev_healthy and scores_all[i] > mean_healthy-theta*standev_healthy):
                        classify[i] = -1
                    else:
                        classify[i] = 1

                precision_out = 0
                precision_total = 0
                for i in range(0, 96):
                    if(classify[i] == labels[i,0] and (labels[i,1] in patient_out)):
                        precision_out = precision_out + 1
                    if(classify[i] == labels[i,0]):
                        precision_total = precision_total + 1

                total = 0
                for i in range(0,2):
                    total = total + list(labels[:,1]).count(patient_out[i])

                precision_totals[theta] =  precision_total/96
                precision_outs[theta] = precision_out/total



            index_max = list(precision_totals).index(max(precision_totals))
            index_min = list(precision_totals[index_max:500]).index(min(precision_totals[index_max:500]))
            if(index_min > index_max):
                thetas[j] = list(precision_outs).index(max(precision_outs[index_max:index_min]))
            else:
                j=j-1

        #plt.plot(thetas)
        #plt.show(block=False)
        return np.mean(thetas)

    def trainAllThetas(self):
        thetas = np.zeros((240, 1))
        index = 0
        for i in range(1,17):
            for j in range(1,17):
                if(i!=j):
                    #print(str(i) + ' **** ' + str(j))
                    TXRX, labels = getData('TX'+str(i)+'RX'+str(j))
                    X_healthy, X_healthy_index, patient_out = separateData(2, TXRX, labels)
                    theta = self.getTheta(TXRX, labels)
                    thetas[index] = theta
                    index = index + 1
                    print(theta)

        return thetas

    def classify240(self, patient_out, thetas):
        classification = np.zeros((96,240))
        index = 0
        for i in range(1,17):
            for j in range(1,17):
                if(i!=j):
                    kde = KDE()
                    TXRX, labels = getData('TX'+str(i)+'RX'+str(j))
                    X_healthy, X_healthy_index, patient_out = separateDataOut(TXRX, labels, patient_out)
                    theta = thetas[index]
                    classification[:,index] = np.resize(self.classified(theta, X_healthy, TXRX, labels, patient_out, kde), (96,))
                    index = index + 1

        decision=np.zeros((96,1))

        for i in range(0,96):
            if(list(classification[i,:]).count(-1)>list(classification[i,:]).count(1)):
                decision[i] = -1
            elif(list(classification[i,:]).count(-1)<list(classification[i,:]).count(1)):
                decision[i] = 1
            else:
                random = np.random.randint(0,2)
                if(random == 0):
                    decision[i] = -1
                else:
                    decision[i] = 1

        return classification, decision, labels


    def classified(self, theta, X_healthy, TXRX, labels, patient_out, kde):
        kde = KDE()
        scores_model, scores_all, mean_healthy, standev_healthy = KDE.trainKDE(kde, X_healthy, TXRX)

        classify = np.zeros((96,1))
        for i in range(0, 96):
            if(scores_all[i] < mean_healthy+theta*standev_healthy and scores_all[i] > mean_healthy-theta*standev_healthy):
                classify[i] = -1
            else:
                classify[i] = 1
        return classify

    def getPrecision(self, classify, labels, patient_out):
        precision_out = 0
        precision_total = 0
        for i in range(0, 96):
            if(classify[i] == labels[i,0] and (labels[i,1] in patient_out)):
                precision_out = precision_out + 1
            if(classify[i] == labels[i,0]):
                precision_total = precision_total + 1

        total = 0
        for i in range(0,2):
            total = total + list(labels[:,1]).count(patient_out[i])

        return precision_out/total, precision_total/96