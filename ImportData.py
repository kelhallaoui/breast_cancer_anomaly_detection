import numpy as np
from numpy import genfromtxt

def importThetas():
    thetas = genfromtxt('Data\\Other\\thetas.csv', delimiter=',')
    return thetas

def getData(TXRX):
    TXRX = genfromtxt('DataAll\\' + TXRX +'.csv', delimiter=',')
    labels = genfromtxt('DataAll\\Labels.csv', delimiter=',')
    return TXRX, labels

def separateData(leave_out, TX1RX2, labels):
    if(leave_out!=0):

        cond = True
        while cond:
            patient_out = np.random.randint(1, 13, leave_out)
            if np.unique(patient_out).shape[0] == leave_out:
                cond = False

        X_healthy = np.zeros((96,20))
        X_healthy_index = np.zeros((96,1))
        index = 0

        for i in range(0, 96):
            if(labels[i,0]==-1 and not(labels[i,1] in patient_out) ):
                X_healthy[index,:] = TX1RX2[i,:]
                X_healthy_index[index,0] = i
                index = index + 1

        X_healthy = X_healthy[0:index, :]
        X_healthy_index = X_healthy_index[0:index, :]

        return X_healthy, X_healthy_index, patient_out

def separateDataOut(TXRX, labels, patient_out):
    X_healthy = np.zeros((96,20))
    X_healthy_index = np.zeros((96,1))
    index = 0

    for i in range(0, 96):
        if(labels[i,0]==-1 and not(labels[i,1] in patient_out) ):
            X_healthy[index,:] = TXRX[i,:]
            X_healthy_index[index,0] = i
            index = index + 1

    X_healthy = X_healthy[0:index, :]
    X_healthy_index = X_healthy_index[0:index, :]

    return X_healthy, X_healthy_index, patient_out

def getPatientOut(leave_out):
    cond = True
    while cond:
        patient_out = np.random.randint(1, 13, leave_out)
        if np.unique(patient_out).shape[0] == leave_out:
            cond = False
    return patient_out