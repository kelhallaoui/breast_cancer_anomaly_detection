from ImportData import getData, separateDataOut
import numpy as np
from lof import LOF

def classify240(patient_out):
    classification = np.zeros((96,240))
    index = 0
    for i in range(1,17):
        for j in range(1,17):
            if(i!=j):
                TXRX, labels = getData('TX'+str(i)+'RX'+str(j))
                X_healthy, X_healthy_index, patient_out = separateDataOut(TXRX, labels, patient_out)
                classification[:,index] = np.resize(classified(X_healthy, TXRX, labels, patient_out), (96,))
                index = index + 1
                print(index)
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

def classified(X_healthy, TXRX, labels, patient_out):
    lof = LOF(X_healthy)

    classify = np.zeros((96,1))
    for i in range(0, 96):
        value = lof.local_outlier_factor(6, TXRX[i,:])
        if(value < 1):
            classify[i] = -1
        else:
            classify[i] = 1
    return classify

def getPrecision(classify, labels, patient_out):
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