from KDE import KDE
from ImportData import getData, separateData, importThetas
import numpy as np
import matplotlib.pyplot as plt

def main_getThetas():
    kde = KDE()
    thetas = np.zeros((240, 1))
    index = 0

    for i in range(1, 17):
        for j in range(1, 17):
            if i != j:
                TX1RX2, labels = getData('TX1RX2')
                X_healthy, X_healthy_index, patient_out = separateData(2, TX1RX2, labels)

                theta = kde.getTheta(TX1RX2, labels)
                thetas[index] = theta
                index += 1
                print(theta)

    print(kde.classified(14.67, TX1RX2, labels))
    TXRX, labels = getData('TX7RX8')
    X_healthy, X_healthy_index, patient_out = separateData(2, TXRX, labels)
    print(kde.getTheta(TXRX, labels))
    print(kde.trainAllThetas())


def eval():
    thetas = importThetas()

    average_out = np.zeros((132, 1))
    average_total = np.zeros((132, 1))
    index = 0

    for i in range(1, 13):
        for j in range(1, 13):
            if i < j:
                patient_out = np.array([i, j])
                kde = KDE()
                classification, decision, labels = kde.classify240(patient_out, thetas)
                out, total = kde.getPrecision(decision, labels, patient_out)
                print(out, total)

                average_out[index, 0] = out
                average_total[index, 0] = total
                index += 1

    print(np.mean(average_out[0:index, 0]))
    print(np.mean(average_total[0:index, 0]))

    plt.plot(average_out[0:index, 0])
    plt.show()

    plt.plot(average_total[0:index,0])
    plt.show()


eval()
