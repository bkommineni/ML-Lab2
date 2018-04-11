from svm_basic import svm_basic
from numpy import *
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split(',')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat

X, Y = loadDataSet('linearly_separable.csv')

plt.scatter([i[0] for i in X], [i[1] for i in X], c=Y, alpha=0.5)
plt.savefig("UsingLabels.png")

clf = svm_basic()
b, weights = clf.fit(X, Y)

hypotheses = clf.predict(mat(X))

labels = []

for i in range(0,len(hypotheses)):
    if hypotheses[i] >= 0 :
        labels.append(1)
    else:
        labels.append(-1)

plt.scatter([i[0] for i in X], [i[1] for i in X], c=labels, alpha=0.5)
plt.savefig("UsingPredictions.png")
