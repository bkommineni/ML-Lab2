from svm_basic import svm_basic
import numpy as np
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

hypotheses = clf.predict(np.mat(X))

labels = []
x = np.arange(min([i[0] for i in X]), max([i[0] for i in X]) , 0.5)
list_x = x.tolist()
list_x = np.append(list_x, [0, -float(b)/float(weights[1])])
x = np.array(list_x)
y = []
for i in range(0,len(x)):
    y.append((-float(b) - (float(weights[1]) * list_x[i])) / float(weights[0]))

for i in range(0,len(hypotheses)):
    if hypotheses[i] >= 0 :
        labels.append(1)
    else:
        labels.append(-1)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter([i[0] for i in X], [i[1] for i in X], c=labels, alpha=0.5)
ax.plot(x, np.array(y))
plt.savefig("UsingPredictions.png")
