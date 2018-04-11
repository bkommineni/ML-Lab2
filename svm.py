from sklearn import svm
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data[:,:2]
Y = iris.target

svc = svm.SVC(kernel='linear')
svc.fit(X, Y)
svc.predict(X)