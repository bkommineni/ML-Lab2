from svmMLiA_revised import *

X,Y = loadDataSet('linearly_separable.csv')

b, alphas = smoPK(X,Y,1.0,0.001,50)

print(b)
print(alphas)
