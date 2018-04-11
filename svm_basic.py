from classifier import classifier
from numpy import *
from time import sleep


class svm_basic(classifier):
    def __init__(self, C=1.0, tol=0.001, maxIter=50):
        self.C = C
        self.tol = tol
        self.maxIter = maxIter

    def selectJrand(self,i):
        j = i  # we want to select any J not equal to i
        while (j == i):
            j = int(random.uniform(0, self.m))
        return j

    def clipAlpha(self, aj, H, L):
        if aj > H:
            aj = H
        if L > aj:
            aj = L
        return aj

    def calcEkK(self, k):
        p = multiply(self.alphas, self.labelMat).T
        q = (self.X * self.X[k, :].T)
        r = float(p * q)
        fXk = r + self.b
        Ek = fXk - float(self.labelMat[k])
        return Ek

    def selectJK(self,i,Ei):  # this is the second choice -heurstic, and calcs Ej
        maxK = -1
        maxDeltaE = 0
        Ej = 0
        self.eCache[i] = [1, Ei]  # set valid #choose the alpha that gives the maximum delta E
        validEcacheList = nonzero(self.eCache[:, 0].A)[0]
        if (len(validEcacheList)) > 1:
            for k in validEcacheList:  # loop through valid Ecache values and find the one that maximizes delta E
                if k == i: continue  # don't calc for i, waste of time
                Ek = self.calcEkK(k)
                deltaE = abs(Ei - Ek)
                if (deltaE > maxDeltaE):
                    maxK = k
                    maxDeltaE = deltaE
                    Ej = Ek
            return maxK, Ej
        else:  # in this case (first time around) we don't have any valid eCache values
            j = self.selectJrand(i)
            Ej = self.calcEkK(j)
        return j, Ej

    def updateEkK(self, k):  # after any alpha has changed update the new value in the cache
        Ek = self.calcEkK(k)
        self.eCache[k] = [1, Ek]

    def innerLK(self,i):
        Ei = self.calcEkK(i)
        if ((self.labelMat[i] * Ei < -self.tol) and (self.alphas[i] < self.C)) or (
                    (self.labelMat[i] * Ei > self.tol) and (self.alphas[i] > 0)):
            j, Ej = self.selectJK(i,Ei)  # this has been changed from selectJrand
            alphaIold = self.alphas[i].copy()
            alphaJold = self.alphas[j].copy()
            if (self.labelMat[i] != self.labelMat[j]):
                L = max(0, self.alphas[j] - self.alphas[i])
                H = min(self.C, self.C + self.alphas[j] - self.alphas[i])
            else:
                L = max(0, self.alphas[j] + self.alphas[i] - self.C)
                H = min(self.C, self.alphas[j] + self.alphas[i])
            if L == H:
                #print("L==H")
                return 0
            eta = 2.0 * self.X[i, :] * self.X[j, :].T - self.X[i, :] * self.X[i, :].T - self.X[j, :] * self.X[j, :].T
            if eta >= 0:
                #print("eta>=0")
                return 0
            self.alphas[j] -= self.labelMat[j] * (Ei - Ej) / eta
            self.alphas[j] = self.clipAlpha(self.alphas[j], H, L)
            self.updateEkK(j)  # added this for the Ecache
            if abs(self.alphas[j] - alphaJold) < 0.00001:
                #print("j not moving enough")
                return 0
            self.alphas[i] += self.labelMat[j] * self.labelMat[i] * (
            alphaJold - self.alphas[j])  # update i by the same amount as j
            self.updateEkK(i)  # added this for the Ecache                    #the update is in the oppostie direction
            b1 = self.b - Ei - self.labelMat[i] * (self.alphas[i] - alphaIold) * self.X[i, :] * self.X[i, :].T - \
                 self.labelMat[j] * ( self.alphas[j] - alphaJold) * self.X[i,:] * self.X[j,:].T
            b2 = self.b - Ej - self.labelMat[i] * (self.alphas[i] - alphaIold) * self.X[i, :] * self.X[j, :].T - \
                 self.labelMat[j] * (self.alphas[j] - alphaJold) * self.X[j,:] * self.X[j,:].T
            if (0 < self.alphas[i]) and (self.C > self.alphas[i]):
                self.b = b1
            elif (0 < self.alphas[j]) and (self.C > self.alphas[j]):
                self.b = b2
            else:
                self.b = (b1 + b2) / 2.0
            return 1
        else:
            return 0

    def calcWs(self):
        X = mat(self.X)
        labelMat = mat(self.labelMat)
        m, n = shape(X)
        w = zeros((n, 1))
        for i in range(m):
            w += multiply(self.alphas[i] * labelMat[i], X[i, :].T)
        return w

    def fit(self, Xin, Yin):
        self.X = mat(Xin)
        self.labelMat = mat(Yin).transpose()
        self.m = shape(Xin)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        self.eCache = mat(zeros((self.m, 2)))

        iter = 0
        entireSet = True
        alphaPairsChanged = 0
        while (iter < self.maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
            alphaPairsChanged = 0
            if entireSet:  # go over all
                for i in range(self.m):
                    alphaPairsChanged += self.innerLK(i)
                    #print("fullSet, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
                iter += 1
            else:  # go over non-bound (railed) alphas
                nonBoundIs = nonzero((self.alphas.A > 0) * (self.alphas.A < self.C))[0]
                for i in nonBoundIs:
                    alphaPairsChanged += self.innerLK(i)
                    #print("non-bound, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
                iter += 1
            if entireSet:
                entireSet = False  # toggle entire set loop
            elif (alphaPairsChanged == 0):
                entireSet = True
            #print("iteration number: %d" % iter)

        Ws = self.calcWs()

        return self.b, Ws

    def predict(self, X):
        hypotheses = []
        Ws = self.calcWs()
        w = asmatrix(Ws)
        w_t = w.transpose()
        for i in range(0,len(X)):
            hypotheses.append(w_t * mat(X[i]).transpose()+ self.b)
        return hypotheses





