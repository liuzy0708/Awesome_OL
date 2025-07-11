""" QRBLS classifier."""

import numpy as np
from sklearn import preprocessing, logger
from numpy import random

class scaler:
    def __init__(self):
        self._mean = 0
        self._std = 0
        self._reg = 0

    def fit_transform(self, traindata):
        self._mean = traindata.mean(axis=0)
        self._std = traindata.std(axis=0)
        return (traindata - self._mean) / (self._std + 1e-6)

    def transform(self, testdata):
        return (testdata - self._mean) / (self._std + 1e-6)
    
class QRDecomposition:
    def __init__(self):
        self.Q = 0
        self.R = 0
        self.W = 0
        self.action = ""
        self._reg = 0
        self.count = 0
        self.L = 0
        self._Q = 0
        self.K = 0
        self._P = 0
        self.Rinverse = 0
        self.rightRinverse = 0
    
    def qr_decomposition(self, A, mode="full"):
        m, n = A.shape
        Q = np.eye(m)
        R = np.array(A, copy=True, dtype=float)
        
        for j in range(min(m, n)):
            u = R[j:, j].copy()
            norm_u = np.linalg.norm(u)
            if norm_u > np.finfo(float).eps:
                u[0] += np.copysign(norm_u, u[0])
                u /= np.linalg.norm(u)
                R[j:, j:] -= 2 * np.outer(u, np.dot(u, R[j:, j:]))
                Q[:, j:] -= 2 * np.outer(Q[:, j:].dot(u), u)
        
        if mode == 'economic': 
            Q = Q[:, :n]
            R = R[:n, :]
        
        return Q, R
    
    def weightsCalculated(self, A, B, reg, mode, action = "addNewData"):
        self._reg = reg
        self.action = action
        self.Q,self.R = self.qr_decomposition(A, mode=mode)
        if action == "addNewData":
            r = self.R.shape[1]
            self.K = self.R.T @ self.R
            self._P = np.linalg.inv(self.K + self._reg * np.eye(r)) 

        self.P = self.pinv(self.R)
        
        self.W = self.P.dot(self.Q.T.dot(B))
        
        return self.W

    def pinv(self, A):
        return np.mat(self._reg*np.eye(A.shape[1])+A.T.dot(A)).I.dot(A.T)
    
        
    def UpdateQR(self, V, B, action = "addNewData"):
        if action == "addNewData":
            q, r = self.qr_decomposition(V)
            h = r
            self._P = self._P - self._P @ h.T @ np.linalg.inv(np.eye(h.shape[0]) + h @ self._P @ h.T) @ h @ self._P
            self.W = self.W + self._P @ V.T @ (B - V @ self.W)

        else:
            h = self.Q.T @ V
            C = V - self.Q @ h

            if np.all(C == 0):
                R_inverse = self.R @ self.P
                F = V.T @ V - h.T @ R_inverse @ h
                N = self.P @ h
                M = np.linalg.inv(F) @ (h.T) @ (np.eye(self.R.shape[0]) - R_inverse)
                self.Q = self.Q
                self.R = np.hstack([self.R, h])
                self.P = self.pinv(self.R)
                self.W = np.vstack([(self.W - N @ M @ B), M @ B])
            else:
                q, r = self.qr_decomposition(C, mode="economic")
                M = self.pinv(r) @ q.T
                N = self.P @ h
                self.Q = np.hstack([self.Q, q])
                R_new_up = np.hstack([self.R, h])
                R_new_down = np.hstack([np.zeros([r.shape[0], self.R.shape[1]]), r])
                self.R = np.vstack([R_new_up, R_new_down])
                self.P = self.pinv(self.R)
                self.W = np.vstack([(self.W - N @ M @ B), M @ B])

        return np.array(self.W)
    
class node_generator:
    def __init__(self, whiten=False):
        self.Wlist = []
        self.blist = []
        self.nonlinear = 0
        self.whiten = whiten
        self.seed = 0

    def sigmoid(self, data):
        return 1.0 / (1 + np.exp(-data))

    def linear(self, data):
        return data

    def tanh(self, data):
        return (np.exp(data) - np.exp(-data)) / (np.exp(data) + np.exp(-data))

    def relu(self, data):
        return np.maximum(data, 0)
    
    def tansig(self,data):
        return (2/(1+np.exp(-2*data)))-1

    def orth(self, W):

        for i in range(0, W.shape[1]):
            w = np.mat(W[:, i].copy()).T
            w_sum = 0
            for j in range(i):
                wj = np.mat(W[:, j].copy()).T

                w_sum += (w.T.dot(wj))[0, 0] * wj
            w -= w_sum
            w = w / np.sqrt(w.T.dot(w))
            W[:, i] = np.ravel(w)

        return W

    def generator(self, shape, times):
        for i in range(times):
            # random.seed(i+self.seed)
            W = 2 * random.random(size=shape) - 1
            if self.whiten == True:
                W = self.orth(W)
            b = 2 * random.random() - 1
            yield (W, b)

    def generator_nodes(self, data, times, batchsize, nonlinear):
        self.Wlist = [elem[0] for elem in self.generator((data.shape[1], batchsize), times)]
        self.blist = [elem[1] for elem in self.generator((data.shape[1], batchsize), times)]

        self.nonlinear = {'linear': self.linear,
                          'sigmoid': self.sigmoid,
                          'tanh': self.tanh,
                          'relu': self.relu,
                          'tansig': self.tansig
                          }[nonlinear]
        nodes = self.nonlinear(data.dot(self.Wlist[0]) + self.blist[0])
        for i in range(1, len(self.Wlist)):
            nodes = np.column_stack((nodes, self.nonlinear(data.dot(self.Wlist[i]) + self.blist[i])))
        return nodes

    def transform(self, testdata):
        testnodes = self.nonlinear(testdata.dot(self.Wlist[0]) + self.blist[0])
        for i in range(1, len(self.Wlist)):
            testnodes = np.column_stack((testnodes, self.nonlinear(testdata.dot(self.Wlist[i]) + self.blist[i])))
        return testnodes

    def update(self, otherW, otherb):
        self.Wlist += otherW
        self.blist += otherb

class QRBLS:
    def __init__(self,
                 Nf=10,
                 Ne=10,
                 N1=10,
                 N2=10,
                 M1=1,
                 M2=5,
                 E1=1,
                 E2=100,
                 E3=1,
                 map_function='tansig',
                 enhence_function='tansig',
                 reg=0.001,
                 n_class=0):

        logger.info("These are the initial parameters of the QRBLS model.")
        self._Nf = Nf
        logger.info(f"Nf: {Nf}")
        self._Ne = Ne
        logger.info(f"Ne: {Ne}")
        self._map_function = map_function
        logger.info(f"map_function: {map_function}")
        self._enhence_function = enhence_function
        logger.info(f"enhence_function: {enhence_function}")
        self._reg = reg
        logger.info(f"reg: {reg}")
        self._N1 = N1
        logger.info(f"N1: {N1}")
        self._N2 = N2
        logger.info(f"N2: {N2}")
        self._M1 = M1
        logger.info(f"M1: {M1}")
        self._M2 = M2
        logger.info(f"M2: {M2}")
        self._E1 = E1
        logger.info(f"E1: {E1}")
        self._E2 = E2
        logger.info(f"E2: {E2}")
        self._E3 = E3
        logger.info(f"E3: {E3}")
        self._n_class = n_class



        self._mode = []
        self.tempLastFeatureLayer = 0
        self.normalscaler = scaler()
        self.onehotencoder = preprocessing.OneHotEncoder(sparse=False)
        self.mapping_generator = node_generator()
        self.enhence_generator = node_generator(whiten=True)
        self.mapping_generatorforaddtionalnodes = node_generator()
        self.enhence_generatorforaddtionalnodesOne = node_generator(whiten=True)
        self.enhence_generatorforaddtionalnodesTwo = node_generator(whiten=True)
        self.qrdecomposition = QRDecomposition() 
        self.local_mapgeneratorlist = []
        self.local_enhgeneratorlist = []
        self.tempTestAddtionalNodes = []
        self.tempTestMappingNodes = []
        self.tempgeneratordic = {}
        self.count = 0

    def fit(self, oridata, orilabel, action = "addNewData"):
        # time_start=time.time()#计时开始

        data = self.normalscaler.fit_transform(oridata)

        if self._n_class == 0:
            label = self.onehotencoder.fit_transform(orilabel.reshape(-1, 1))
        else:
            label = np.eye(self._n_class)[orilabel]
        
        mappingdata = self.mapping_generator.generator_nodes(data, self._Nf, self._N1, self._map_function)

        enhencedata = self.enhence_generator.generator_nodes(mappingdata, self._Ne, self._N2, self._enhence_function)

        inputdata = np.column_stack((mappingdata, enhencedata))
        
        self.W = self.qrdecomposition.weightsCalculated(inputdata, label, self._reg, mode = "economic", action = action)
        
        self.tempLastFeatureLayer = mappingdata

        return self.W
        

    def softmax_norm(self, array):
        exp_array = np.matrix(np.exp(array))
        sum_exp_array = np.sum(exp_array, axis=1)
        softmax_array = exp_array / sum_exp_array
        return softmax_array

    def decode(self, Y_onehot):
        Y = []
        for i in range(Y_onehot.shape[0]):
            lis = np.ravel(Y_onehot[i, :]).tolist()
            Y.append(lis.index(max(lis)))
        return np.array(Y)

    def predict(self, testdata):
        logit = self.predict_proba(testdata)
        return self.decode(self.softmax_norm(logit))


    def predict_proba(self, testdata):
        testdata = self.normalscaler.transform(testdata)
        test_inputdata = self.transform(testdata)
        org_prediction = test_inputdata.dot(self.W)
        return self.softmax_norm(org_prediction)

    def transform(self, data):
        mappingdata = self.mapping_generator.transform(data)
        enhencedata = self.enhence_generator.transform(mappingdata)
        inputdata = np.column_stack((mappingdata, enhencedata))
        if len(self.tempgeneratordic) > 0 :
            tempmappingnodes = mappingdata
            addtionalmappingdata = np.empty([data.shape[0],0])
            addtionalenhancenodes1 = np.empty([data.shape[0],0])
            addtionalenhancenodes2 = np.empty([data.shape[0],0])
            for i in range(self.count):
                for key,value in self.tempgeneratordic.items():
                    # print(key,self.count)
                    if key == "feature_%d" % i:
                        addtionalmappingdata = value.transform(data)
                        tempmappingnodes = np.hstack([tempmappingnodes,addtionalmappingdata])
                    if key == "enhence1_%d" % i:
                        addtionalenhancenodes1 = value.transform(tempmappingnodes)
                    if key == "enhence2_%d" % i:
                        addtionalenhancenodes2 = value.transform(addtionalmappingdata)
                inputdata = np.hstack([inputdata,addtionalmappingdata,addtionalenhancenodes1,addtionalenhancenodes2])
        return inputdata
    

    def partial_fit(self, data, label, mode = "addNewData"):
        self._mode = mode
        xdata = self.normalscaler.transform(data)
        if self._n_class == 0:
            xlabel = self.onehotencoder.transform((label).reshape(-1, 1))
        else:
            xlabel = np.zeros([label.shape[0],self._n_class])
            for i in range(label.shape[0]):
                xlabel[i,int(label[i])] = 1  

        if mode == "addNewData":
            xdata = self.transform(xdata)
            self.W = self.qrdecomposition.UpdateQR(xdata,xlabel)
        else:
            if mode == "addNewFeatureNodes":
                self.tempgeneratordic["feature_%d" % self.count] = node_generator()
                self.tempgeneratordic["enhence1_%d" % self.count] = node_generator(whiten=True)
                self.tempgeneratordic["enhence2_%d" % self.count] = node_generator(whiten=True)

                addtionalmappingdata = self.tempgeneratordic["feature_%d" % self.count].generator_nodes(xdata, self._M1, self._M2, self._map_function)
                self.tempLastFeatureLayer = np.hstack([self.tempLastFeatureLayer,addtionalmappingdata])
                addtionalenhencedataOne = self.tempgeneratordic["enhence1_%d" % self.count].generator_nodes(self.tempLastFeatureLayer, self._E1, self._E2, self._enhence_function)
                addtionalenhencedataTwo = self.tempgeneratordic["enhence2_%d" % self.count].generator_nodes(addtionalmappingdata, self._E1, self._E3, self._enhence_function)
                TheAddedNodes = np.hstack([addtionalmappingdata,addtionalenhencedataOne,addtionalenhencedataTwo])

                self.count+=1
            else:
                self.tempgeneratordic["enhence1_%d" % self.count] = node_generator(whiten=True)
                addtionalenhencedata = self.tempgeneratordic["enhence1_%d" % self.count].generator_nodes(self.tempLastFeatureLayer, self._E1, self._E2, self._enhence_function)
                TheAddedNodes = addtionalenhencedata
                self.count+=1
            self.W, direct_matrix, iter_matrix = self.qrdecomposition.UpdateQR(TheAddedNodes,xlabel,action=mode)
            return direct_matrix, iter_matrix
    














