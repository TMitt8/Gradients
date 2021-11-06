from variable import Variable 
from sklearn.metrics import accuracy_score
import math
import numpy as np
import pandas as pd

class MulLogReg():
    variables = {}
    mbs = {}
    
    def __init__(self):
        self.b_var = Variable(name = "b")
        MulLogReg.variables[self.b_var.name] = self.b_var
        MulLogReg.mbs["b"] = 1

    def fit(self, X , y1):
    
        if isinstance(X, pd.DataFrame):
            Xcol = X.values.T.tolist()
        else:
            Xcol = X
        
        if type(y1) == pd.core.series.Series:
            y = y1.values.tolist()
        else:
            y = y1

        l = 1
        for i in range(len(Xcol)):
            MulLogReg.variables["m" + str(l)] = Variable(name = "m" + str(l))
            MulLogReg.mbs["m" + str(l)] = 0.5
            l+=1

        checkGradient = True
        gradient = []
        while checkGradient:
            cost = 0
            for j in range(len(Xcol[0])):
                yreg = 0
                m = 1
                for i in range(len(Xcol)):
                    yreg += MulLogReg.variables["m" + str(m)] * Xcol[m-1][j]
                    m+=1
                y_hat = 1/(1 + Variable.exp((yreg + self.b_var) * -1))
                cost += y[j] * Variable.log(y_hat) + (1 - y[j]) * Variable.log(1 - y_hat)
               
            
            cost = cost * -1
            gradient = cost.grad(MulLogReg.mbs)
            
            j = 0
            for key in MulLogReg.mbs:
                MulLogReg.mbs[key] = MulLogReg.mbs[key] - 0.01 * gradient[j]
                j+=1
            
            checkGradient = False
            for i in gradient:
                if abs(i) > 0.1:
                    checkGradient = True

    def predict(self, X):
        predictions = []
        yhat = 0
        for j in range(len(X[0])):
                yhat = 0
                m = 1
                for i in range(len(X)):
                    yhat += MulLogReg.mbs["m" + str(m)] * X[m-1][j]
                    m+=1
                function = 1/(1 + np.exp((yhat + MulLogReg.mbs["b"]) * -1))
                if function > 0.5:
                    predictions.append(1)
                else:
                    predictions.append(0)

        
        return predictions
            
model = MulLogReg()
X_test = [[6, 15, 14, 10, 17, 15, 7, 9, 14, 14, 5, 8, 8, 18, 4, 5, 4, 15, 15],[0.5, 1.5, 1.5, 0.5, 1.5, 1.5, 0.5, 0.5, 1.5, 1.5, 0.5, 0.5, 0.5, 1.5, 0.5, 0.5, 0.5, 1.5, 1.5], [15, 16, 20, 13, 14, 15, 11, 18, 19, 14, 15, 19, 12, 17, 17, 12, 11, 19, 17, 15]]
y_test = [1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0]
model.fit(X_test, y_test)
y_pred = model.predict(X_test)
print(y_pred)
print(accuracy_score(y_test, y_pred))
