from variable import Variable 
from sklearn.metrics import accuracy_score
import math

class LogisticRegression():
    m = 1
    b = 2
    
    def __init__(self):
        self.m_var = Variable(name = "m")
        self.b_var = Variable(name = "b")

    def fit(self, X , y):
        gradient = [1, 1]
        while abs(gradient[1]) > 0.01 or abs(gradient[0]) > 0.01:
            cost = 0
            i = 0
            for element in y:
                y_hat = 1/(1 + Variable.exp((X[i] * self.m_var + self.b_var)* -1))
                cost += element * Variable.log(y_hat) + (1 - element) * Variable.log(1 - y_hat)
                i+=1
            
            cost = cost * -1
            gradient = cost.grad({'m': LogisticRegression.m, 'b': LogisticRegression.b})
            
            LogisticRegression.m -= gradient[0] * 0.01
            LogisticRegression.b -= gradient[1] * 0.01

    def predict(self, X):
        predictions = []
        for x in X:
            function = 1/(1 + math.exp((x * LogisticRegression.m + LogisticRegression.b)* -1))
            if function > 0.5:
                predictions.append(1)
            else:
                predictions.append(0)

        
        return predictions
            

model = LogisticRegression()
X_test = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
y_test = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
model.fit(X_test, y_test)
print(model.m)
print(model.b)
y_pred = model.predict(X_test)
print(y_pred)
print(accuracy_score(y_test, y_pred))
