import numpy as np

class Variable():
    inputs = []
    num_of_variables = 0
    def __init__(self, name=None, evaluate=None, grad = None) :
        if evaluate == None:
            self.evaluate = lambda values: values[self.name]
        else:
            self.evaluate = evaluate
        if grad == None:
            Variable.num_of_variables += 1
            self.current = Variable.num_of_variables
            output = [0] * Variable.num_of_variables
            output[self.current-1] = 1
            self.grad = lambda values: np.array(output+[0]*(Variable.num_of_variables - self.current))
        else:
            self.grad = grad
            
        if name != None:
            self.name = name          # its key in the evaluation dictionary

        self.inputs.append(name)
    
    def __add__(self, other):
        if isinstance(other, (int, float)):
            return Variable(evaluate = lambda values: self.evaluate(values) + other, grad = self.grad)
            
        return Variable(evaluate = lambda values: self.evaluate(values) + other.evaluate(values), grad = lambda values: self.grad(values) 
                                    + other.grad(values))
    
    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Variable(evaluate = lambda values: self.evaluate(values) * other, grad = lambda values: self.grad(values) * other)
        
        return Variable(evaluate = lambda values: self.evaluate(values) * other.evaluate(values), grad = lambda values: self.evaluate(values) * other.grad(values) + other.evaluate(values) * self.grad(values))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __sub__(self, other):
        return self.__add__(other.__mul__(-1))

    def __rsub__(self, other):
        return (self.__mul__(-1)).__add__(other)

    def __pow__(self, other):
        if isinstance(other, (int, float)):
            return Variable(evaluate = lambda values: self.evaluate(values) ** other, grad = lambda values: other * self.evaluate(values)**(other-1) * self.grad(values))
        
        return Variable(evaluate = lambda values: self.evaluate(values) ** other.evaluate(values))

    def __truediv__(self, other):
        return self.__mul__(other.__pow__(-1))

    def __rtruediv__(self, other):
        return (self.__pow__(-1)).__mul__(other)

    @classmethod
    def exp(cls, other):
        return Variable(
            evaluate = lambda values: np.exp(other.evaluate(values)),
            grad = lambda values: np.exp(other.evaluate(values)) * other.grad(values)
        )
    
    @classmethod
    def log(cls, other):
        return Variable(
            evaluate = lambda values: np.log(other.evaluate(values)),
            grad = lambda values: 1/other.evaluate(values) * other.grad(values)
        )
    


        
        
# x_1 = Variable(name="x_1")
# x_2 = Variable(name="x_2")
# x_3 = Variable(name ="x_3")

# z = Variable.exp(x_1 + x_2**2) + 3 * Variable.log(27 - x_1 * x_2 * x_3)
# print(z.grad({'x_1': 3, 'x_2': 1, 'x_3': 7}))



