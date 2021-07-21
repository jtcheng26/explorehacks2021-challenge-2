import torch


class PolyRegressionModel:
    '''
    Polynomial regression using gradient descent
    '''

    def __init__(self, features, actual):
        '''
        Initialize model

        features: tensor w/ features
        actual: result tensor
        '''
        self.features = features
        self.actual = actual
        initial = self.randomize()
        self.weights = initial["weights"]
        self.bias = initial["bias"]
        self.features = initial["features"]

    def randomize(self):
        '''
        Initialize random weights and biases
        '''
        w = torch.rand(self.features.size()[1], requires_grad=True)
        b = torch.rand(1, requires_grad=True)
        f = self.features.t()

        return {
            "weights": w,
            "bias": b,
            "features": f,
        }

    @staticmethod
    def cost(actual, predicted):
        '''
        Return cost vector
        '''
        return torch.sum((actual - predicted) ** 2) / torch.numel(actual)

    def predict(self):
        '''
        Return prediction vector
        '''
        return (self.weights.double() @ self.features.double()) + self.bias.double()

    def learn(self, rate, iterations):
        '''
        Run gradient descent
        '''
        def iterate():
            m = self.cost(self.actual, self.predict())
            m.backward()

            with torch.no_grad():
                self.weights -= self.weights.grad * rate
                self.bias -= self.bias.grad * rate
                self.weights.grad.zero_()
                self.bias.grad.zero_()

        for _ in range(iterations):
            iterate()
