import numpy as np
import matplotlib.pyplot as plt
X = np.linspace(0, 1,10)
y = X + np.random.normal(0, 0.1, (10,))

X = X.reshape(-1,1)
y = y.reshape(-1,1)

class LinearRegression:
    #def __init__(self):
        #self.lr = lr
        #self.num_epochs = num_epochs
        #self.weights = None
    def initialisation(self,n_features):
        return np.zeros((n_features, 1))

    def make_prediction(self, X, weights):
        return X@weights

    def mse(self, y, y_pred):
        return np.divide((np.sum(y_pred-y))**2,y.shape[0])

    def grad_mse(self, X, y, weights):
        n_samples = X.shape[0]
        #y_pred = self.make_prediction(X, weights)
        #grad = -(2 / n_samples) * np.dot(X.T, (y_pred-y))
        return np.divide(2*X.T@(self.make_prediction(X, weights)-y),n_samples)

    def grad_update(self, weights,grad,lr):

        return weights - lr * grad
    def plot(self,losses):
        plt.plot(losses)
        plt.xlabel("Epoch")
        plt.ylabel("Losses")
        plt.title("Loss curve")
        plt.show()

    def fit(self, X, y):
        lr = 0.1
        num_epochs = 40
        n_features = X.shape[1]

        # Initialize weights
        weights = self.initialisation(n_features)

        # Training loop
        losses = []
        for _ in range(num_epochs):
            # Make predictions
            y_pred = self.make_prediction(X,weights)

            # Compute loss
            loss = self.mse(y, y_pred)
            # Compute gradient
            grad = self.grad_mse(X, y, weights)
            #print(loss)
            weights = self.grad_update(weights,grad,lr)
            losses.append(loss)
            print(f"Epoch {_} loss:{loss}")
        return self.plot(losses)#, print("hello")

   # def predict(self, X):
       # return self.make_prediction(X)
#l=LinearRegression()
#l.fit(X,y)
