class LinearRegression:
    def __init__(self, lr = 10e-5, num_epochs=50):
        self.lr = lr
        self.num_epochs = num_epochs
        self.weights = None

    def make_prediction(self, X):
        return np.dot(X, self.weights)

    def mse(self, y, y_pred):
        return np.mean((y - y_pred) ** 2)

    def grad_mse(self, X, y, y_pred):
        n_samples = X.shape[0]
        y_pred = self.make_prediction(X)
        grad = -(2 / n_samples) * np.dot(X.T, (y - y_pred))
        return grad

    def grad_update(self, grad):
        self.weights -= self.lr * grad

    def fit(self, X, y):
        n_features = X.shape[1]

        # Initialize weights
        self.weights = np.zeros((n_features, 1))

        # Training loop
        losses = []
        for _ in range(self.num_epochs):
            # Make predictions
            y_pred = self.make_prediction(X)

            # Compute loss
            loss = self.mse(y, y_pred)
            losses.append(loss)

            # Compute gradient
            grad = self.grad_mse(X, y, self.weights)

            # Update weights
            self.grad_update(grad)

    def predict(self, X):
        return self.make_prediction(X)
