import numpy as np

class Percpetron:

    def __init__(self, input_dim, lr =0.01):
        self.lr = lr
        self.weights = np.zeros(input_dim)
        self.bias = 0.0

    def activation(self, x):
        return 1 if x >= 0 else 0
    
    def predict_one(self, x):
        linear_output = np.dot(x, self.weights) + self.bias
        return self.activation(linear_output)
    
    def train(self, X, y, epochs = 5):
        n_samples = X.shape[0]

        for epoch in range(epochs):
            errors = 0

            for i in range (n_samples):
                x_i = X[i]
                y_true = y[i]

                y_pred = self.predict_one(x_i)

                if y_pred != y_true:
                    update = self.lr * (y_true - y_pred)
                    self.weights += update * x_i
                    self.bias += update 
                    errors += 1

            print(f"[PERCEPTRON] Epoch {epoch + 1}/{epoch} - misclassifications: {errors}")
        
    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])
    
    def predict_proba(self, X):
        z = X @ self.weights + self.bias
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))