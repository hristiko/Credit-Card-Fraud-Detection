import numpy as np
from config import PERCEPTRON_LR, PERCEPTRON_EPOCHS

class Perceptron:

    def __init__(self, input_dim, lr = PERCEPTRON_LR):
        self.lr = lr
        self.weights = np.zeros(input_dim)
        self.bias = 0.0

    def activation(self, x):
        return 1 if x >= 0 else 0
    
    def predict_one(self, x):
        linear_output = np.dot(x, self.weights) + self.bias
        return self.activation(linear_output)
    
    def train(self, X, y, epochs = PERCEPTRON_EPOCHS):
        n_samples = X.shape[0]

        for epoch in range(epochs):
            errors = 0

            for i in range (n_samples):
                x_i = X[i]
                y_true = y[i]

                y_predicted = self.predict_one(x_i)

                if y_predicted != y_true:
                    error = y_true - y_predicted
                    update = self.lr * error
                    self.weights += update * x_i
                    self.bias += update 
                    errors += 1

            print(f"[PERCEPTRON] Epoch {epoch + 1}/{epoch} - misclassifications: {errors}")
        