import numpy as np
import matplotlib.pyplot as plt
import time
import mlflow
import mlflow.pyfunc
import pandas as pd

class LogisticRegression:
    def __init__(self, k, n, lr=0.001, max_iter=1000, l2_penalty=False, lambda_=0.01, momentum=0.9):
        self.k = k          # number of classes
        self.n = n          # number of features
        self.lr = lr
        self.max_iter = max_iter
        self.l2_penalty = l2_penalty
        self.lambda_ = lambda_
        self.momentum = momentum

    def _xavier_init(self):
        limit = np.sqrt(6 / (self.n + self.k))
        W = np.random.uniform(-limit, limit, size=(self.n, self.k))
        b = np.zeros((1, self.k))
        return W, b

    def softmax(self, Z):
        Z = np.array(Z, dtype=float)
        Z = Z - np.max(Z, axis=1, keepdims=True)  # stability trick
        expZ = np.exp(Z)
        return expZ / np.sum(expZ, axis=1, keepdims=True)

    def _predict(self, X):
        return self.softmax(np.dot(X, self.W) + self.b)

    def predict(self, X_test):
        return np.argmax(self._predict(X_test), axis=1)

    def gradient(self, X, Y):
        X = np.array(X, dtype=float)
        Y = np.array(Y, dtype=float)
        m = X.shape[0]

        H = self._predict(X)
        loss = -np.sum(Y * np.log(H + 1e-9)) / m

        grad_W = np.dot(X.T, (H - Y)) / m
        grad_b = np.sum(H - Y, axis=0, keepdims=True) / m

        if self.l2_penalty:
            grad_W += (self.lambda_ / m) * self.W
            loss += (self.lambda_ / (2*m)) * np.sum(self.W**2)

        return loss, grad_W, grad_b

    def fit(self, X, Y):
        X = np.array(X, dtype=float)
        Y = np.array(Y, dtype=float)

        # Xavier initialization
        self.W, self.b = self._xavier_init()
        self.losses = []

        # Initialize velocities
        vW = np.zeros_like(self.W)
        vb = np.zeros_like(self.b)

        start_time = time.time()
        for i in range(self.max_iter):
            # Full batch gradient descent
            loss, grad_W, grad_b = self.gradient(X, Y)

            # Momentum update
            vW = self.momentum * vW - self.lr * grad_W
            vb = self.momentum * vb - self.lr * grad_b

            self.W += vW
            self.b += vb

            if i % 100 == 0:
                self.losses.append(loss)
                print(f"Loss at iteration {i}: {loss}")

        print(f"Time taken: {time.time() - start_time:.2f} seconds")

    def plot(self):
        plt.figure(figsize=(8,5))
        plt.plot(np.arange(len(self.losses))*100, self.losses, label="Train losses")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Training Loss over Iterations")
        plt.legend()
        plt.show()



class CarPricePredictor:
    def __init__(self, model, encoder, numeric_cols, cat_cols, mean, std):
        self.model = model
        self.encoder = encoder          # OrdinalEncoder fitted on train data
        self.numeric_cols = numeric_cols
        self.cat_cols = cat_cols
        self.mean = mean                # mean of polynomial features (train)
        self.std = std                  # std of polynomial features (train)

    def preprocess(self, X_raw):
        # Convert Series to DataFrame if needed
        if isinstance(X_raw, pd.Series):
            X = X_raw.to_frame().T
        else:
            X = X_raw.copy()

        # Check required columns
        for col in self.numeric_cols + self.cat_cols:
            if col not in X.columns:
                raise ValueError(f"Missing column: {col}")

        # Encode categorical columns
        X[self.cat_cols] = self.encoder.transform(X[self.cat_cols])

        # Reorder columns exactly as training
        X = X[self.numeric_cols + self.cat_cols]

        # Standardize features
        X_std = (X - self.mean) / self.std

        return X_std

    def predict(self, X_raw):
        X_processed = self.preprocess(X_raw)
        return np.argmax(self.model._predict(X_processed), axis=1)
    

class CarPriceWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, predictor):
        self.predictor = predictor

    def predict(self, context, model_input):
        # Make sure it works with DataFrames or Series
        return self.predictor.predict(model_input)