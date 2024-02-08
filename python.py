import numpy as np
class Perceptron:
    def __init__(self, learning_rate, epochs):
        self.weights = None
        self.bias = None
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation(self, z):
        return np.heaviside(z, 0) # haviside(z) heaviside -> activation

    def fit(self, X, y):
        n_features = X.shape[1]

        # Initializing weights and bias
        self.weights = np.zeros((n_features))
        self.bias = 0

        # Iterating until the number of epochs
        for epoch in range(self.epochs):

            # Traversing through the entire training set
            for i in range(len(X)):
                z = np.dot(X, self.weights) + self.bias  # Finding the dot product and adding the bias
                y_pred = self.activation(z)  # Passing through an activation function

                # Updating weights and bias
                self.weights = self.weights + self.learning_rate * (y[i] - y_pred[i]) * X[i]
                self.bias = self.bias + self.learning_rate * (y[i] - y_pred[i])

        return self.weights, self.bias

    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self.activation(z)

from sklearn.datasets import load_iris

iris = load_iris() 
from sklearn.model_selection import train_test_split

X = iris.data[:, (0, 1)] # petal length, petal width
y = (iris.target == 0).astype(np.dtype)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
perceptron = Perceptron(0.001, 100)

perceptron.fit(X_train, y_train)

pred = perceptron.predict(X_test)
from sklearn.metrics import accuracy_score

accuracy_score(pred, y_test)
from sklearn.metrics import classification_report

report = classification_report(pred, y_test, digits=2)
print(report)
from sklearn.linear_model import Perceptron


sk_perceptron = Perceptron()
sk_perceptron.fit(X_train, y_train)
sk_perceptron_pred = sk_perceptron.predict(X_test)

# Accuracypython
print(accuracy_score(sk_perceptron_pred, y_test))