import numpy as np

class MyKNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """Store the training vectors and labels."""
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def euclidean_distance(self, a, b):
        """Calculate Euclidean Distance between two feature vectors."""
        return np.sqrt(np.sum((a - b) ** 2))

    def predict(self, x_test):
        """Predict identity of a face vector using KNN."""
        distances = []

        # Step 1: compute distance to all training vectors
        for i, x_train in enumerate(self.X_train):
            dist = self.euclidean_distance(x_train, x_test)
            distances.append((dist, self.y_train[i]))

        # Step 2: sort by distance
        distances.sort(key=lambda x: x[0])

        # Step 3: choose k nearest
        k_nearest = distances[:self.k]

        # Step 4: count labels
        labels = [label for (_, label) in k_nearest]
        prediction = max(set(labels), key=labels.count)

        return prediction
