import matplotlib.pyplot as plot
from matplotlib import style
import numpy as np

X = np.array([[1, 2], [5, 8], [8, 8], [9, 11], [1, 0.6], [1.5, 1.8]])

colors = 10 * ["g", "r", "c", "b", "k"]


class K_means:
    def __init__(self, tol=0.01, itr=300, k=2):
        self.k = k
        self.tol = tol
        self.itr = itr

    def predict(self, data):
        distances = [[np.linalg.norm(data - self.centroids[centroid])] for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification

    def fit(self, data):
        self.centroids = {}
        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.itr):
            self.classifications = {}
            for i in range(self.k):
                self.classifications[i] = []

            for featureset in data:
                distances = [[np.linalg.norm(featureset - self.centroids[centroid])] for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)

            prev_centroids = dict(self.centroids)
            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification],axis=0)

            optimized = True
            for c in self.centroids:
                orignal_centroids = prev_centroids[c]
                current_centroids = self.centroids[c]
                if np.sum((current_centroids - orignal_centroids) / orignal_centroids * 100.0) > self.tol:
                    optimized = False

                if optimized:
                    break


clf = K_means()
clf.fit(X)

for centroid in clf.centroids:
    plot.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1], marker="o", color="k", s=150, linewidths=5)

for classification in clf.classifications:
    color = colors[classification]
    for featureset in clf.classifications[classification]:
        plot.scatter(featureset[0], featureset[1], color=colors[classification], marker="x", s=150, linewidths=5)

plot.show()
