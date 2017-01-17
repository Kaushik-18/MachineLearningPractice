import matplotlib.pyplot as plot
from matplotlib import style
import numpy as np

X = np.array([[1, 2], [5, 8], [8, 8], [9, 11], [1, 0.6], [1.5, 1.8]])

colors = 10 * ["g", "r", "c", "b", "k"]


class MeanShift:
    def __init__(self, radius=None, norm_radius_steps=100):
        self.radius = radius
        self.norm_radius_step = norm_radius_steps

    def fit(self, data):

        if self.radius is None:
            all_data_centroid = np.average(data, axis=0)
            all_data_norm = np.linalg.norm(all_data_centroid)
            self.radius = all_data_norm / self.norm_radius_step

        centroids = {}
        for i in range(len(data)):
            centroids[i] = data[i]

        weights = [i for i in range(self.norm_radius_step)][::-1]

        while True:
            new_centroids = []
            for i in centroids:
                in_bandwidth = []
                centroid = centroids[i]
                for fetureset in data:
                    distance = np.linalg.norm(fetureset - centroid)
                    if distance == 0:
                        distance = 0.0000001
                    weight_index = int(distance / self.radius)
                    if weight_index > self.norm_radius_step - 1:
                        weight_index = self.norm_radius_step - 1
                    to_add = (weights[weight_index] ** 2) * [fetureset]
                    in_bandwidth += to_add

                # if np.linalg.norm(fetureset - centroid) < self.radius:
                #    in_bandwidth.append(fetureset)

                new_centre = np.average(in_bandwidth, axis=0)
                new_centroids.append(tuple(new_centre))

            uniques = sorted(list(set(new_centroids)))

            to_pop = []
            for i in uniques:
                for ii in uniques:
                    if i == ii:
                        pass
                    elif (np.linalg.norm(np.array(i) - np.array(ii))) <= self.radius:
                        to_pop.append(ii)
                        break

            for i in to_pop:
                try:
                    uniques.remove(i)
                except:
                    print("error in uniques")

            prev_centroids = dict(centroids)

            centroids = {}
            for i in range(len(uniques)):
                centroids[i] = np.array(uniques[i])

            optimized = True
            for i in centroids:
                if not np.array_equal(centroids[i], prev_centroids[i]):
                    optimized = False
                if not optimized:
                    break

            if optimized:
                break

        self.centroids = centroids
        self.classifications = {}
        for i in range(len(self.centroids)):
            self.classifications[i] = []

        for fetureset in data:
            distance = [np.linalg.norm(fetureset - self.centroids[centroid]) for centroid in self.centroids]
            classification = distance.index(min(distance))
            self.classifications[classification].append(fetureset)

    # def predict(self, data):
    #     distance = [np.linalg.norm(fetureset - self.centroids[centroid]) for centroid in self.centroids]
    #     classification = distance.index(min(distance))
    #     return classification


plot.scatter(X[:, 0], X[:, 1], s=150)

clf = MeanShift()
clf.fit(X)
plot_centroid = clf.centroids
for c in plot_centroid:
    plot.scatter(plot_centroid[c][0], plot_centroid[c][1], color='k', marker='*', s=150)

plot.show()
