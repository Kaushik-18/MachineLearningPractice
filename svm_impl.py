import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style


class SupportVectorMachine:
    def _init_(self, visuals=True):
        self.visuals = visuals
        self.colors = {1: 'r', -1: 'b'}
        if visuals:
            self.figures = plt.figure()
            self.axis = self.figures.add_subplot(1, 1, 1)

    def fit(self, data):
        pass

    def predict(self, features):
        classify = np.sign(np.dot(np.array(features), self.w) + self.b)

        return classify


values_dict = {-1: np.array([1, 7], [2, 8], [3, 8]),
               1: np.array([5, -1], [6, -1], [7, 3])}

