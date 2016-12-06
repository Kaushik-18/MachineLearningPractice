import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style


class SupportVectorMachine:
    def __init__(self, visuals=True):
        self.visuals = visuals
        self.colors = {1: 'r', -1: 'b'}
        if visuals:
            self.figures = plt.figure()
            self.axis = self.figures.add_subplot(1, 1, 1)

    def fit(self, data):
        self.data = data
        opt_dict = {}
        transforms = [[1, 1], [-1, 1], [-1, -1], [1, -1]]

        all_data = []
        for y in self.data:
            for feature_set in self.data[y]:
                for feature in feature_set:
                    all_data.append(feature)

        self.max_feature = max(all_data)
        self.min_fature = min(all_data)
        all_data = None

        step_sizes = [self.max_feature * 0.1,
                      self.max_feature * 0.01,
                      self.max_feature * 0.001]

        b_multiple = 5
        latest_optimum = self.max_feature * 10
        b_range_multiple = 2

        for step in step_sizes:
            w = np.array([latest_optimum, latest_optimum])
            optimzed = False
            while not optimzed:
                for b in np.arange(-1 * (self.max_feature * b_range_multiple), self.max_feature * b_range_multiple,
                                   b_multiple * step):
                    for transformation in transforms:
                        w_t = w * transformation
                        found_options = True
                        for value in self.data:
                            for x in self.data[value]:
                                y = value
                                if not y * ((np.dot(w_t, x)) + b) >= 1:
                                    found_options = False

                        if found_options:
                            opt_dict[np.linalg.norm(w_t)] = [w_t, b]

                if (w[0]) < 0:
                    optimzed = True
                    print("optimized one step ", w[0])
                else:
                    w = w - step

            norms = sorted([n for n in opt_dict])
            # |w| : [w,b]
            opt_choice = opt_dict[norms[0]]
            self.w = opt_choice[0]
            self.b = opt_choice[1]
            latest_optimum = opt_choice[0][0] + step * 2

    def predict(self, features):
        classify = np.sign(np.dot(np.array(features), self.w) + self.b)
        if classify != 0 and self.visuals:
            self.axis.scatter(features[0], features[1], s=200, marker="*", color=[classify])
        return classify

    def visualize(self):
        [[self.axis.scatter(x[0], x[1], s=100, color=self.colors[i]) for x in values_dict[i]] for i in values_dict]

        def hyperplane(x, w, b, v):
            return (-w[0] * x - b + v) / w[1]

        datarange = (self.min_fature * 0.9, self.max_feature * 1.1)
        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]
        print(hyp_x_max,hyp_x_min)
        pv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
        pv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
        self.axis.plot([hyp_x_min, hyp_x_max], [pv1, pv2])
        nv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.axis.plot([hyp_x_min, hyp_x_max], [nv1, nv2])
        db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.axis.plot([hyp_x_min, hyp_x_max], [db1, db2])
        plt.show()


values_dict = {-1: np.array([[1, 7],
                             [2, 8],
                             [3, 8], ]),

               1: np.array([[5, 1],
                            [6, -1],
                            [7, 3], ])}

svm = SupportVectorMachine()
svm.fit(data=values_dict)
svm.visualize()
