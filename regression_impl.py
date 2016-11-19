from statistics import mean
import matplotlib.pyplot as plt
import numpy as np

x_data = np.array([1, 5, 8, 9, 10, 12])
y_data = np.array([3, 7, 9, 11, 14, 18])


def calc_best_slope(x, y):
    slope = (((mean(x) * mean(y)) - mean(x * y)) / ((mean(x) * mean(x)) - mean(x * x)))
    return slope

def calc_best_intercept(slope,y,x):
    line = mean(y) - slope*mean(x)
    return line

fit_slope = calc_best_slope(x_data,y_data)
fit_intercept = calc_best_intercept(fit_slope,y_data,x_data)

rgs_line = [fit_slope*x + fit_intercept for x in x_data ]


# Calculating r squared error , higher the error value, the better
def calculate_squared_error(y_original,y_line):
    return sum((y_line - y_original) ** 2)


def calculate_coefficent(y_orignal,y_line):
    y_mean_line = [mean(y_orignal) for _ in y_orignal]
    squared_error_mean = calculate_squared_error(y_orignal,y_mean_line)
    squared_error_regression = calculate_squared_error(y_orignal,y_line)
    return 1 - (squared_error_regression/squared_error_mean)


determination_coefficent = calculate_coefficent(y_data,rgs_line)
print(determination_coefficent)

plt.scatter(x_data,y_data)
plt.plot(x_data,rgs_line)
plt.show()
