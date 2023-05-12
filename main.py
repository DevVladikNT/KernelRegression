import numpy as np
from matplotlib import pyplot as plt
from functools import lru_cache


# Initial data on coronavirus in St. Petersburg from 03.2020 to 06.2020
x = np.arange(118)
x = x / len(x)  # We bring X to the interval [0; 1] for the correct operation of all cores
y = np.array([0, 0, 1, 0, 1, 0, 0, 0, 3, 1, 0, 3, 0, 0, 1, 4, 2, 0,
              0, 5, 0, 5, 0, 11, 5, 8, 48, 27, 22, 9, 15, 20, 35, 69,
              34, 44, 35, 80, 69, 121, 121, 130, 154, 424, 139, 114,
              86, 127, 294, 191, 253, 215, 151, 161, 198, 290, 336,
              349, 323, 295, 317, 226, 312, 306, 375, 425, 414, 307,
              339, 435, 460, 541, 525, 451, 425, 453, 455, 408, 389,
              363, 384, 374, 363, 387, 383, 369, 365, 369, 364, 376,
              380, 375, 378, 347, 340, 326, 318, 313, 303, 274, 262,
              256, 252, 234, 228, 231, 228, 232, 229, 217, 220, 218,
              221, 223, 224, 217, 219, 253])
# Core type
# 1 - Epanechnikov core
# 2 - Quartic core
# 3 - Triangle core
# 4 - Rectangular core
# 5 - Gaussian kernel
# 6 - Cauchy core
core_type = 5
gamma_core_type = 6
h = 0.1  # Window size
predict = 0.9  # What % of the sample will be used for prediction training
bins_count = 10


# Returns core value
@lru_cache()
def core(r, type_val=core_type):
    if type_val == 1:
        return 3/4 * (1 - r ** 2)
    elif type_val == 2:
        return 15/16 * (1 - r ** 2) ** 2
    elif type_val == 3:
        return 1 - np.abs(r)
    elif type_val == 4:
        return 1/2
    elif type_val == 5:
        return (2 * np.pi) ** (-1/2) * np.exp(-1/2 * r ** 2)
    elif type_val == 6:
        return 1 / (np.pi * (1 + r ** 2))


# Distance from x1 to x2
@lru_cache()
def distance(x1, x2):
    return np.abs(x1 - x2)


# The value of the kernel regression at point xi
def regression_result(xi, x_arr=x, y_arr=y, gamma_arr=np.ones(len(x))):
    upper_part = 0
    lower_part = 0
    for i in range(len(x_arr)):
        k_val = core(distance(xi, x_arr[i]) / h, core_type)
        upper_part += y_arr[i] * gamma_arr[i] * k_val
        lower_part += gamma_arr[i] * k_val
    return upper_part / lower_part


# Standard deviation of real and calculated data
def error(regressed, actual):
    result = []
    for i in range(len(actual)):
        result.append((regressed[i] - actual[i]) ** 2)
    return np.sqrt(np.mean(np.array(result)))


def main():
    global h

    # We output a graph describing the core
    x_nuclear = np.arange(100)
    x_nuclear = x_nuclear / len(x_nuclear)
    y_nuclear = []
    for x_cur in x_nuclear:
        y_nuclear.append(core(x_cur))
    plt.plot(x_nuclear, y_nuclear)
    plt.title('Core')
    plt.xlabel('r')
    plt.ylabel('F(r)')
    plt.show()

    # Building a density plot
    prob = []
    for xi in x:
        sum_i = 0
        for xj in x:
            sum_i += core((xi - xj) / h)
        sum_i /= len(x) * h
        prob.append(sum_i)
    plt.hist(x, bins=bins_count, edgecolor='black', label='Data histogram')
    plt.plot(x, np.array(prob) * (len(x) / bins_count), label='Distribution density')
    plt.title('Distribution')
    plt.xlabel('x')
    plt.ylabel('count')
    plt.legend()
    plt.show()

    # Building a plot describing the kernel regression
    y_fact = []
    for i in range(len(x)):
        y_fact.append(regression_result(x[i]))
    plt.plot(x, y_fact, label='Kernel regression')
    plt.plot(x, y, label='Real data')
    plt.title(f'Kernel regression, h={h}')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.show()

    # Building a plot describing the weighted kernel regression
    last_diff = 1000000
    gamma = np.ones(len(x))
    while True:
        y_fact = []
        # We calculate values with gamma coefficients
        for i in range(len(x)):
            y_fact.append(regression_result(x[i], gamma_arr=gamma))
        # If the error has grown or stabilized, we exit
        if (diff := error(y_fact, y)) >= last_diff:
            print(diff)
            break
        print(diff)
        last_diff = diff
        # We calculate new values of gamma coefficients
        for i in range(len(y_fact)):
            gamma[i] = core(np.abs(y_fact[i] - y[i]), gamma_core_type)
    plt.plot(x, y_fact, label='Kernel regression')
    plt.plot(x, y, label='Real data')
    plt.title(f'Weighted kernel regression, h={h}')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.show()

    # We split the sample into training and test
    x_train = x[:int(predict * len(x))]
    y_train = y[:int(predict * len(y))]
    y_fact = []
    # We calculate the values of kernel regression in the training range
    for xi in x_train:
        y_fact.append(regression_result(xi, x_arr=x_train, y_arr=y_train))
    x_test = x[int(predict * len(x)):]
    y_test = []
    # We calculate the prediction on the test range
    # using a kernel regression built on the training range
    for xi in x_test:
        y_test.append(regression_result(xi, x_arr=x_train, y_arr=y_train))
    plt.plot(x_train, y_fact, label='Kernel regression')
    plt.plot(x_test, y_test, label='Prediction')
    plt.plot(x, y, label='Real data')
    plt.title(f'Prediction by kernel regression, h={h}')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.show()

    # min_error = 1000000
    # min_error_h = 0
    # for hi in ((h_arr := np.arange(100) + 1) / len(h_arr)):
    #     h = hi
    #     y_fact = []
    #     for i in range(len(x)):
    #         y_fact.append(regression_result(x[i], x, y, np.ones(len(x))))
    #     if new_error := error(y_fact, y) < min_error:
    #         min_error = new_error
    #         min_error_h = hi
    # print(min_error_h)
    # h = min_error_h
    # y_fact = []
    # for i in range(len(x)):
    #     y_fact.append(regression_result(x[i], x, y, np.ones(len(x))))
    # plt.plot(x, y_fact)
    # plt.plot(x, y)
    # plt.show()


if __name__ == '__main__':
    main()
