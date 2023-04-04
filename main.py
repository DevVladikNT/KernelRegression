import numpy as np
from matplotlib import pyplot as plt
from functools import lru_cache


# Исходные данные по коронавирусу в СПб с 03.2020 по 06.2020
x = np.arange(118)
x = x / len(x)  # Приводим X к промежутку [0; 1] для корректной работы всех ядер
# x = np.sort(np.random.random(118))
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
# Тип ядра
# 1 - Оптимальное (Епанечникова)
# 2 - Квартичное
# 3 - Треугольное
# 4 - Прямоугольное
# 5 - Гауссовское
nuclear_type = 5
gamma_nuclear_type = 5
h = 0.1  # Размер окна
predict = 0.9  # Какой % выборки будет использован для обучения предсказанию
bins_count = 10


# Гауссовское ядро
@lru_cache()
def nuclear(r, type_val=nuclear_type):
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


# Расстояние от x1 до x2
@lru_cache()
def distance(x1, x2):
    return np.abs(x1 - x2)


# Значение ядерной регрессии в точке xi
def regression_result(xi, x_arr=x, y_arr=y, gamma_arr=np.ones(len(x))):
    upper_part = 0
    lower_part = 0
    for i in range(len(x_arr)):
        k_val = nuclear(distance(xi, x_arr[i]) / h, nuclear_type)
        upper_part += y_arr[i] * gamma_arr[i] * k_val
        lower_part += gamma_arr[i] * k_val
    return upper_part / lower_part


# Среднеквадратичное отклонение реальных и вычисленных данных
def error(regressed, actual):
    result = []
    for i in range(len(actual)):
        result.append((regressed[i] - actual[i]) ** 2)
    return np.sqrt(np.mean(np.array(result)))


def main():
    global h

    # Выводим график, описывающий ядро
    x_nuclear = np.arange(100)
    x_nuclear = x_nuclear / len(x_nuclear)
    y_nuclear = []
    for x_cur in x_nuclear:
        y_nuclear.append(nuclear(x_cur))
    plt.plot(x_nuclear, y_nuclear)
    plt.title('Ядро')
    plt.xlabel('r')
    plt.ylabel('F(r)')
    plt.show()

    # Строим график плотности
    prob = []
    for xi in x:
        sum_i = 0
        for xj in x:
            sum_i += nuclear((xi - xj) / h)
        sum_i /= len(x) * h
        prob.append(sum_i)
    plt.hist(x, bins=bins_count, edgecolor='black', label='Data histogram')
    plt.plot(x, np.array(prob) * (len(x) / bins_count), label='Плотность распределения')
    plt.title('Распределение')
    plt.xlabel('x')
    plt.ylabel('count')
    plt.legend()
    plt.show()

    # Выводим график, описывающий ядерную регрессию
    y_fact = []
    for i in range(len(x)):
        y_fact.append(regression_result(x[i]))
    plt.plot(x, y_fact, label='Ядерная регрессия')
    plt.plot(x, y, label='Реальные данные')
    plt.title(f'Ядерная регрессия, h={h}')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.show()

    # Вычисляем взвешенную ядерную регрессию
    last_diff = 1000000
    gamma = np.ones(len(x))
    while True:
        y_fact = []
        # Вычисляем значения с коэффициентами gamma
        for i in range(len(x)):
            y_fact.append(regression_result(x[i], gamma_arr=gamma))
        # Если ошибка выросла или стабилизировалась, выходим
        if (diff := error(y_fact, y)) >= last_diff:
            print(diff)
            break
        print(diff)
        last_diff = diff
        # Вычисляем новые значения коэффициентов gamma
        for i in range(len(y_fact)):
            gamma[i] = nuclear(np.abs(y_fact[i] - y[i]), gamma_nuclear_type)
    plt.plot(x, y_fact, label='Ядерная регрессия')
    plt.plot(x, y, label='Реальные данные')
    plt.title(f'Взвешенная ядерная регрессия, h={h}')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.show()

    # Разбиваем выборку на тренировочную и тестовую
    x_train = x[:int(predict * len(x))]
    y_train = y[:int(predict * len(y))]
    y_fact = []
    # Вычисляем значения ядерной регресси на тренировочном диапазоне
    for xi in x_train:
        y_fact.append(regression_result(xi, x_arr=x_train, y_arr=y_train))
    x_test = x[int(predict * len(x)):]
    y_test = []
    # Вычисляем предсказание на тестовом диапазоне,
    # используя ядерную регрессию, построенную на тренировочном диапазоне
    for xi in x_test:
        y_test.append(regression_result(xi, x_arr=x_train, y_arr=y_train))
    plt.plot(x_train, y_fact, label='Ядерная регрессия')
    plt.plot(x_test, y_test, label='Предсказание')
    plt.plot(x, y, label='Реальные данные')
    plt.title(f'Предсказание ядерной регрессией, h={h}')
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
