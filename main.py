import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


def main():
    data = pd.read_csv('files/oil.csv', delimiter=',')
    X = data[['palmitic', 'palmitoleic', 'stearic', 'oleic', 'linoleic', 'linolenic', 'arachidic', 'eicosenoic']]

    y_region = data['region']

    X_train_region, X_test_region, y_train_region, y_test_region = train_test_split(X, y_region, test_size=0.4,
                                                                                    random_state=0)

    best_k_region = None
    best_accuracy_region = 0
    error_rates_region = []

    for k in range(1, 36):
        knn_region = KNeighborsClassifier(n_neighbors=k)
        knn_region.fit(X_train_region, y_train_region)
        y_pred_region = knn_region.predict(X_test_region)
        accuracy_region = accuracy_score(y_test_region, y_pred_region)
        error_rates_region.append(np.mean(y_pred_region != y_test_region))

        if accuracy_region > best_accuracy_region:
            best_accuracy_region = accuracy_region
            best_k_region = k

    plt.figure(figsize=(10, 6))
    plt.plot(error_rates_region)
    plt.xlabel("Количество соседей (k)")
    plt.ylabel("Ошибка")
    plt.title("Ошибки классификации регионов")
    plt.grid()
    plt.show()

    print("Лучшее значение k для задачи минимума:", best_k_region)

    best_knn_region = KNeighborsClassifier(n_neighbors=best_k_region)
    best_knn_region.fit(X_train_region, y_train_region)
    y_pred_region = best_knn_region.predict(X_test_region)

    conf_matrix_region = confusion_matrix(y_test_region, y_pred_region)
    print('Таблица сопряженности для задачи минимума:')
    print(conf_matrix_region)

    error_rate_region = 1 - best_accuracy_region
    print(f'Процент ошибок на тестовой выборке для задачи минимума: {error_rate_region * 100:.2f}%')

    y_area = data['area']

    X_train_area, X_test_area, y_train_area, y_test_area = train_test_split(X, y_area, test_size=0.4, random_state=0)

    best_k_area = None
    best_accuracy_area = 0
    error_rates_area = []

    for k in range(1, 36):
        knn_area = KNeighborsClassifier(n_neighbors=k)
        knn_area.fit(X_train_area, y_train_area)
        y_pred_area = knn_area.predict(X_test_area)
        accuracy_area = accuracy_score(y_test_area, y_pred_area)
        error_rates_area.append(np.mean(y_pred_area != y_test_area))

        if accuracy_area > best_accuracy_area:
            best_accuracy_area = accuracy_area
            best_k_area = k

    plt.figure(figsize=(10, 6))
    plt.plot(error_rates_area)
    plt.xlabel("Количество соседей (k)")
    plt.ylabel("Ошибка")
    plt.title("Ошибки классификации областей внутри региона")
    plt.grid()
    plt.show()

    print("Лучшее значение k для задачи максимума:", best_k_area)

    best_knn_area = KNeighborsClassifier(n_neighbors=best_k_area)
    best_knn_area.fit(X_train_area, y_train_area)
    y_pred_area = best_knn_area.predict(X_test_area)

    conf_matrix_area = confusion_matrix(y_test_area, y_pred_area)
    print('Таблица сопряженности для задачи максимума:')
    print(conf_matrix_area)

    error_rate_area = 1 - best_accuracy_area
    print(f'Процент ошибок на тестовой выборке для задачи максимума: {error_rate_area * 100:.2f}%')


if __name__ == "__main__":
    main()