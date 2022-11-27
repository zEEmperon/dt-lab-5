import math
import pandas as pd
import numpy as np
from tabulate import tabulate
from itertools import combinations

K1 = 'K1'
K2 = 'K2'
alpha = 5
beta = 4


def load_csv(filename):
    return pd.read_csv(filename)


def print_task(no):
    print("Завдання {}:".format(no))


def get_K1(df: pd.DataFrame) -> pd.DataFrame:
    return df[df['Class'] == 1]


def get_K2(df: pd.DataFrame) -> pd.DataFrame:
    return df[df['Class'] != 1]


def get_df_without_class_attr(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(['Class'], axis=1)


def get_M_x(x_subset: pd.DataFrame) -> int:
    n = len(x_subset)
    return (1 / n) * x_subset.sum()


def get_M_x_for_all_x(df: pd.DataFrame) -> dict:
    dictionary = {}
    for header_label in df:
        dictionary[header_label] = get_M_x(df[header_label])
    return dictionary


def get_D_x(x_subset: pd.DataFrame, M_x: int) -> int:
    n = len(x_subset) - 1
    return (1 / n) * sum([*map(lambda x: (x - M_x) ** 2, x_subset.to_numpy())])


def get_D_x_for_all_x(df: pd.DataFrame, m_x: dict) -> dict:
    dictionary = {}
    for header_label in df:
        dictionary[header_label] = get_D_x(df[header_label], m_x[header_label])
    return dictionary


def get_normed_x(df: pd.DataFrame, d_x: dict) -> pd.DataFrame:
    return df / d_x.values()


def get_instances_combinations(list_with_indexes: list) -> list:
    return [*combinations(list_with_indexes, 2)]


def get_R_and_phi(df: pd.DataFrame) -> list:
    df_without_class_attr = get_df_without_class_attr(df)
    instances_combinations = get_instances_combinations(df.index)

    get_Q = lambda class_value: 1 if class_value == 1 else -1

    res_arr = []

    for comb in instances_combinations:
        first_index, second_index = comb
        R = math.sqrt(sum((df_without_class_attr.loc[first_index] - df_without_class_attr.loc[second_index]) ** 2))
        Q = get_Q(df.loc[second_index]['Class'])
        phi = Q / 1 - alpha * R ** beta
        res_arr.append([first_index, second_index, R, phi])
    return res_arr


def classify(df: pd.DataFrame, threshold_phi: int, phi_list: list) -> list:
    get_class = lambda y: K1 if y == 1 else K2

    n_K1 = len(get_K1(df))
    n_K2 = len(get_K2(df))

    res = []
    for i in df.index:
        actual_class = get_class(df.loc[i]['Class'])
        neighborhoods = [*filter(lambda x: x[0] == i or x[1] == i, phi_list)]

        k1_phi = 0
        k2_phi = 0
        a = 1

        for n in neighborhoods:
            n_index = n[0] if n[0] != i else n[1]
            n_phi = n[2]
            if df.loc[n_index]['Class'] == 1:
                k1_phi += n_phi
            else:
                k2_phi += n_phi

        total_potential = (1 / n_K1) * k1_phi + (1 / (n_K2 - 1)) * k2_phi
        predicted_class = K1 if total_potential >= threshold_phi else K2

        res.append([i, total_potential, predicted_class, actual_class])

    return res


def get_stats(classified: list) -> list:
    classified = np.array(classified)
    n = len(classified)
    real_K1_n = len(classified[classified[:, 3] == K1])
    real_K2_n = len(classified[classified[:, 3] == K2])
    predicted_K1_n = len(classified[classified[:, 2] == K1])
    predicted_K2_n = len(classified[classified[:, 2] == K2])
    misclassified = len(classified[classified[:, 2] != classified[:, 3]])
    return [n, real_K1_n, real_K2_n, predicted_K1_n, predicted_K2_n, misclassified]


def print_stats(stats: list) -> None:
    n, real_K1_n, real_K2_n, predicted_K1_n, predicted_K2_n, misclassified = stats
    print("Всього примірників:", n)
    print("Фактично приналежних до К1 =", real_K1_n)
    print("Фактично приналежних до К2 =", real_K2_n)
    print("Прогнозовано приналежних до К1 =", predicted_K1_n)
    print("Прогнозовано приналежних до К2 =", predicted_K2_n)
    print("Неправильно класифікованих =", misclassified)
    print("Відносна помилка = {}%".format(round((misclassified / n) * 100, 3)))


def print_stats_probabilities(stats: list) -> None:
    n, real_K1_n, real_K2_n, predicted_K1_n, predicted_K2_n, misclassified = stats
    print("P(K1) =", real_K1_n / n)
    print("P(K2) =", real_K2_n / n)
    print("P(ріш K1) =", predicted_K1_n / n)
    print("P(ріш K2) =", predicted_K2_n / n)


def main():
    training_set_filename = "training_set.csv"
    training_df = load_csv(training_set_filename)
    training_df_without_class_attr = get_df_without_class_attr(training_df)

    # Нормовані значення ознак
    m_x_for_training_set = get_M_x_for_all_x(training_df_without_class_attr)
    d_x_for_training_set = get_D_x_for_all_x(training_df_without_class_attr, m_x_for_training_set)

    normed_training_df = get_normed_x(training_df_without_class_attr, d_x_for_training_set)
    normed_training_df['Class'] = training_df['Class']

    # Узагальнена відстань та потенціал для і-ого і j-ого екземплярів
    # 276 для навчальної вибірки
    label = "Узагальнена відстань та потенціал для і-ого і j-ого екземплярів"
    R_and_phi_for_training_set = get_R_and_phi(normed_training_df)

    table_data = [
        *map(lambda results: [results[0] + 1, results[1] + 1, results[2], results[3]], R_and_phi_for_training_set)]
    col_names = ["Номер i-го примірника", "Номер j-го примірника", "Узагальнена відстань (R)", "Потенціал (phi)"]

    print()
    print_task("3.5 і 3.6")
    print(label)
    print(tabulate(table_data, headers=col_names, tablefmt="fancy_grid"))

    # classify
    threshold_phi = 15

    phi_for_training_set = [*map(lambda i: [i[0], i[1], i[2]], R_and_phi_for_training_set)]
    training_set_classification_res = classify(normed_training_df, threshold_phi, phi_for_training_set)
    training_set_classification_stats = get_stats(training_set_classification_res)

    col_names = ['Номер примірника', 'Сумарний потенціал', 'Прогн. клас', 'Справжній клас']
    table_data = training_set_classification_res

    print()
    print(tabulate(table_data, headers=col_names, tablefmt="fancy_grid"))

    print()
    print("Результати класифікації для навчальної вибірки:")
    print()
    print_stats(training_set_classification_stats)
    print()
    print_stats_probabilities(training_set_classification_stats)


if __name__ == '__main__':
    main()
