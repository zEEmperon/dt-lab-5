import pandas as pd
import numpy as np
from tabulate import tabulate


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


def get_beta_coefs(df: pd.DataFrame) -> np.array:
    df_K1 = get_K1(df)
    df_K2 = get_K2(df)

    df_K1_without_class_attr = get_df_without_class_attr(df_K1)
    df_K2_without_class_attr = get_df_without_class_attr(df_K2)

    K1_m = df_K1_without_class_attr.mean()
    K2_m = df_K2_without_class_attr.mean()

    Sk1 = df_K1_without_class_attr.cov()
    Sk2 = df_K2_without_class_attr.cov()

    n1 = len(df_K1)
    n2 = len(df_K2)
    Sk = 1 / (n1 + n2 - 2) * (n1 * Sk1 + n2 * Sk2)

    inv_Sk = pd.DataFrame(np.linalg.inv(Sk.values), Sk.columns, Sk.index)
    beta_coefs = inv_Sk * (K1_m - K2_m)
    beta_coefs = np.diagonal(beta_coefs)

    return beta_coefs


def main():
    training_set_filename = "training_set.csv"
    test_set_filename = "test_set.csv"

    training_df = load_csv(training_set_filename)
    test_df = load_csv(test_set_filename)

    training_df_without_class_attr = get_df_without_class_attr(training_df)
    test_df_without_class_attr = get_df_without_class_attr(test_df)

    beta_coefs = get_beta_coefs(training_df)
    print(beta_coefs)


if __name__ == '__main__':
    main()
