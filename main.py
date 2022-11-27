import pandas as pd
import numpy as np
from tabulate import tabulate


def load_csv(filename):
    return pd.read_csv(filename)


def print_task(no):
    print("Завдання {}:".format(no))


def main():
    training_set_filename = "training_set.csv"
    test_set_filename = "test_set.csv"

    training_df = load_csv(training_set_filename)
    test_df = load_csv(test_set_filename)


if __name__ == '__main__':
    main()
