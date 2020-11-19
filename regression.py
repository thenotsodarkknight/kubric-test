import requests
import pandas
import scipy
import numpy
import sys


TRAIN_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_test.csv"


def predict_price(area) -> float:
    """
    This method must accept as input an array `area` (represents a list of areas sizes in sq feet) and must return the respective predicted prices (price per sq foot) using the linear regression model that you build.

    You can run this program from the command line using `python3 regression.py`.
    """
    response = requests.get(TRAIN_DATA_URL)
    # YOUR IMPLEMENTATION HERE
    ...


if __name__ == "__main__":
    # DO NOT CHANGE THE FOLLOWING CODE
    from data import validation_data
    areas = numpy.array(list(validation_data.keys()))
    prices = numpy.array(list(validation_data.values()))
    predicted_prices = predict_price(areas)
    rmse = numpy.sqrt(numpy.mean((predicted_prices - prices) ** 2))
    try:
        assert rmse < 170
    except AssertionError:
        print(f"Root mean squared error is too high - {rmse}. Expected it to be under 170")
        sys.exit(1)
import requests
import pandas as pd
import scipy
import numpy
import numpy as np
import sys


TRAIN_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_test.csv"


def predict_price(area) -> float:
    """
    This method must accept as input an array `area` (represents a list of areas sizes in sq feet) and must return the respective predicted prices (price per sq foot) using the linear regression model that you build.
    You can run this program from the command line using `python3 regression.py`.
    """
    response = requests.get(TRAIN_DATA_URL)
    # YOUR IMPLEMENTATION HERE
    df_train = pd.read_csv(TRAIN_DATA_URL, header=None)
    df_test = pd.read_csv(TEST_DATA_URL, header=None)
    df_train = df_train.transpose()
    df_test = df_test.transpose()
    df_train.columns = df_train.loc[0]
    df_test.columns = df_test.loc[0]
    df_train.drop(0, inplace=True)
    df_test.drop(0, inplace=True)
    df_train = df_train.reset_index().drop(columns=['index'])
    df_test = df_test.reset_index().drop(columns=['index'])
    x_train = df_train.iloc[:, 0].to_numpy()
    y_train = df_train.iloc[:, 1].to_numpy()
    mn_a = np.min(x_train)
    mx_a = np.max(x_train)
    x_train = (x_train - mn_a)/(mx_a - mn_a)
    mn_p = np.min(y_train)
    mx_p = np.max(y_train)
    y_train = (y_train - mn_p)/(mx_p - mn_p)
    w = 1
    b = 1
    f = 1e-5
    grad_w = 1
    grad_b = 1
    while abs(grad_w)+abs(grad_b) > 1e-4:
        grad_w = -2 * np.sum(np.multiply(y_train - w * x_train - b, x_train))
        grad_b = -2 * np.sum(y_train - w * x_train - b)
        w = w - f*grad_w
        b = b - f*grad_b
    w_real = w * (mx_p - mn_p)/(mx_a - mn_a)
    b_real = (b - w * (mn_a)/(mx_a - mn_a))*(mx_p - mn_p) + mn_p
    return w_real * area + b_real    

if __name__ == "__main__":
    # DO NOT CHANGE THE FOLLOWING CODE
    from data import validation_data
    areas = numpy.array(list(validation_data.keys()))
    prices = numpy.array(list(validation_data.values()))
    predicted_prices = predict_price(areas)
    rmse = numpy.sqrt(numpy.mean((predicted_prices - prices) ** 2))
    try:
        assert rmse < 170
    except AssertionError:
        print(f"Root mean squared error is too high - {rmse}. Expected it to be under 170")
        sys.exit(1)
    print(f"Success. RMSE = {rmse}")    print(f"Success. RMSE = {rmse}")
