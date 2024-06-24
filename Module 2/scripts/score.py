import argparse
import os
import pickle

import numpy as np
import pandas as pd
from housing_price_predictor.logger import logger_start
from sklearn.metrics import mean_absolute_error, mean_squared_error

"""
This python script is used to score the trained models using the test data.

Example
-------
    $ python score.py --dataset_folder --model_folder

Attributes
----------
dataset_folder : string
    A specific folder to contain the datasets. Default string is datasets.
model_folder : string
    A specific folder to contain the model_pickles. Default string is model_pickles.
"""


def model_scores(
    housing_prepared,
    housing_labels,
    X_test_prepared,
    y_test,
    lin_reg,
    tree_reg,
    final_model,
):
    """Loads housing data to a Pandas DataFrame.

    Args:
        housing_prepared : pandas.DataFrame
            Training dataframe
        housing_labels : pandas.Series
            Training labels
        X_test_prepared : pandas.DataFrame
            Testing dataframe
        y_test : pandas.Series
            Testing labels
        lin_reg : LinearRegression
            Best Linear Regression trained
        tree_reg : DecisionTreeRegressor
            Best Decision Tree Regressor trained
        final_model : RandomForestRegressor
            Best Random Forest Regressor trained
    """
    housing_predictions = lin_reg.predict(housing_prepared)
    lin_mse = mean_squared_error(housing_labels, housing_predictions)
    lin_rmse = np.sqrt(lin_mse)
    print(f"For the LinearRegression model, root_mean_squared_error = {lin_rmse}")

    lin_mae = mean_absolute_error(housing_labels, housing_predictions)
    print(f"For the LinearRegression model, mean_absolute_error = {lin_mae}")

    housing_predictions = tree_reg.predict(housing_prepared)
    tree_mse = mean_squared_error(housing_labels, housing_predictions)
    tree_rmse = np.sqrt(tree_mse)
    tree_rmse
    # print(f"For the DecisionTreeRegressor model, root_mean_squared_error = {tree_rmse}")

    final_predictions = final_model.predict(X_test_prepared)
    final_mse = mean_squared_error(y_test, final_predictions)
    final_rmse = np.sqrt(final_mse)
    print(f"For the Best Final Model model, root_mean_squared_error = {final_rmse}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-df", "--dataset_folder", default="datasets")
    parser.add_argument("-mf", "--model_folder", default="model_pickles")
    parser.add_argument("-ll", "--log-level", default="DEBUG")
    parser.add_argument("-lp", "--log-path", default=None)
    parser.add_argument("-ncl", "--no-console-log", default=False)
    args = parser.parse_args()

    logger_ = logger_start(
        log_level=args.log_level,
        log_path=args.log_path,
        console_=(not args.no_console_log),
    )

    csv_path = os.path.join(str(args.dataset_folder), "housing_prepared.csv")
    housing_prepared = pd.read_csv(csv_path)

    csv_path = os.path.join(str(args.dataset_folder), "housing_labels.csv")
    housing_labels = pd.read_csv(csv_path)

    csv_path = os.path.join(str(args.dataset_folder), "X_test_prepared.csv")
    X_test_prepared = pd.read_csv(csv_path)

    csv_path = os.path.join(str(args.dataset_folder), "y_test.csv")
    y_test = pd.read_csv(csv_path)

    logger_.info("Datasets loaded")

    pkl_path = os.path.join(str(args.model_folder), "lin_reg.pkl")
    lin_reg = pickle.load(open(pkl_path, "rb"))

    pkl_path = os.path.join(str(args.model_folder), "tree_reg.pkl")
    tree_reg = pickle.load(open(pkl_path, "rb"))

    pkl_path = os.path.join(str(args.model_folder), "final_model.pkl")
    final_model = pickle.load(open(pkl_path, "rb"))

    logger_.info("Models loaded.")

    model_scores(
        housing_prepared,
        housing_labels,
        X_test_prepared,
        y_test,
        lin_reg,
        tree_reg,
        final_model,
    )
