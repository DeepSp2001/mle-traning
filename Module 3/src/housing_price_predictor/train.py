import argparse
import os
import pickle
import warnings
from datetime import datetime

import mlflow
import numpy as np
import pandas as pd
from housing_price_predictor.logger import logger_start
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor

"""
This python script is used to train the models using the train data.

Example
-------
    $ python train.py  --dataset_folder --model_folder

Attributes
----------
dataset_folder : string
    A specific folder to contain the datasets. Default string is datasets.
model_folder : string
    A specific folder to contain the model_pickles. Default string is model_pickles.
"""


def model_training(housing_prepared, housing_labels, X_test_prepared, y_test):
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

    Returns:
        lin_reg : LinearRegression
            Best Linear Regression trained
        tree_reg : DecisionTreeRegressor
            Best Decision Tree Regressor trained
        final_model : RandomForestRegressor
            Best Random Forest Regressor trained
    """

    with mlflow.start_run(run_name="Train Run", nested=True) as train_run:

        print("RUN ID train:", train_run.info.run_id)

        mlflow.autolog()
        mlflow.log_param("train_run_param", "train_run_value")

        with mlflow.start_run(run_name="LinearRegression", nested=True) as linreg_run:
            print("RUN ID LinearRegression:", linreg_run.info.run_id)
            mlflow.sklearn.autolog()
            mlflow.log_param("linreg_run_param", "linreg_run_value")
            lin_reg = LinearRegression()
            lin_reg.fit(housing_prepared, housing_labels)

        with mlflow.start_run(
            run_name="DecisionTreeRegressor", nested=True
        ) as decreg_run:
            print("RUN ID DecisionTreeRegressor:", decreg_run.info.run_id)
            mlflow.sklearn.autolog()
            mlflow.log_param("decreg_run_param", "decreg_run_value")
            tree_reg = DecisionTreeRegressor(random_state=42)
            tree_reg.fit(housing_prepared, housing_labels)

        param_distribs = {
            "n_estimators": randint(low=1, high=200),
            "max_features": randint(low=1, high=8),
        }

        forest_reg = RandomForestRegressor(random_state=42)
        rnd_search = RandomizedSearchCV(
            forest_reg,
            param_distributions=param_distribs,
            n_iter=10,
            cv=5,
            scoring="neg_mean_squared_error",
            random_state=42,
        )
        rnd_search.fit(housing_prepared, housing_labels)
        cvres = rnd_search.cv_results_
        for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
            print(np.sqrt(-mean_score), params)

        with mlflow.start_run(run_name="RandomizedSearchCV", nested=True) as run1:
            print("RUN ID RandomizedSearchCV:", run1.info.run_id)
            mlflow.log_param("folds", rnd_search.cv)

            print("Logging metrics")
            i = 1
            for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
                with mlflow.start_run(
                    run_name="RandomizedSearchCV " + str(i), nested=True
                ):

                    mlflow.log_params(params)
                    mlflow.log_metric("mean_test_score", np.sqrt(-mean_score), step=i)
                i += 1

        param_grid = [
            # try 12 (3×4) combinations of hyperparameters
            {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
            # then try 6 (2×3) combinations with bootstrap set as False
            {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
        ]

        forest_reg = RandomForestRegressor(random_state=42)
        # train across 5 folds, that's a total of (12+6)*5=90 rounds of training
        grid_search = GridSearchCV(
            forest_reg,
            param_grid,
            cv=5,
            scoring="neg_mean_squared_error",
            return_train_score=True,
        )
        grid_search.fit(housing_prepared, housing_labels)

        grid_search.best_params_
        cvres = grid_search.cv_results_
        for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
            print(np.sqrt(-mean_score), params)

        feature_importances = grid_search.best_estimator_.feature_importances_
        sorted(zip(feature_importances, housing_prepared.columns), reverse=True)

        with mlflow.start_run(run_name="GridSearchCV", nested=True) as run2:
            print("RUN ID GridSearchCV:", run2.info.run_id)
            mlflow.log_param("folds", grid_search.cv)

            print("Logging metrics")
            i = 1
            for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):

                with mlflow.start_run(run_name="GridSearchCV " + str(i), nested=True):

                    mlflow.log_params(params)
                    mlflow.log_metric("mean_test_score", np.sqrt(-mean_score), step=i)
                i += 1

        final_model = grid_search.best_estimator_

    return lin_reg, tree_reg, final_model


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
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

    logger_.info("Datasets loaded.")

    lin_reg, tree_reg, final_model = model_training(
        housing_prepared, housing_labels, X_test_prepared, y_test
    )

    logger_.info("Models trained.")

    os.makedirs(str(args.model_folder), exist_ok=True)

    pkl_path = os.path.join(str(args.model_folder), "lin_reg.pkl")
    pickle.dump(lin_reg, open(pkl_path, "wb"))

    pkl_path = os.path.join(str(args.model_folder), "tree_reg.pkl")
    pickle.dump(tree_reg, open(pkl_path, "wb"))

    pkl_path = os.path.join(str(args.model_folder), "final_model.pkl")
    pickle.dump(final_model, open(pkl_path, "wb"))

    logger_.info("Models saved.")
