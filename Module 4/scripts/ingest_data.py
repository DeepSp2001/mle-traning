import argparse
import os
import tarfile

import mlflow
import numpy as np
import pandas as pd
from housing_price_predictor.logger import logger_start
from six.moves import urllib  # type: ignore
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

"""
This python script is used to ingest and load the data into program.

Example
-------
    $ python ingest_data.py --dataset_folder

Attributes
----------
dataset_folder : string
    A specific folder to contain the datasets. Default string is datasets.
"""


def fetch_housing_data(housing_url, housing_path):
    """Fetches the housing.tar file from GitHub and unpacks it.

    Args:
        housing_url : string
            GitHub URL from where housing.tar file is downloaded.

        housing_path  : string
            Place where the housing.tar file is unpacked.
    """
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path):
    """Loads housing data to a Pandas DataFrame.

    Args:
        housing_path  : string
            Place where the housing.tar file is unpacked.

    Returns:
        pandas.DataFrame
            Housing Dataframe
    """
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def load_data():
    """Fetches the housing.tar file from GitHub and returns the housing pandas dataframe.

    Returns:
        pandas.DataFrame
            Housing Dataframe
    """
    DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
    HOUSING_PATH = os.path.join("datasets", "housing")
    HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
    fetch_housing_data(HOUSING_URL, HOUSING_PATH)
    housing = load_housing_data(HOUSING_PATH)
    return housing


def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)


def data_modification():
    """Loads the housing dataset from GitHub and splits it into traning and test datasets
    with labels.

    Returns:
        housing_prepared : pandas.DataFrame
            Training dataframe
        housing_labels : pandas.Series
            Training labels
        X_test_prepared : pandas.DataFrame
            Testing dataframe
        y_test : pandas.Series
            Testing labels
    """
    with mlflow.start_run(run_name="Ingest Run", nested=True) as ingest_run:

        print("RUN ID ingest:", ingest_run.info.run_id)

        mlflow.log_param("ingest_run_param", "ingest_run_value")

        housing = load_data()
        train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

        housing["income_cat"] = pd.cut(
            housing["median_income"],
            bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
            labels=[1, 2, 3, 4, 5],
        )

        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, test_index in split.split(housing, housing["income_cat"]):
            strat_train_set = housing.loc[train_index]
            strat_test_set = housing.loc[test_index]

        train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

        compare_props = pd.DataFrame(
            {
                "Overall": income_cat_proportions(housing),
                "Stratified": income_cat_proportions(strat_test_set),
                "Random": income_cat_proportions(test_set),
            }
        ).sort_index()
        compare_props["Rand. %error"] = (
            100 * compare_props["Random"] / compare_props["Overall"] - 100
        )
        compare_props["Strat. %error"] = (
            100 * compare_props["Stratified"] / compare_props["Overall"] - 100
        )

        for set_ in (strat_train_set, strat_test_set):
            set_.drop("income_cat", axis=1, inplace=True)

        housing = strat_train_set.copy()
        housing.plot(kind="scatter", x="longitude", y="latitude")
        housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

        one_hot = pd.get_dummies(housing["ocean_proximity"])
        housing = housing.drop("ocean_proximity", axis=1)
        housing = housing.join(one_hot)
        corr_matrix = housing.corr()
        corr_matrix["median_house_value"].sort_values(ascending=False)
        housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
        housing["bedrooms_per_room"] = (
            housing["total_bedrooms"] / housing["total_rooms"]
        )
        housing["population_per_household"] = (
            housing["population"] / housing["households"]
        )

        housing = strat_train_set.drop(
            "median_house_value", axis=1
        )  # drop labels for training set
        housing_labels = strat_train_set["median_house_value"].copy()

        imputer = SimpleImputer(strategy="median")

        housing_num = housing.drop("ocean_proximity", axis=1)

        imputer.fit(housing_num)
        X = imputer.transform(housing_num)

        housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing.index)
        housing_tr["rooms_per_household"] = (
            housing_tr["total_rooms"] / housing_tr["households"]
        )
        housing_tr["bedrooms_per_room"] = (
            housing_tr["total_bedrooms"] / housing_tr["total_rooms"]
        )
        housing_tr["population_per_household"] = (
            housing_tr["population"] / housing_tr["households"]
        )

        housing_cat = housing[["ocean_proximity"]]
        housing_prepared = housing_tr.join(pd.get_dummies(housing_cat, drop_first=True))

        X_test = strat_test_set.drop("median_house_value", axis=1)
        y_test = strat_test_set["median_house_value"].copy()

        X_test_num = X_test.drop("ocean_proximity", axis=1)
        X_test_prepared = imputer.transform(X_test_num)
        X_test_prepared = pd.DataFrame(
            X_test_prepared, columns=X_test_num.columns, index=X_test.index
        )
        X_test_prepared["rooms_per_household"] = (
            X_test_prepared["total_rooms"] / X_test_prepared["households"]
        )
        X_test_prepared["bedrooms_per_room"] = (
            X_test_prepared["total_bedrooms"] / X_test_prepared["total_rooms"]
        )
        X_test_prepared["population_per_household"] = (
            X_test_prepared["population"] / X_test_prepared["households"]
        )

        X_test_cat = X_test[["ocean_proximity"]]
        X_test_prepared = X_test_prepared.join(
            pd.get_dummies(X_test_cat, drop_first=True)
        )

    return housing_prepared, housing_labels, X_test_prepared, y_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-df", "--dataset_folder", default="datasets")
    parser.add_argument("-ll", "--log-level", default="DEBUG")
    parser.add_argument("-lp", "--log-path", default=None)
    parser.add_argument("-ncl", "--no-console-log", default=False)
    args = parser.parse_args()

    logger_ = logger_start(
        log_level=args.log_level,
        log_path=args.log_path,
        console_=(not args.no_console_log),
    )

    housing_prepared, housing_labels, X_test_prepared, y_test = data_modification()

    logger_.info("Datasets downloaded.")

    os.makedirs(str(args.dataset_folder), exist_ok=True)
    csv_path = os.path.join(str(args.dataset_folder), "housing_prepared.csv")
    housing_prepared.to_csv(csv_path, index=False)

    csv_path = os.path.join(str(args.dataset_folder), "housing_labels.csv")
    housing_labels.to_csv(csv_path, index=False)

    csv_path = os.path.join(str(args.dataset_folder), "X_test_prepared.csv")
    X_test_prepared.to_csv(csv_path, index=False)

    csv_path = os.path.join(str(args.dataset_folder), "y_test.csv")
    y_test.to_csv(csv_path, index=False)

    logger_.info("Datasets saved.")
