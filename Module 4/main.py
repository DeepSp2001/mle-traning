import argparse
import os
import pickle

import mlflow
from housing_price_predictor import ingest_data, logger, score, train

"""
This python script is used to load, train and score the housing dataset.

Example
-------
    $ python main.py  --dataset_folder --model_folder

Attributes
----------
dataset_folder : string
    A specific folder to contain the datasets. Default string is datasets.
model_folder : string
    A specific folder to contain the model_pickles. Default string is model_pickles.
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-df", "--dataset_folder", default="datasets")
    parser.add_argument("-mf", "--model_folder", default="model_pickles")
    parser.add_argument("-ll", "--log-level", default="DEBUG")
    parser.add_argument("-lp", "--log-path", default=None)
    parser.add_argument("-ncl", "--no-console-log", default=False)
    args = parser.parse_args()

    logger_ = logger.logger_start(
        log_level=args.log_level,
        log_path=args.log_path,
        console_=(not args.no_console_log),
    )
    with mlflow.start_run(run_name="Main Run") as main_run:

        print("RUN ID score:", main_run.info.run_id)

        mlflow.log_param("main_run_param", "main_run_value")

        housing_prepared, housing_labels, X_test_prepared, y_test = (
            ingest_data.data_modification()
        )

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

        lin_reg, tree_reg, final_model = train.model_training(
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

        score.model_scores(
            housing_prepared,
            housing_labels,
            X_test_prepared,
            y_test,
            lin_reg,
            tree_reg,
            final_model,
        )
