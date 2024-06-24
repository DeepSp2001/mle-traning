import pandas as pd
from housing_price_predictor import ingest_data


def test_ingestion():
    housing_prepared, housing_labels, X_test_prepared, y_test = (
        ingest_data.data_modification()
    )

    assert (
        type(housing_prepared) == pd.DataFrame
    ), "Training dataframe not loaded successfully"
    assert type(housing_labels) == pd.Series, "Training labels not loaded successfully"
    assert (
        type(X_test_prepared) == pd.DataFrame
    ), "Testing dataframe not loaded successfully"
    assert type(y_test) == pd.Series, "Testing labels not loaded successfully"
