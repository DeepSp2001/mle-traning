# housing_price_predictor

Unveiling the future of your real estate dreams, Housing Price Predictor empowers you with market knowledge. This innovative package analyzes complex data to provide insightful predictions on home prices. By considering factors like longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population it empowers you to make informed decisions. Navigate the housing market with confidence - whether buying, selling, or investing, Housing Price Predictor is your key to unlocking real estate possibilities.

## Modules

Modules of the package housing_price_predictor help in different parts of the program. The modules are :-

- ingest_data : - Download and create training and validation datasets
- train :- To train the model(s)
- score :- To score the model(s)
- logger :- To store logger info and activate logger.

Refer to docs for more info.

## Scripts

- **ingest_data.py** :- To download and create training and validation datasets. The script accepts the output folder/file path as an user argument. Arguments are :-

  - dataset_folder :- Location to store the dataset.
  - log-level :- One of ["INFO","DEBUG","WARNING","ERROR","CRITICAL"], default - `"DEBUG"`
  - log-path :- Folder to store the log files
  - no-console-log :- Wether to write logs to console

- **train.py** :- To train the model(s). The script accepts arguments for input (dataset) and output folders (model pickles).

  - dataset_folder :- Location to extract the dataset.
  - model_folder :- Location to store the trained models.
  - log-level :- One of ["INFO","DEBUG","WARNING","ERROR","CRITICAL"], default - `"DEBUG"`
  - log-path :- Folder to store the log files
  - no-console-log :- Wether to write logs to console

- **score.py** :- To score the model(s). The script accepts arguments for model folder, dataset folder and any outputs.

  - dataset_folder :- Location to extract the dataset.
  - model_folder :- Location to extract the trained models.
  - log-level :- One of ["INFO","DEBUG","WARNING","ERROR","CRITICAL"], default - `"DEBUG"`
  - log-path :- Folder to store the log files
  - no-console-log :- Wether to write logs to console

## Testing

Please install pytest in your conda environment before testing.

- We can run all the test cases present in the test folder by using pytest command:

```
(conda_env):~/tests$ pytest -v
```

- To run a specific test case or function:

```
(conda_env):~/tests/unittests/tsttemplate$ pytest -v <test_module>.py::<test_function>
```

## Docs

Run the folloing command to launch a local UI server.

    ```
    (conda_env)youruser@yourpc:~yourWorkspacePath/<project_name>/docs/build/html$ python3 -m http.server {port}
    ```
    Example :-
    ```
    $ python3 -m http.server 8000
    ```

Now open a browser and navigate to https://localhost:8000.
For editing and development, please refer to [Sphinx](https://www.writethedocs.org/guide/tools/sphinx/) documentation.
