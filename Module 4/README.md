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

## ML Flow

MLflow is an open-source platform for machine learning lifecycle management. We use MLflow in our project for understanding the parameters of our models.

To run mlflow on our `main.py` python script, first install the `v0.4` of our package present in `dist` folder in `Module 3`. Then run the following commands :-

```
python main.py
mlflow ui --port 8080
```

Now open a browser and navigate to https://127.0.0.1:8080. The `images/mlflow_work` folder contains a few demo images.

## Packaging and Installation

### Installation
Unzip the file and go to the dist folder. There are 2 files, namely  `housing_price_predictor-0.4.tar.gz` and `housing_price_predictor-0.4-py3-none-any.whl`. Use pip to install these files. 

```
conda create env --name test python
conda activate test
cd Module \4
cd dist
pip install ./housing_price_predictor-0.4.tar.gz
```

### Test the installation 
For testing if isntallation has correctly been done, install `pytest` and run it on the whole Module folder. 
```
(conda_env):~$ pip install pytest
(conda_env):~$ cd test
(conda_env):~/tests$ pytest -v
```

### Run the application 
To run the application, navigate to the `Module 4` folder and run the following command.
```
python main.py
mlflow ui --port 8080
```

### Look for log
To look for logs, give a file name with the main.py file to store the log.
```
python main.py --log-path log.txt
```