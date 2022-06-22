"""An introductory ETL (extract-transform-load) pipeline utilizing Prefect

The module loads in the Spaceship Titanic dataframe, performs the necessary preprocessing to get it in a format that can be used for modeling, trains model(s), and then saves the accuracy score

    Typical usage example from JupyterLab's Terminal within the src/ folder:

    $ python -m spaceship_titanic
"""

# python standard library packages
from tabnanny import verbose
from typing import List, Tuple
import os
import argparse

# third-party packages
from prefect import task, Flow

import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn import model_selection
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from xgboost import XGBClassifier

# my modules
from spaceship_titanic import _helper
from spaceship_titanic import upload_download_gcp
import spaceship_titanic

# print(spaceship_titanic.hello)
# print(upload_download_gcp.bucket_name) bucket_name is a global variable within the upload_download_gcp file that can be accessed through dot notation (it's in a different namespace than the prefect_etl.py module)


# extract
@task
def load_dataset(download_dataset: bool, path_name: str = "train.csv") -> pd.DataFrame:
    """Downloads the train.csv from GCP and loads into memory the Spacheship Titanic dataset used to the train the model

    Args:
        download_dataset: whether or not to download training set from GCP bucket
        path_name: name of the csv file for obtaining the model locally

    Returns:
        The Pandas DataFrame that is the training data
    """

    if download_dataset:
        upload_download_gcp.download_files()

    df = pd.read_csv(
        os.path.join(
            os.path.join(os.getcwd(), spaceship_titanic.data_raw_dir), path_name
        )
    )

    return df


# transform
@task
def transform_dataset(df: pd.DataFrame, xgboost_only: bool) -> Tuple:
    """Performs preprocessing on the dataset to prepare it for modeling

    Takes in the original dataframe, performs type conversion, variable selection (only using numeric for now), handling of missing values, and train-test split to avoid data leakage

    Args:
        df: the Spaceship Titanic training dataset
        xgboost_only: whether or not to only run the XGBoost model, which has automatic handling of missing data values

    Returns:
        Tuple that comprises the train-test split for the features and labels
    """

    col_to_conv = ["CryoSleep", "Transported", "VIP"]
    for col in col_to_conv:
        df[col] = 1 * df[col]  # converting from True-False to 1-0
        if col != "Transported":
            df[col] = df[col].astype("Int64")
        else:
            df[col] = df[col].astype(int)

    numeric_columns = df.select_dtypes(include=["number"])

    # removing the null row values if we're utilizing models other than xgboost
    if not xgboost_only:
        numeric_columns = _helper.remove_rows_with_nulls(numeric_columns)
    else:
        numeric_columns.fillna(spaceship_titanic.missing_value_number, inplace=True)

    # perform train-test split to save test data so that it can be used later (typically want to save for later, but not including that at this point)
    X_train, X_test, y_train, y_test = train_test_split(
        numeric_columns.drop("Transported", axis=1),
        numeric_columns["Transported"],
        test_size=spaceship_titanic.test_size,
        random_state=spaceship_titanic.random_state,
    )

    return (X_train, X_test, y_train, y_test)


# transform
@task
def create_model_output(df: Tuple, xgboost_only: bool) -> Tuple[List, List]:
    """Perform the model training and saving the results

    In order to perform the k-fold cross validation while avoiding data leakage, need to perform the normalization for each particular fold (fit and normalize the train step, and apply normalization to test step); this is the use of scikit-learn's Pipeline module. After performing the k-fold cross-validation, each result is saved as an element in a list; therefore, we have an outer list comprised of each trained model, and an inner list that contains k accuracy score corresponding to each of the k-folds

    Args:
        df: tuple that contains the result from the transform_dataset Prefect task
        xgboost_only: whether or not to only run the XGBoost model, which has automatic handling of missing data values

    Returns:
        A list of lists for each of the results, as outlined in the description above
    """
    X_train, _, y_train, _ = df

    # prepare models
    models = []
    if not xgboost_only:
        models.append(("LR", LogisticRegression()))
        models.append(("LDA", LinearDiscriminantAnalysis()))
        models.append(("KNN", KNeighborsClassifier()))
        models.append(("SVM", SVC(random_state=spaceship_titanic.random_state)))
        models.append(
            ("RF", RandomForestClassifier(random_state=spaceship_titanic.random_state))
        )
        models.append(
            (
                "GBM",
                GradientBoostingClassifier(random_state=spaceship_titanic.random_state),
            )
        )
        models.append(
            ("ABM", AdaBoostClassifier(random_state=spaceship_titanic.random_state))
        )
    models.append(
        (
            "XGB",
            XGBClassifier(
                missing=spaceship_titanic.missing_value_number,
                random_state=spaceship_titanic.random_state,
                n_jobs=-1,
            ),
        )
    )

    # print the model parameters for XGBClassifier
    # print(models[-1][1])

    results = []
    for name, model in models:

        # define the evaluation procedure
        kfold = model_selection.KFold(
            n_splits=spaceship_titanic.n_splits,
            shuffle=True,
            random_state=spaceship_titanic.random_state,
        )

        """
        To utilize scikit-learn's Pipeline, first element is the name of the step (a string) and second is configured object of the step, such as a transform or a model. The model is only supported as the final step, although we can have as many transforms as we like in the sequence.
        """
        if not xgboost_only:
            # define the pipeline
            steps = list()
            steps.append(("scaler", preprocessing.Normalizer()))
            steps.append(("model", model))
            pipeline = Pipeline(steps=steps)

            cv_results = model_selection.cross_val_score(
                pipeline,
                X_train,
                y_train,
                cv=kfold,
                scoring=spaceship_titanic.scoring,
                n_jobs=-1,
            )

        else:
            cv_results = model_selection.cross_val_score(
                model,
                X_train,
                y_train,
                cv=kfold,
                scoring=spaceship_titanic.scoring,
                n_jobs=-1,
            )
        # report performance
        print(f"{name}: {cv_results.mean():.4f}, {cv_results.std():.4f}")
        # add scores to the list
        results.append(cv_results)

    return (results, models)


# load
@task
def save_dataframe(
    results_models: List,
    column: str = "normalize_remove_na",
    csv_name: str = "prefect_model_summary.csv",
):
    """Find the mean from the k-fold cross validation for each model, and saves it to a csv file

    Args:
        results: list of lists that contains the result from the create_model_output Prefect task
        column: column name for the csv file to identify the preprocessing steps taken to obtain the particular result

    Returns:
        CSV file containing the results for each model that is saved to disk. For example:
        Models,normalize_remove_na
        LR,0.7967
        KNN,0.7617
        RF,0.7883
        SVM,0.7975
    """

    # unpacking the tuple
    results, models = results_models

    source_model_path = os.path.join(os.getcwd(), spaceship_titanic.model_folder)
    if not os.path.exists(source_model_path):
        os.mkdir(source_model_path)

    # dictionary where df_name is key, and list of results is value
    df_model_means = {}

    # full_result is the a list of lists (outer list is for each model, and inner list is the accuracy scores for each of the 5 folds
    get_model_means = []
    for cv_results in results:
        get_model_means.append(np.round(cv_results.mean(), 4))

    df_model_means[column] = get_model_means

    model_results = pd.DataFrame(
        df_model_means, index=[get_model_names[0] for get_model_names in models]
    )

    return model_results.to_csv(
        os.path.join(source_model_path, csv_name), index_label="Models"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="An introductory ETL (extract-transform-load) pipeline utilizing Prefect"
    )

    parser.add_argument(
        "-d",
        "--download",
        action="store_true",
        help="Whether or not to download training set from the GCP bucket",
    )

    parser.add_argument(
        "-xgb",
        "--xgboost",
        action="store_true",
        help="Only run the XGBoost model rather than any other model",
    )

    args = parser.parse_args()

    with Flow("First ETL Spaceship Flow") as f:
        df = load_dataset(args.download)
        df_dict = transform_dataset(df, args.xgboost)
        results_models = create_model_output(df_dict, args.xgboost)
        save_dataframe(results_models)

    f.run()

    # GitHub location: https://github.com/PrefectHQ/prefect/blob/6a69b3c618de71fd0ef154b14ff408fe9fb3af2d/src/prefect/core/flow.py#L1310
    # f.visualize(filename="/app/reports/figures/visualize_etl")  # tracking dependencies;
