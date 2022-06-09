"""An introductory ETL (extract-transform-load) pipeline utilizing Prefect

The module loads in the Spaceship Titanic dataframe, performs the necessary preprocessing to get it in a format that can be used for modeling, trains model(s), and then saves the accuracy score

    Typical usage example from Terminal in JupyterLab:

    $ python prefect_etl.py
"""

# standard library packages
from typing import List, Dict, Tuple

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

# my modules
import _helper

# model parameters
seed = 7
n_splits = 5
random_state = 109
test_size = 0.2
scoring = 'accuracy'

# prepare models
models = []
models = []
models.append(('LR', LogisticRegression())) 
models.append(('LDA', LinearDiscriminantAnalysis())) 
models.append(('KNN', KNeighborsClassifier())) 
models.append(('SVM', SVC()))
models.append(('RF', RandomForestClassifier()))
models.append(('GBM',GradientBoostingClassifier()))
models.append(('ABM',AdaBoostClassifier()))

# extract
@task
def load_dataset(path_name:str = 'data/train.csv') -> pd.DataFrame:
    """Loads in the Spacheship Titanic dataset used to the train the model
    
    Args:
        path_name: path for obtaining the model
        
    Returns:
        The Pandas DataFrame that is the training data
    """
    df = pd.read_csv(path_name)
    return df

# transform
@task
def transform_dataset(df: pd.DataFrame) -> Tuple:
    """Performs preprocessing on the dataset to prepare it for modeling

    Takes in the original dataframe, performs type conversion, variable selection (only using numeric for now), handling of missing values, and train-test split to avoid data leakage

    Args:
        df: the Spaceship Titanic training dataset

    Returns:
        df_dict: a dictionary comprised of the feature columns that will be used for training the models (stored as a value corresponding the 'X' key), and the "Transported" labels (stored as a value within correpsponding to the 'Y' key)
    """
    
    col_to_conv = ["CryoSleep", "Transported", "VIP"]
    for col in col_to_conv:
        df[col] = 1 * df[col] # converting from True-False to 1-0
        if col != "Transported":
            df[col] = df[col].astype("Int64")
        else:
            df[col] = df[col].astype(int)
    
    numeric_columns = df.select_dtypes(include=['number'])
    
    numeric_columns_remove_na = _helper.remove_rows_with_nulls(numeric_columns)
    
    # perform train-test split to save test data so that it can be used later (typically want to save, but not doing at this point)
    X_train, X_test, y_train, y_test = train_test_split(numeric_columns_remove_na.drop("Transported", axis=1),
                                                        numeric_columns_remove_na["Transported"], test_size=test_size,
                                                        random_state=random_state)

    return (X_train, X_test, y_train, y_test)

# transform
@task
def create_model_output(df: Tuple) -> List:
    '''Perform the model training and saving the results

    In order to perform the k-fold cross validation while avoiding data leakage, need to perform the normalization for each particular fold (fit and normalize the train step, and apply normalization to test step); this is the use of scikit-learn's Pipeline module. After performing the k-fold cross-validation, each result is saved as an element in a list; therefore, we have an outer list comprised of each trained model, and an inner list that contains k accuracy score corresponding to each of the k-folds

    Args:
        df: dictionary that contains the result from the transform_dataset Prefect task

    Returns:
        A list of lists for each of the results, as outlined in the description above
    '''
    X_train, X_test, y_train, y_test = df
    
    results= []
    for name, model in models:
        '''
        To utilize scikit-learn's Pipeline, first element is the name of the step (a string) and second is configured object of the step, such as a transform or a model. The model is only supported as the final step, although we can have as many transforms as we like in the sequence.
        '''
        
        # define the pipeline
        steps = list()
        steps.append(('scaler', preprocessing.Normalizer()))
        steps.append(('model', model))
        pipeline = Pipeline(steps=steps)

        # define the evaluation procedure
        kfold = model_selection.KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        cv_results = model_selection.cross_val_score(pipeline, X_train,
                                                     y_train, cv=kfold, scoring=scoring, n_jobs=-1)
        # report performance
        print(f'{name}: {cv_results.mean():.4f}, {cv_results.std():.4f}')
        # add scores to the list
        results.append(cv_results)

    return results

# load
@task
def save_dataframe(results: List, column: str = 'normalize_remove_na'):
    '''Find the mean from the k-fold cross validation for each model, and saves it to a csv file

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
    '''
    
    # dictionary where df_name is key, and list of results is value
    df_model_means = {}

    # full_result is the a list of lists (outer list is for each model, and inner list is the accuracy scores for each of the 5 folds
    get_model_means = []
    for cv_results in results:
        get_model_means.append(np.round(cv_results.mean(), 4))
    
    df_model_means[column] = get_model_means

    model_results = pd.DataFrame(df_model_means, index=[get_model_names[0] for get_model_names in models])

    return model_results.to_csv('prefect_output.csv', index_label='Models')

with Flow("First ETL Spaceship Flow") as f:

    df = load_dataset()
    df_dict = transform_dataset(df)
    results = create_model_output(df_dict)
    save_dataframe(results)

if __name__=='__main__':
    f.run()

    # GitHub location: https://github.com/PrefectHQ/prefect/blob/6a69b3c618de71fd0ef154b14ff408fe9fb3af2d/src/prefect/core/flow.py#L1310
    # f.visualize(filename='visualize_etl') # tracking dependencies;


