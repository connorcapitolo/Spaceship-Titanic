# standard library packages
from typing import List, Dict

# third-party packages
from prefect import task, Flow

import numpy as np
import pandas as pd

from sklearn import model_selection
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# my modules
import _helper

# model parameters
seed = 7
n_splits = 5
scoring = 'accuracy'

# prepare models
models = {
    'LR': LogisticRegression(),
    'KNN': KNeighborsClassifier(),
    'RF': RandomForestClassifier(),
    'SVM': SVC()
}

# extract
@task
def load_dataset(path_name:str = 'data/train.csv') -> pd.DataFrame:
    df = pd.read_csv(path_name)
    return df

# transform
@task
def transform_dataset(df: pd.DataFrame) -> Dict:
    
    col_to_conv = ["CryoSleep", "Transported", "VIP"]
    for col in col_to_conv:
        df[col] = 1 * df[col] # converting from True-False to 1-0
        if col != "Transported":
            df[col] = df[col].astype("Int64")
        else:
            df[col] = df[col].astype(int)
    
    numeric_columns = df.select_dtypes(include=['number'])
    
    numeric_columns_remove_na = _helper.remove_rows_with_nulls(numeric_columns)
    
    normalize_df_remove_na_numeric = preprocessing.Normalizer(
    ).fit_transform(numeric_columns_remove_na.drop('Transported', axis=1))
    
    df_dict = {
        'X': normalize_df_remove_na_numeric,
        'y': numeric_columns_remove_na['Transported']
    }

    return df_dict

# transform
@task
def create_model_output(df: Dict) -> List:
    '''# saving results as a list of lists
    
    outer list is for each model, and inner list is the accuracy scores for each of the 5 folds)
    
    '''
    results = []
    for name, model in models.items():
        kfold = model_selection.KFold(
            n_splits=n_splits, shuffle=True, random_state=seed)
        cv_results = model_selection.cross_val_score(model, df['X'],
                                                     df['y'], cv=kfold, scoring=scoring)
        results.append(cv_results)
        print(f'{name}: {cv_results.mean():.4f}, {cv_results.std():.4f}')

    return results

# load
@task
def save_dataframe(results: List, column: str = 'normalize_remove_na'):
    
    # dictionary where df_name is key, and list of results is value
    df_model_means = {}

    # full_result is the a list of lists (outer list is for each model, and inner list is the accuracy scores for each of the 5 folds
    get_model_means = []
    for cv_results in results:
        get_model_means.append(np.round(cv_results.mean(), 4))
    
    df_model_means[column] = get_model_means

    model_results = pd.DataFrame(df_model_means, index=models.keys())

    return model_results.to_csv('prefect_output.csv')

with Flow("First ETL Spaceship Flow") as f:

    df = load_dataset()
    df_dict = transform_dataset(df)
    results = create_model_output(df_dict)
    save_dataframe(results)

f.run()
# f.visualize() # tracking dependencies
