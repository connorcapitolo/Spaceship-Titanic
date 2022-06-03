import pandas as pd


def remove_rows_with_nulls(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna()

def fill_numeric_missing_values(df: pd.DataFrame, /, numeric:str = 'median', non_numeric:str = 'top_frequency') -> pd.DataFrame:
    numeric_columns = df.select_dtypes(include=['float64','Int64'])
    assert numeric_columns.shape[1] == 9
    
    mean_median_cols = {}
    for col in numeric_columns.columns:
        
        if numeric == 'median':
            mean_median_cols[col] = numeric_columns[col].median()
        else:
            mean_median_cols[col] = numeric_columns[col].mean()
            
    return numeric_columns.fillna(value=mean_median_cols)