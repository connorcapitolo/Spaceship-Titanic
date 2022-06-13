#!/usr/bin/env python

# third-party modules
import pandas as pd

# my modules
import src.spaceship_titanic.features._helper as _helper

# from /app/ directory, run in Terminal either of the two options ...
# $ python -m pytest .
# $ python -m pytest tests
def test_remove_rows_nulls():
    df = pd.read_csv("/app/data/raw/train.csv")
    df = _helper.remove_rows_with_nulls(df)
    assert df.isnull().sum().sum() == 0


def test_fill_nulls():
    df = pd.read_csv("/app/data/raw/train.csv")
    df = _helper.fill_numeric_missing_values(df)
    assert df.isnull().sum().sum() == 0
