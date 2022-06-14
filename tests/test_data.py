#!/usr/bin/env python

# third-party modules
import pandas as pd

# my modules
from spaceship_titanic.features import _helper


def test_remove_rows_nulls():
    df = pd.read_csv("/app/data/raw/train.csv")
    df = _helper.remove_rows_with_nulls(df)
    assert df.isnull().sum().sum() == 0


def test_fill_nulls():
    df = pd.read_csv("/app/data/raw/train.csv")
    df = _helper.fill_numeric_missing_values(df)
    assert df.isnull().sum().sum() == 0
