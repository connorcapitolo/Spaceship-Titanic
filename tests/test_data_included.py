#!/usr/bin/env python

# python standard library
import os

# third-party modules
import pandas as pd

# my modules
from spaceship_titanic.features import _helper

train_relative_path = "../data/raw/train.csv"


def test_remove_rows_nulls():
    df = pd.read_csv(os.path.join(os.getcwd(), train_relative_path))
    df = _helper.remove_rows_with_nulls(df)
    assert df.isnull().sum().sum() == 0


def test_fill_nulls():
    df = pd.read_csv(os.path.join(os.getcwd(), train_relative_path))
    df = _helper.fill_numeric_missing_values(df)
    assert df.isnull().sum().sum() == 0
