import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def train_test_split(df, target, features_arr, test_size=.2):
    '''Takes in a dateframe and returns a train test split with target variable and feature variables'''
    X = df[features_arr]
    y = df[target]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_val, y_train, y_val

def month_day_year_split(datetime_column):
    '''Takes a column in datetime format and returns month, day, and year columns'''
    month = datetime_column.dt.month
    day = datetime_column.dt.day
    year = datetime_column.dt.year
    return month, day, year

#def train_test_split_date(df, target, features_arr, date_column_str, to_date):
#    X = df[features_arr]
#    y = df[target]
#
#    mask = df[df[date_column_str] <= to_date]
#    mask_not = ~mask
#
#    X_train = X[mask]
#    y_train = y[mask]
#
#    X_val = X[mask_not]
#    y_val = y[mask_not]
#
#    return X_train, X_val, y_train, y_val
