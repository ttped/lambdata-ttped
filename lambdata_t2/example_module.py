import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


class FastModel():
    """A class to help speed up building models

    Args:
        dataframe: the dataframe you wish to analysis on
        target: the feature you wish to predict from dataframe
        features: the features you wish to train the data on
    Returns:
        Nothing
    """
    df = None
    target = None
    features_arr = None
    X = None
    y = None
    X_train = None
    X_val = None
    y_train = None
    y_val = None
    model = None

    def __init__(self, dataframe, target, features_arr):
        '''Initializes dataframe and creates a train/test split on target/features'''
        self.df = dataframe
        self.target = target
        self.features_arr = features_arr
        self.X = self.df[features_arr]
        self.y = self.df[target]
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X,
                                                                              self.y,
                                                                              test_size=.2,
                                                                              random_state=42)
        pass


    def tt_split(self, target, features_arr):
        '''Saves a train test split to AutoDataScience
        Args:
            target: The target feature you wish to predict. Will be assigned X variables.
            features: The features you wish to train on. Will be assigned Y variables.
        Returns:
            X_train: Training features
            X_val: Validation features
            y_train: Training target
            y_val: Validation target
        '''
        self.X = self.df[features_arr]
        self.y = self.df[target]
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X,
                                                                              self.y,
                                                                              test_size=.2,
                                                                              random_state=42)
        pass


    def get_train_test(self):
        '''Returns X_train, X_val, y_train, y_val'''
        return self.X_train, self.X_val, self.y_train, self.y_val


    def set_model(self, model):
        '''
        Saves the model then applies a fit

        Args:
            model: The model you wish to use for the data
        '''
        self.model = model
        self.model.fit(self.X_train, self.y_train)
        pass

    def print_model_scores(self):
        '''Returns various performance metrics on the model'''
        print('Training Scores:', self.model.score(self.X_train, self.y_train))
        print('Validation Scores:', self.model.score(self.X_val, self.y_val))
        print('Training Mean Absolute Error:', mean_absolute_error(self.model.predict(self.X_train), self.y_train))
        print('Validation Mean Absolute Error:', mean_absolute_error(self.model.predict(self.X_val), self.y_val))

    
def auto_split(df, target, features_arr, test_size=.2):
    '''Takes in a dateframe and returns a train test split
        with target variable and feature variables'''
    X = df[features_arr]
    y = df[target]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_val, y_train, y_val


def month_day_year_split(datetime_column):
    '''Takes a column in datetime format and returns month,
        day, and year columns'''
    month = datetime_column.dt.month
    day = datetime_column.dt.day
    year = datetime_column.dt.year
    return month, day, year


if __name__ == '__main__':
    pass
