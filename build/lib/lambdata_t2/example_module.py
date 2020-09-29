import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


class FastModel():
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
        '''Saves a train test split to AutoDataScience'''
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
        '''Sets the model'''
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
