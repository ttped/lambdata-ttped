import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)


import unittest
from lambdata_t2 import example_module as em 
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


class TestExampleModule(unittest.TestCase):
    df = pd.DataFrame({'nums':[1,2,3,4,5,6,7,8,9,10]})
    df['squared'] = df['nums'] ** 2
    df['cubed'] = df['nums'] ** 3

    target = 'squared'
    features = df.columns.drop(target)

    model = RandomForestRegressor()

    fast = em.FastModel(df, target, features)

    def test_compare_df(self):
        self.assertTrue(self.df.equals(self.fast.df))

    def test_size(self):
        self.assertTrue(len(self.df) == 10)

    def test_check_split_data(self):
        self.assertTrue(len(self.fast.X_train) == len(self.fast.y_train))

    


if __name__ == '__main__':
    unittest.main()

    
