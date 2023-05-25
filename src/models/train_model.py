import sys,os
project_dir = os.getcwd()
sys.path.append(project_dir)

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from src.features.build_features import preprocessing
from src.data.read_data import X_train,X_valid,y_train,y_valid

class Train():
    def __init__(self,) -> None:
        pass

    def linear_regression(self) -> LinearRegression():
        lin_reg_pipe = make_pipeline( self.import_preprocessor(), LinearRegression())
        lin_reg_pipe.fit(X_train,y_train)
        return lin_reg_pipe
    
    def tree_regressor(self) -> DecisionTreeRegressor:
        tree_reg = make_pipeline( self.import_preprocessor(), DecisionTreeRegressor(max_depth=100 random_state=0))
        tree_reg.fit(X_train, y_train)
        return tree_reg
    
    def import_preprocessor(self):
        file = os.path.join(project_dir, 'models/first_preprocessor.joblib')
        return joblib.load(file)
    
    def export_model(model):
        file = os.path.join(project_dir, 'models/first_model.joblib')
        joblib.dump( model ,file)



if __name__ == '__main__':
    pass







