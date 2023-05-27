import sys,os
project_dir = os.getcwd()
sys.path.append(project_dir)

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
#from src.features.build_features import preprocessing
from src.data.read_data import split_data
from xgboost import XGBRegressor

X_train,X_valid,y_train,y_valid = split_data()

class Train():
    def __init__(self,) -> None:
        self.preprocessor = self.import_preprocessor()

    def import_preprocessor(self):
        file = os.path.join(project_dir, 'models/first_preprocessor.joblib')
        return joblib.load(file)
    
    def export_model(model, model_name):
        file = os.path.join(project_dir, f'models/{model_name}.joblib')
        joblib.dump( model ,file)

    #models
    def linear_regression(self) -> LinearRegression():
        lin_reg_pipe = make_pipeline( self.import_preprocessor(),  LinearRegression())
        lin_reg_pipe.fit(X_train,y_train)
        return lin_reg_pipe
    
    def tree_regressor(self) -> DecisionTreeRegressor():
        tree_reg = make_pipeline( self.import_preprocessor(), DecisionTreeRegressor(max_depth=100, random_state=0) )
        tree_reg.fit(X_train, y_train)
        return tree_reg
    
    def random_forest(self) -> RandomForestRegressor():
        for_reg = make_pipeline( self.preprocessor, RandomForestRegressor(n_estimators=100, random_state=0))
        for_reg.fit(X_train, y_train)
        return for_reg

    def xgb_regressor(self):
        select_model = XGBRegressor(n_estimators=500, learning_Rate=0.05)
        select_model.fit(self.preprocessor.fit_transform(X_train), y_train, early_stopping_rounds=5, eval_set=[(self.preprocessor.fit_transform(X_valid), y_valid)], verbose=False )
        xgb_reg = make_pipeline( self.preprocessor, select_model )
        #xgb_reg.fit(X_train, y_train, xgbregressor__early_stopping_rounds=5, xgbregressor__eval_set=[(X_valid, y_valid)], xgbregressor__verbose=False )
        #xgb_reg.fit(X_train, y_train, )
        return xgb_reg
    
x = Train().linear_regression()
Train.export_model(x, 'linear_regression')


if __name__ == '__main__':
    pass