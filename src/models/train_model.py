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

from src.data.read_data import split_data
from src.features.build_features import preprocessing, reshape
from xgboost import XGBRegressor

X_train,X_valid,y_train,y_valid = split_data()
#preprocessor = joblib.load( os.path.join(project_dir, 'models/first_preprocessor.joblib') )

def export_model(model, model_name):
    file = os.path.join(project_dir, f'models/{model_name}.joblib')
    joblib.dump( model ,file)



#models
lin_reg_pipe = make_pipeline( preprocessing,  LinearRegression())
lin_reg_pipe.fit(X_train,y_train)


tree_reg = make_pipeline( preprocessing, DecisionTreeRegressor(max_depth=100, random_state=0) )
tree_reg.fit(X_train, y_train)

for_reg = make_pipeline( preprocessing, RandomForestRegressor(n_estimators=100, random_state=0))
for_reg.fit(X_train, y_train)

xgb_reg = XGBRegressor(n_estimators=500, learning_Rate=0.05, early_stopping_rounds=5, random_state=0)
xgb_reg.fit( preprocessing.fit_transform(X_train), y_train,  
            eval_set=[( preprocessing.fit_transform(X_valid), y_valid)], 
            verbose=False )
xgb_reg = make_pipeline( preprocessing, xgb_reg )



if __name__ == '__main__':
    pass