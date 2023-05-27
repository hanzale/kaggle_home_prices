import os, sys
project_dir = os.getcwd()
sys.path.append(project_dir)

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from src.data.read_data import raw_data

raw_path = os.path.join(os.getcwd(), 'data/raw/test.csv')
output_path = os.path.join(os.getcwd(), 'data/interim/')
output_files = ["X_train.csv", "X_valid.csv", "y_train.csv", "y_valid.csv"]
raw_data = raw_data()

def split(self):
    """"""

    bins=[raw_data['SalePrice'].min()-1, raw_data['SalePrice'].quantile(0.25), raw_data['SalePrice'].quantile(0.5), raw_data['SalePrice'].quantile(0.75), raw_data['SalePrice'].max()]
    raw_data["price_cat"] = pd.cut(raw_data["SalePrice"], bins=bins, labels=[0, 1, 2, 3])
    train, valid = train_test_split(raw_data, test_size=0.2, stratify=raw_data['price_cat'], random_state=0)

    train.drop(columns=['price_cat'], axis=1, inplace=True)
    valid.drop(columns=['price_cat'], axis=1, inplace=True)

    X_train = train[ [col for col in train.columns if col != 'SalePrice'] ]
    y_train = train['SalePrice']
    X_valid = valid[ [col for col in valid.columns if col != 'SalePrice'] ]
    y_valid = valid['SalePrice']

    pd.DataFrame.to_csv(X_train, os.path.join(output_path,'X_train.csv'), index=True)
    pd.DataFrame.to_csv(X_valid, os.path.join(output_path,'X_valid.csv'), index=True)
    pd.DataFrame.to_csv(y_train, os.path.join(output_path,'y_train.csv'), index=True)
    pd.DataFrame.to_csv(y_valid, os.path.join(output_path,'y_valid.csv'), index=True)




if __name__ == '__main__':
    pass
