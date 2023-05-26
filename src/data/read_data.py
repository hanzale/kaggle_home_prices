import os,sys
import pandas as pd


def raw_data():
    project_dir = os.getcwd()
    fp_raw_data = os.path.join(project_dir,'data/raw/train.csv')
    raw_data = pd.read_csv(fp_raw_data, index_col='Id')
    return raw_data

def split_data():
    project_dir = os.getcwd()
    fp_data = os.path.join(project_dir, 'data/interim')

    X_train = pd.read_csv(os.path.join(fp_data,'X_train.csv'), index_col='Id')
    X_valid = pd.read_csv(os.path.join(fp_data,'X_valid.csv'), index_col='Id')
    y_train = pd.read_csv(os.path.join(fp_data,'y_train.csv'), index_col='Id')
    y_valid = pd.read_csv(os.path.join(fp_data,'y_valid.csv'), index_col='Id')

    return X_train, X_valid, y_train, y_valid

def test_data():
    project_dir = os.getcwd()
    fp_raw_data = os.path.join(project_dir,'data/raw/test.csv')
    test_data = pd.read_csv(fp_raw_data, index_col='Id')
    return test_data

if __name__ == '__main__':
    pass