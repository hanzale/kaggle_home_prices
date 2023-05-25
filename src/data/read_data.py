import os,sys
import pandas as pd

project_dir = os.getcwd()
sys.path.append(project_dir)
fp_data = os.path.join(project_dir, 'data/interim')

X_train = pd.read_csv(os.path.join(fp_data,'X_train.csv'), index_col='Unnamed: 0')
X_valid = pd.read_csv(os.path.join(fp_data,'X_valid.csv'), index_col='Unnamed: 0')
y_train = pd.read_csv(os.path.join(fp_data,'y_train.csv'), index_col='Unnamed: 0')
y_valid = pd.read_csv(os.path.join(fp_data,'y_valid.csv'), index_col='Unnamed: 0')

