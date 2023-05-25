import os, sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

class Split():
    def __init__(self) -> None:
        self.raw_path = os.path.join('data/raw/housing')
        self.output_path = os.path.join('data/interim/')
        self.output_files = ["X_train.csv", "X_valid.csv", "y_train.csv", "y_valid.csv"]
        self.raw_data = pd.read_csv( self.raw_path.joinpath('housing.csv') )
        self.check()

    def check(self):
        
        if not Path(self.raw_path).exists():
            raise logging.warning( BaseException( FileNotFoundError ))

        elif set(os.listdir(self.output_path)) >= set(self.output_files):
            logging.warning('Files exist!')
            raise  FileExistsError()

        

    def default(self):
        logging.info('Splitting the shuffled data into test (%80) & train (%20) with "sklearn.train_test_split"' )
        
        #split the data to valid & train
        train, valid = train_test_split(self.raw_data, train_size=0.8, test_size=0.2, random_state=0)

        #split features (X) and target (y)
        X_train = train[ [col for col in train.columns if col != 'median_house_value'] ]
        y_train = train['median_house_value']
        X_valid = valid[ [col for col in valid.columns if col != 'median_house_value'] ]
        y_valid = valid['median_house_value']

        self.write(X_train, X_valid, y_train, y_valid)
        

    def strat(self):
        logging.info('Splitting the data stratified by target  into test (%80) & train (%20) with "sklearn.StratifiedShuffleSplit"' )

        self.raw_data["income_cat"] = pd.cut(self.raw_data["median_income"], bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])
        train, valid = train_test_split(self.raw_data, test_size=0.2, stratify=self.raw_data['income_cat'], random_state=0)

        train.drop(columns=['income_cat'], axis=1, inplace=True)
        valid.drop(columns=['income_cat'], axis=1, inplace=True)

        X_train = train[ [col for col in train.columns if col != 'median_house_value'] ]
        y_train = train['median_house_value']
        X_valid = valid[ [col for col in valid.columns if col != 'median_house_value'] ]
        y_valid = valid['median_house_value']

        self.write(X_train, X_valid, y_train, y_valid)

    def write(self, X_train, X_valid, y_train, y_valid ):
        logger = logging.getLogger(__file__)
        logger.info('Saving to /data/interim as csv files with names X/y train/value' )

        pd.DataFrame.to_csv(X_train, self.output_path.joinpath('X_train.csv'), index=True)
        pd.DataFrame.to_csv(X_valid, self.output_path.joinpath('X_valid.csv'), index=True)
        pd.DataFrame.to_csv(y_train, self.output_path.joinpath('y_train.csv'), index=True)
        pd.DataFrame.to_csv(y_valid, self.output_path.joinpath('y_valid.csv'), index=True)



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    project_dir = os.getcwd()
    sys.path.append(project_dir)