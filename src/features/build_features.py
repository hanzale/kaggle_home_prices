import sys, os, joblib
project_dir = os.getcwd()
sys.path.append(project_dir)

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler
from sklearn.compose import ColumnTransformer, make_column_selector, make_column_transformer

#'OverallQual', 'OverallCond',
features = [ 'YearBuilt', 'GrLivArea' ]

impute = Pipeline( steps=[
    ('imputer', SimpleImputer(strategy="median"))
    ],
    memory=os.path.join(os.getcwd(), 'models')
    )

preprocessing = ColumnTransformer(transformers=[
    ('impute&scale', impute, features)
   ],
    remainder='drop')

def reshape(X):
    return np.reshape(X, (-1,1))
reshape = FunctionTransformer(reshape)

def export_preprocess(preprocessing=preprocessing):
    output_file = os.path.join(project_dir, 'models/first_preprocessor.joblib' )
    joblib.dump(preprocessing, output_file)


if __name__ == '__main__':
    pass

