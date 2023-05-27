import sys, os, joblib
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler
from sklearn.compose import ColumnTransformer, make_column_selector, make_column_transformer
from sklearn.metrics.pairwise import rbf_kernel

from sklearn.cluster import KMeans

obvious_features = ['1stFlrSF', '2ndFlrSF', 'BsmtFinSF1', 'Fireplaces', 'FullBath', 'GarageArea', 
                    'GarageCars', 'GrLivArea', 'LotFrontage', 'MasVnrArea', 'OpenPorchSF', 
                    'SalePrice', 'TotRmsAbvGrd', 'TotalBsmtSF', 'WoodDeckSF']

categoric_columns = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour',
       'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1',
       'Condition2', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond',
       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
       'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond',
       'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC',
       'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu',
       'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive',
       'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']


class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state
    
    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, n_init='auto', random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self # always return self!
    
    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)
    
    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]


def column_ratio(X=None):
    return X[:, [0]] / X[:, [1]]

def ratio_name(function_transformer, feature_names_in):
    return ["ratio"] # feature names out

def ratio_pipeline():
    make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler(),
        memory=os.path.join(os.getcwd(), 'models') 
        )

def conv_column(X):
    return X[:, 0] == X[:, 0].astype('category')
conv_dtype_to_cat = FunctionTransformer(conv_column, feature_names_out='one-to-one')

log_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    FunctionTransformer(np.log, feature_names_out="one-to-one"),
    StandardScaler(),
    memory=os.path.join(os.getcwd(), 'models')
    )


cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=0)

cat_pipeline = make_pipeline( SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore"))
default_num_pipeline = make_pipeline( SimpleImputer(strategy="median"), StandardScaler(), memory=os.path.join(os.getcwd(), 'models'))

"""
("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
("people_per_house", ratio_pipeline(), ["population", "households"]),
("log", log_pipeline, ["total_bedrooms", "total_rooms", "population","households", "median_income"]),
("geo", cluster_simil, ["latitude", "longitude"]),
"""


preprocessing = ColumnTransformer(transformers=[

    #divide two cols with each other
    #('average_room_size', ratio_pipeline(), ['GrLivArea', 'TotRmsAbvGrd']),
    # impute & onehot encode categorical columns
    ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
    ],
    remainder=default_num_pipeline)


def export_preprocess(preprocessing=preprocessing):
    project_dir = os.getcwd()
    output_file = os.path.join(project_dir, 'models/first_preprocessor.joblib' )
    joblib.dump(preprocessing, output_file)

export_preprocess()

if __name__ == '__main__':
    pass

