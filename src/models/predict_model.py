import os,sys,joblib

project_dir = os.getcwd()
sys.path.append(project_dir)

from src.models.train_model import Train
from src.data.read_data import split_data
from sklearn.metrics import mean_squared_error
from joblib import dump, load

X_train,X_valid,y_train,y_valid = split_data()

linear_model_pipeline_trained = Train().linear_regression()
#tree_reg_model_pipeline_trained = Train().tree_regressor()

#Train.export_model(model=linear_model_pipeline_trained )
model = joblib.load( os.path.join(project_dir, 'models/first_model.joblib') )

predictions = model.score(X_valid, y_valid)


#res = mean_squared_error(y_valid, predictions, squared=False)


print("Score:", predictions )
#print("Mean squared Error:", res)

