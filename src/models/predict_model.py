import os,sys,joblib

project_dir = os.getcwd()
sys.path.append(project_dir)

from src.models.train_model import Train
from src.data.read_data import split_data
from sklearn.metrics import mean_squared_error
from joblib import dump, load

X_train,X_valid,y_train,y_valid = split_data()

#linear_model_pipeline_trained = Train().linear_regression()
#tree_reg_model_pipeline_trained = Train().tree_regressor()
#xg_boost = Train().xgb_regressor()
#Train.export_model(xg_boost, 'xg_boost')
#model = xg_boost

#Train.export_model(model=linear_model_pipeline_trained, model_name='linear_model' )
model = joblib.load( os.path.join(project_dir, 'models/random_forest.joblib') )

predictions = model.predict(X_valid)
print(y_valid.shape)
#print(predictions.shape)

res = mean_squared_error(y_valid, predictions, squared=False)
print(res)


#print("Mean squared Error:", res)

