import os,sys,joblib
project_dir = os.getcwd()
sys.path.append(project_dir)

from src.data.read_data import split_data
from src.models.train_model import tree_reg
from sklearn.metrics import mean_squared_error, explained_variance_score

X_train,X_valid,y_train,y_valid = split_data()

model = tree_reg
predictions = model.predict(X_valid)
#print(y_valid.shape)
#print(predictions.shape)

mean_squared_error(y_valid, predictions )

#print("Mean squared Error:", res)

