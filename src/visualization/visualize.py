import os,sys,joblib
project_dir = os.getcwd()
sys.path.append(project_dir)

from src.data.read_data import split_data
from src.models.predict_model import predictions

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


X_train,X_valid,y_train,y_valid = split_data()


error = y_valid['SalePrice'] - predictions

#sns.residplot(x=predictions, y=error )
plt.scatter(x=predictions, y=error)

plt.show()

