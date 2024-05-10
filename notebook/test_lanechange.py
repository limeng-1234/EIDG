import torch
import pandas as pd
import os
import numpy as np
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
# 获取当前文件所在目录的绝对路径
current_directory = os.getcwd()
# 拼接文件夹路径
folder_path = os.path.join(current_directory, "..", "data")
# 拼接文件路径
data = np.load(os.path.join(folder_path, "lane_change_data.npz"))
x_train = data['x_train']
x_test = data['x_test']
y_train = data['y_train']
y_test = data['y_test']


other_params = {'learning_rate': 0.1, 'n_estimators': 60, 'max_depth': 10, 'min_child_weight': 0.2, 'seed': 0,
                'subsample': 0.9, 'colsample_bytree': 0.8, 'gamma': 0.01, 'reg_alpha': 0, 'reg_lambda': 0.5}
model = xgb.XGBClassifier(**other_params)
model.fit(x_train,
          y_train)
y_hat = model.predict(x_test)
Xgbc_score_a = accuracy_score(y_test, y_hat)
Xgbc_score_a

from EIDG import EIDG
# from EIDG1 import EIDG
def model_modity(input):
    return model.predict_proba(input)[:,0]
eidg_instance = EIDG(x_baselines=x_test, model = model.predict_proba ,steps=100, h=0.1, m=10, pos=True)
explainer_xgb_eidg_values = eidg_instance.integrated_gradients(x_test)