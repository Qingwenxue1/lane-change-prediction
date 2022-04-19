import os
import time
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import xgboost
from sklearn import neural_network, svm, ensemble
from imblearn.over_sampling import SMOTEN
from imblearn.pipeline import make_pipeline
from sklearn.metrics import precision_score, accuracy_score, roc_auc_score, recall_score, f1_score, \
    classification_report,  mean_squared_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import normalize

warnings.filterwarnings("ignore")


def _xgboost():
    model = xgboost.XGBClassifier(eval_metric='mlogloss')
    random_grid = {
        'xgbclassifier__n_estimators': [50, 100, 200, 300, 500],
        'xgbclassifier__learning_rate': [0.01, 0.1, 0.2, 0.3],
        'xgbclassifier__max_depth': [3, 6, 9],
        'xgbclassifier__min_child_weight': [0, 2, 5, 10, 20],
        'xgbclassifier__subsample': [0.6, 0.7, 0.8, 0.9],
        'xgbclassifier__colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9],
        'xgbclassifier__reg_alpha': [0, 0.25, 0.5, 0.75, 1],
        'xgbclassifier__reg_lambda': [0, 0.2, 0.4, 0.6, 0.8, 1]
    }
    return model, random_grid

def adaboost():
    model = ensemble.AdaBoostClassifier()
    params = {
                # 'adaboostclassifier__num_leaves': (10, 50),
              # 'adaboostclassifier__min_data_in_leaf': (10, 50),
              'adaboostclassifier__max_depth': (5, 20),
              'adaboostclassifier__learning_rate': (0.001, 0.1),
              'adaboostclassifier__bagging_fraction': (0.5, 1),
              'adaboostclassifier__feature_fraction': (0.5, 1),
              'adaboostclassifier__lambda_l1': (0, 10),
              'adaboostclassifier__lambda_l2': (0, 10),
              # 'adaboostclassifier__min_gain_to_split': (0.001, 0.1),
              # 'adaboostclassifier__min_child_weight': (0.001, 100)
    }
    return model, params


def SVM():
    model = svm.SVC(probability=True)
    random_grid = {
        'svc__kernel': ['rbf'],
        'svc__gamma': [1e-3, 1e-4],
        'svc__C': [1, 10, 100, 1000],
    }
    return model, random_grid


def run_model(_model, _X_test, _y_test):
    t0 = time.time()
    _y_pred = _model.predict(_X_test)
    _y_pred_prob = _model.predict_proba(_X_test)
    _y_pred_prob = _y_pred_prob[:, 1]
    time_taken = time.time() - t0
    print("Time taken = {}".format(time_taken))
    return _model, _y_pred, _y_pred_prob


def evaluation_model(_model, _X_test, _y_test, _y_pred, _y_pred_prob):
    accuracy = accuracy_score(_y_test, _y_pred)
    precision = precision_score(_y_test, _y_pred, average=None)
    recall = recall_score(_y_test, _y_pred, average=None)
    f1 = f1_score(_y_test, _y_pred, average='weighted')
    mse = mean_squared_error(y_test, y_pred)

    plt.rcParams["font.family"] = "Times New Roman"
    print("Accuracy = {}".format(accuracy))
    print("Precision = {}".format(precision))
    print("Recall = {}".format(recall))
    print("F1_score = {}".format(f1))
    print("mse = {}".format(mse))
    print(classification_report(_y_test, _y_pred, digits=5))

    return accuracy


## 输入变量
path1 = r'E:\PythonProjects1\Lane change prediciton\lane-change-prediction-NGSIM'
data1 = pd.read_csv(os.path.join(path1, 'car-following-intention-st.csv'), parse_dates=True, encoding='gbk')
data2 = pd.read_csv(os.path.join(path1, 'lane-change-intention-st.csv'), parse_dates=True, encoding='gbk')
data = pd.concat([data1, data2], axis=0)


data.loc[data['Traffic'] == 'A', 'Traffic'] = 0
data.loc[data['Traffic'] == 'B', 'Traffic'] = 1
data.loc[data['Traffic'] == 'C', 'Traffic'] = 2
data.loc[data['Traffic'] == 'D', 'Traffic'] = 3
data.loc[data['Traffic'] == 'E', 'Traffic'] = 4


dx = data.drop(['maneuver', 'class', 'Traffic'], axis=1).values
# dx = data.drop(['maneuver'], axis=1).values
dy = data['Traffic']



# 提取输入变量，不加入情景因素
xdata = normalize(dx, axis=0, norm='max')

X_train, X_test, y_train, y_test = train_test_split(xdata, dy.astype(int), test_size=0.2, random_state=1)


# model, random_grid = _xgboost()
# model, random_grid = adaboost()
# model, random_grid = svm()


# #调整参数，建立pipline，防止数据泄露产生的模型偏移
# sm = SMOTEN(random_state=0)  # ##无法用数字表达的特征就是分类特征，其他事数值特征
# pipline = make_pipeline(sm, model)
# imodel = RandomizedSearchCV(estimator=pipline, param_distributions=random_grid, n_iter=10, cv=5, verbose=2,
#                                   random_state=42, n_jobs=-1)


# imodel = xgboost.XGBClassifier(eval_metric='mlogloss')
# imodel = ensemble.AdaBoostClassifier()
imodel = svm.SVC(probability=True, kernel='linear')

imodel.fit(X_train, y_train)
imodel, y_pred, y_pred_prob = run_model(imodel, X_test, y_test)
accuracy_ml = evaluation_model(imodel, X_test, y_test, y_pred, y_pred_prob)



