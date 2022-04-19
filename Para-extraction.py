
import math
import os
import numpy as np
import pandas as pd
import pywt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def ev(_x, _y):
    RMSE = math.sqrt(mean_squared_error(_x, _y))
    MAE = mean_absolute_error(_x, _y)
    # MAPE = np.mean(np.abs((_x - _y) / _x))
    R2 = r2_score(_x, _y)
    Re = np.array([RMSE, MAE, R2])
    return Re


def statis(_x):
    _x1 = np.mean(_x)
    _x2 = np.std(_x)
    _x3 = np.max(_x)
    _x4 = np.min(_x)
    _x7 = np.percentile(_x, 95)
    _xx = pd.DataFrame([[_x1, _x2, _x3, _x4, _x7]])
    return _xx


# 换道前的数据
path1 = r'E:\PythonProjects1\Lane change prediciton\lane-change-prediction-NGSIM'
data1 = pd.read_csv(os.path.join(path1, 'car-following-prediction(surround).csv'), parse_dates=True, encoding='gbk')
data2 = pd.read_csv(os.path.join(path1, 'lane-change-prediction(surround).csv'), parse_dates=True, encoding='gbk')
lk = pd.read_csv(os.path.join(path1, 'car-following-QKV(selection).csv'), parse_dates=True, encoding='gbk')
lc = pd.read_csv(os.path.join(path1, 'lane-change-QKV.csv'), parse_dates=True, encoding='gbk')


# 提取跟驰参数
m1 = pd.DataFrame()

data = data1
xx = lk
for i in range(np.size(xx, 0)):
# for i in range(5):
    print(i)
    xdata = data[(data[['id']].values == xx.at[i, 'id']) & (data[['frame']].values >= xx.at[i, 'startframe'])
                  & (data[['frame']].values <= xx.at[i, 'endframe']-5)]
    xdata = xdata.reset_index()
    xxdata = xdata[['speedx', 'speedy', 'accx', 'accy', 'yaw', 'speedr', 'gap']]

    xxdata['x1'] = pd.DataFrame(xdata[['localy-left-pre']].values - xdata[['localy']].values)
    xxdata['r1'] = pd.DataFrame(xdata[['speedy-left-pre']].values - xdata[['speedy']].values)
    xxdata['x2'] = pd.DataFrame(xdata[['localy']].values - xdata[['localy-left-foll']].values)
    xxdata['r2'] = pd.DataFrame(xdata[['speedy']].values - xdata[['speedy-left-foll']].values)
    xxdata['x3'] = pd.DataFrame(xdata[['localy-right-pre']].values - xdata[['localy']].values)
    xxdata['r3'] = pd.DataFrame(xdata[['speedy-right-pre']].values - xdata[['speedy']].values)
    xxdata['x4'] = pd.DataFrame(xdata[['localy']].values - xdata[['localy-right-foll']].values)
    xxdata['r4'] = pd.DataFrame(xdata[['speedy']].values - xdata[['speedy-right-foll']].values)

    # 找到id不存在的情况设置为0
    a1 = np.where(xdata[['id-left-pre']].values == 0)
    a2 = np.where(xdata[['id-left-foll']].values == 0)
    a3 = np.where(xdata[['id-right-pre']].values == 0)
    a4 = np.where(xdata[['id-right-foll']].values == 0)

    xxdata.loc[a1[0], 'x1'] = 0
    xxdata.loc[a1[0], 'r1'] = 0
    xxdata.loc[a2[0], 'x2'] = 0
    xxdata.loc[a2[0], 'r2'] = 0
    xxdata.loc[a3[0], 'x3'] = 0
    xxdata.loc[a3[0], 'r3'] = 0
    xxdata.loc[a4[0], 'x4'] = 0
    xxdata.loc[a4[0], 'r4'] = 0

    xx1 = pd.DataFrame()
    names = ['speedx', 'speedy', 'accx', 'accy', 'yaw', 'speedr', 'gap', 'x1', 'r1', 'x2', 'r2', 'x3', 'r3', 'x4', 'r4']

    for name in names:
        xxxdata = xxdata[name]
        x1 = statis(xxxdata)
        x1.columns = [name, name, name, name, name]
        xx1 = pd.concat([xx1, x1], axis=1)

    m1 = pd.concat([m1, xx1], axis=0)

m1 = m1.reset_index(drop=True)
mdata = xx[['class', 'Traffic', 'maneuver']]  #
m1 = pd.concat([m1, mdata], axis=1)

m1.to_csv(os.path.join(path1, 'car-following-intention-st.csv'), index=False, sep=',')
