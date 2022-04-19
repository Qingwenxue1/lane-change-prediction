
import os
import pandas as pd
import numpy as np
from time import time
import scipy.io as sio

'''
# ## 数据初始化
path = r'E:\PythonProjects1\Lane change prediciton\lane-change-prediction-NGSIM'
pfile = str('E:/PythonProjects1/Lane change prediciton/lane-change-prediction-NGSIM/') + str('dataset-042.mat')


# 提取原始数据，换道样本记录以及换道轨迹
data = pd.DataFrame(sio.loadmat(pfile)['data']) # 所有原始数据
lc = pd.DataFrame(sio.loadmat(pfile)['lc'])  # 换道样本记录
ydata = pd.DataFrame(sio.loadmat(pfile)['ydata'])  # 换道样本的原始轨迹
data.columns = ['id', 'frame', 'localx', 'localy', 'length', 'width', 'speedx', 'speedy',
                'accx', 'accy', 'class', 'lane', 'precedingid', 'speedr', 'gap', 'yaw',
                'maneuver']
lc.columns = ['id', 'maneuver', 'startline', 'endline', 'startframe', 'endframe',
                             'num', 'lanechangeframe', 'class']
ydata.columns = data.columns


# 提取换道后5s的换道数据 xdata
xdata = pd.DataFrame()
for i in range(np.size(lc, 0)):
    xxdata = ydata[(ydata[['id']].values == lc.at[i, 'id']) & (ydata[['frame']].values > lc.at[i, 'lanechangeframe'])
                  & (ydata[['frame']].values <= lc.at[i, 'endframe'])]
    xdata = pd.concat([xdata, xxdata], axis=0)
xdata = xdata.reset_index(drop=True)


# 把换道前5s的参数提取出来 ddata 尤其是目标车道
ddata = pd.read_csv(os.path.join(path, 'lane-change-prediction(surround).csv'), parse_dates=True, encoding='gbk')

Preceding0 = ddata[['precedingid', 'gap', 'speedr']]  #换道前的前车
targetpreceding0 = pd.DataFrame()
targetfollowing0 = pd.DataFrame()

for i in range(np.size(lc, 0)):
    mm = ddata[(ddata[['id']].values == lc.at[i, 'id']) & (ddata[['frame']].values >= lc.at[i, 'startframe'])
                  & (ddata[['frame']].values <= lc.at[i, 'lanechangeframe'])]

    if lc.at[i, 'maneuver'] == 2:  # 右换道
        xx = mm[['id-right-pre', 'localy-right-pre', 'speedy-right-pre']]
        yy = mm[['id-right-foll', 'localy-right-foll', 'speedy-right-foll']]
        xx.columns = ['id-target-pre', 'localy-target-pre', 'speedy-target-pre']
        yy.columns = ['id-target-foll', 'localy-target-foll', 'speedy-target-foll']
    else:
        xx = mm[['id-left-pre', 'localy-left-pre', 'speedy-left-pre']]
        yy = mm[['id-left-foll', 'localy-left-foll', 'speedy-left-foll']]
        xx.columns = ['id-target-pre', 'localy-target-pre', 'speedy-target-pre']
        yy.columns = ['id-target-foll', 'localy-target-foll', 'speedy-target-foll']

    targetpreceding0 = pd.concat([targetpreceding0, xx], axis=0)
    targetfollowing0 = pd.concat([targetfollowing0, yy], axis=0)




# 把换道后的特征提取出来 提取换道后5s的换道数据 xdata
Preceding = pd.DataFrame(np.zeros((np.size(xdata, 0), 3)))  # 换道后原车道前车都设置为0
Preceding.columns = ['precedingid', 'gap', 'speedr']
targetpreceding = xdata[['precedingid']]  # 换道后目标车道前车
yy = pd.DataFrame(xdata[['gap']].values + xdata[['localy']].values + xdata[['length']].values)
zz = pd.DataFrame(xdata[['speedr']].values + xdata[['speedy']].values)
targetpreceding = pd.concat([targetpreceding, yy, zz], axis=1)
targetpreceding.columns = ['id-target-pre', 'localy-target-pre', 'speedy-target-pre']

a = np.where(targetpreceding[['id-target-pre']].values == 0)
targetpreceding.loc[a[0], 'localy-target-pre'] = 0
targetpreceding.loc[a[0], 'speedy-target-pre'] = 0


targetfollowing = pd.DataFrame()

# 提取换道后5s的换道数据 xdata
idata = xdata
dff2 = np.zeros((1, 3))
dff2 = pd.DataFrame(dff2)
dff2.columns = ['id', 'localy', 'speedy']

#  换道样本后5s  后车
for j in range(np.size(idata, 0)):
    print(j)

    dff1 = data[(data[['lane']].values == idata.at[j, 'lane']) & (data[['frame']].values == idata.at[j, 'frame'])
                & (data[['localy']].values < idata.at[j, 'localy'])]

    if np.size(dff1, 0) == 0:
        targetfollowing = pd.concat([targetfollowing, dff2], axis=0)

    else:
        distance = dff1[['localy']] - idata.at[j, 'localy']
        y1 = np.max(distance[['localy']].values)
        dff3 = dff1.loc[distance['localy'].values == y1, ['id', 'localy', 'speedy']]

        targetfollowing = pd.concat([targetfollowing, dff3], axis=0)

targetfollowing = targetfollowing.reset_index(drop=True)
targetfollowing.columns = ['id-target-foll', 'localy-target-foll', 'speedy-target-foll']




# 换道前后的数据三个车辆配对起来 ddata前5s  xdata后5s  ydata是所有的10s
Precedingxx = pd.DataFrame()
TargetPrecedingxx = pd.DataFrame()
TargetFollowingxx = pd.DataFrame()


for j in range(np.size(lc, 0)):
    yy0 = Preceding0[(ddata[['id']].values == lc.at[j, 'id']) & (ddata[['frame']].values <= lc.at[j, 'lanechangeframe'])
                     & (ddata[['frame']].values >= lc.at[j, 'startframe'])]
    yy1 = Preceding[(xdata[['id']].values == lc.at[j, 'id']) & (xdata[['frame']].values > lc.at[j, 'lanechangeframe'])
                    & (xdata[['frame']].values <= lc.at[j, 'endframe'])]
    Precedingx = pd.concat([yy0, yy1], axis=0)


    yy2 = targetpreceding0[(ddata[['id']].values == lc.at[j, 'id']) & (ddata[['frame']].values <= lc.at[j, 'lanechangeframe'])
                           & (ddata[['frame']].values >= lc.at[j, 'startframe'])]
    yy3 = targetpreceding[(xdata[['id']].values == lc.at[j, 'id']) & (xdata[['frame']].values > lc.at[j, 'lanechangeframe'])
                          & (xdata[['frame']].values <= lc.at[j, 'endframe'])]
    TargetPrecedingx = pd.concat([yy2, yy3], axis=0)


    yy4 = targetfollowing0[(ddata[['id']].values == lc.at[j, 'id']) & (ddata[['frame']].values <= lc.at[j, 'lanechangeframe'])
                           & (ddata[['frame']].values >= lc.at[j, 'startframe'])]
    yy5 = targetfollowing[(xdata[['id']].values == lc.at[j, 'id']) & (xdata[['frame']].values > lc.at[j, 'lanechangeframe'])
                          & (xdata[['frame']].values <= lc.at[j, 'endframe'])]
    TargetFollowingx = pd.concat([yy4, yy5], axis=0)


    Precedingxx = pd.concat([Precedingxx, Precedingx], axis=0)
    TargetPrecedingxx = pd.concat([TargetPrecedingxx, TargetPrecedingx], axis=0)
    TargetFollowingxx = pd.concat([TargetFollowingxx, TargetFollowingx], axis=0)


Precedingxx = Precedingxx.reset_index(drop=True)
TargetPrecedingxx = TargetPrecedingxx.reset_index(drop=True)
TargetFollowingxx = TargetFollowingxx.reset_index(drop=True)
qdata = pd.concat([ydata, Precedingxx, TargetPrecedingxx, TargetFollowingxx], axis=1)

qdata.to_csv(os.path.join(path, 'lane-change-prediction(trajectory).csv'), index=False, sep=',')








# 将目标车道的前后车的轨迹变成相对间隔和相对速度
path = r'E:\PythonProjects1\Lane change prediciton\lane-change-prediction-NGSIM'
xdata = pd.read_csv(os.path.join(path, 'lane-change-prediction(trajectory).csv'), parse_dates=True, encoding='gbk')
lc = pd.read_csv(os.path.join(path, 'lane-change-QKV.csv'), parse_dates=True, encoding='gbk')
xx = lc

xdata['x1'] = pd.DataFrame(xdata[['localy-target-pre']].values - xdata[['localy']].values)
xdata['r1'] = pd.DataFrame(xdata[['speedy-target-pre']].values - xdata[['speedy']].values)
xdata['x2'] = pd.DataFrame(xdata[['localy']].values - xdata[['localy-target-foll']].values)
xdata['r2'] = pd.DataFrame(xdata[['speedy']].values - xdata[['speedy-target-foll']].values)

a1 = np.where(xdata[['id-target-pre']].values == 0)
a2 = np.where(xdata[['id-target-foll']].values == 0)

xdata.loc[a1[0], 'x1'] = 0
xdata.loc[a1[0], 'r1'] = 0
xdata.loc[a2[0], 'x2'] = 0
xdata.loc[a2[0], 'r2'] = 0

xxxdata = [['speedx', 'speedy', 'accx', 'accy', 'yaw', 'speedr', 'gap', 'localx', 'localy']]

yydata = pd.DataFrame()
for i in range(np.size(xx, 0)):
    print(i)
    ydata = xdata[(xdata[['id']].values == xx.at[i, 'id']) & (xdata[['frame']].values <= xx.at[i, 'endframe'])
                 & (xdata[['frame']].values >= xx.at[i, 'startframe'])]
    ydata = ydata.reset_index()

    ydata['id'] = i
    ydata['Traffic'] = xx.at[1, 'Traffic']
    ydata['class'] = xx.at[1, 'class']
    yydata = pd.concat([yydata, ydata], axis=0)


data = yydata[['id', 'speedx', 'speedy', 'accx', 'accy', 'yaw', 'speedr', 'gap', 'x1', 'r1', 'x2', 'r2',
                     'localx', 'localy', 'Traffic', 'class', 'maneuver']]


data.loc[data['Traffic'] == 'A', 'Traffic'] = 0
data.loc[data['Traffic'] == 'B', 'Traffic'] = 1
data.loc[data['Traffic'] == 'C', 'Traffic'] = 2
data.loc[data['Traffic'] == 'D', 'Traffic'] = 3
data.loc[data['Traffic'] == 'E', 'Traffic'] = 4
data.to_csv(os.path.join(path, 'lane-change-prediction(trajectory2).csv'), index=False, sep=',')
'''





path = r'E:\PythonProjects1\Lane change prediciton\lane-change-prediction-NGSIM'
ddata = pd.read_csv(os.path.join(path, 'lane-change-prediction(trajectory).csv'), parse_dates=True, encoding='gbk')
xdata = pd.read_csv(os.path.join(path, 'lane-change-prediction(trajectory2).csv'), parse_dates=True, encoding='gbk')
lc = pd.read_csv(os.path.join(path, 'lane-change-QKV.csv'), parse_dates=True, encoding='gbk')
xx = lc
yydata = pd.DataFrame()
# 由于前后各5s，时间拉的有点长，变成换道前后2s
for i in range(np.size(xx, 0)):
    print(i)
    ydata = xdata[(xdata[['id']].values == i) & (ddata[['frame']].values <= xx.at[i, 'lanechangeframe']+20)
                 & (ddata[['frame']].values >= xx.at[i, 'lanechangeframe']-20)]
    ydata = ydata.reset_index(drop=True)

    yydata = pd.concat([yydata, ydata], axis=0)



yydata.to_csv(os.path.join(path, 'lane-change-prediction(trajectory3).csv'), index=False, sep=',')








