
import os
import pandas as pd
import numpy as np
from time import time
import scipy.io as sio


# ## 数据初始化
save_path = r'E:\PythonProjects1\Lane change prediciton\lane-change-prediction-NGSIM'
pfile = str('E:/PythonProjects1/Lane change prediciton/lane-change-prediction-NGSIM/') + str('data04-2.mat')


# 通过标签索引，提取数据
data = pd.DataFrame(sio.loadmat(pfile)['data'])
lc = pd.DataFrame(sio.loadmat(pfile)['lc'])
ydata = pd.DataFrame(sio.loadmat(pfile)['ydata'])  # 换道样本

data.columns = ['id', 'frame', 'localx', 'localy', 'length', 'width', 'speedx', 'speedy',
                'accx', 'accy', 'class', 'lane', 'precedingid', 'speedr', 'gap', 'yaw',
                'maneuver']
lc.columns = ['id', 'maneuver', 'startline', 'endline', 'startframe', 'endframe',
                             'num', 'lanechangeframe', 'class']
ydata.columns = data.columns

xdata = pd.DataFrame()
# 提取换道前5s的数据，用来识别交通流、车辆类型以及intention预测
for i in range(np.size(lc, 0)):
    xxdata = ydata[(ydata[['id']].values == lc.at[i, 'id']) & (ydata[['frame']].values >= lc.at[i, 'startframe'])
                  & (ydata[['frame']].values <= lc.at[i, 'lanechangeframe'])]
    xdata = pd.concat([xdata, xxdata], axis=0)

xdata = xdata.reset_index(drop=True)

xdata['leftlane'] = xdata['lane'] - 1
xdata['rightlane'] = xdata['lane'] + 1



# ### 数据匹配
leftPreceding = pd.DataFrame()
leftFollowing = pd.DataFrame()
rightPreceding = pd.DataFrame()
rightFollowing = pd.DataFrame()


# 换道前的数据
idata = xdata
dff2 = np.zeros((1, 3))
dff2 = pd.DataFrame(dff2)
dff2.columns = ['id', 'localy', 'speedy']

#  跟驰样本
#  左边车道前后车
for j in range(np.size(idata, 0)):
    print(j)

    x1 = idata.at[j, 'localy'] + 120
    x2 = idata.at[j, 'localy'] - 120
    dff1 = data[(data[['lane']].values == idata.at[j, 'leftlane']) & (data[['frame']].values == idata.at[j, 'frame'])
                & (data[['localy']].values < x1) & (data[['localy']].values > x2)]

    if np.size(dff1, 0) == 0:
        leftPreceding = pd.concat([leftPreceding, dff2], axis=0)
        leftFollowing = pd.concat([leftFollowing, dff2], axis=0)


    else:
        distance = dff1[['localy']] - idata.at[j, 'localy']
        distance1 = distance[distance[['localy']].values > 0]
        distance2 = distance[distance[['localy']].values < 0]

        if np.size(distance1, 0) != 0:
            y1 = np.min(distance1[['localy']].values)
            dff3 = dff1.loc[distance['localy'].values == y1, ['id', 'localy', 'speedy']]
        else:
            dff3 = pd.DataFrame(np.zeros((1, 3)))
            dff3.columns = ['id', 'localy', 'speedy']


        if np.size(distance2, 0) != 0:
            y2 = np.max(distance2[['localy']].values)
            dff4 = dff1.loc[distance['localy'].values == y2, ['id', 'localy', 'speedy']]
        else:
            dff4 = pd.DataFrame(np.zeros((1, 3)))
            dff4.columns = ['id', 'localy', 'speedy']


        leftPreceding = pd.concat([leftPreceding, dff3], axis=0)
        leftFollowing = pd.concat([leftFollowing, dff4], axis=0)

leftPreceding = leftPreceding.reset_index(drop=True)
leftFollowing = leftFollowing.reset_index(drop=True)
mdata = pd.concat([leftPreceding, leftFollowing], axis=1)




# 右边车道前后车
for j in range(np.size(idata, 0)):

    x1 = idata.at[j, 'localy'] + 120
    x2 = idata.at[j, 'localy'] - 120
    dff1 = data[(data[['lane']].values == idata.at[j, 'rightlane']) & (data[['frame']].values == idata.at[j, 'frame'])
                & (data[['localy']].values < x1) & (data[['localy']].values > x2)]

    if np.size(dff1, 0) == 0:
        rightPreceding = pd.concat([rightPreceding, dff2], axis=0)
        rightFollowing = pd.concat([rightFollowing, dff2], axis=0)

    else:
        distance = dff1[['localy']] - idata.at[j, 'localy']
        distance1 = distance[distance[['localy']].values > 0]
        distance2 = distance[distance[['localy']].values < 0]

        if np.size(distance1, 0) != 0:
            y1 = np.min(distance1[['localy']].values)
            dff3 = dff1.loc[distance['localy'].values == y1, ['id', 'localy', 'speedy']]
        else:
            dff3 = pd.DataFrame(np.zeros((1, 3)))
            dff3.columns = ['id', 'localy', 'speedy']

        if np.size(distance2, 0) != 0:
            y2 = np.max(distance2[['localy']].values)
            dff4 = dff1.loc[distance['localy'].values == y2, ['id', 'localy', 'speedy']]
        else:
            dff4 = pd.DataFrame(np.zeros((1, 3)))
            dff4.columns = ['id', 'localy', 'speedy']

        rightPreceding = pd.concat([rightPreceding, dff3], axis=0)
        rightFollowing = pd.concat([rightFollowing, dff4], axis=0)

rightPreceding = rightPreceding.reset_index(drop=True)
rightFollowing = rightFollowing.reset_index(drop=True)
mmdata = pd.concat([rightPreceding, rightFollowing], axis=1)


ndata = pd.concat([xdata, mdata, mmdata], axis=1)
ndata.to_csv(os.path.join(save_path, 'lane-change-surrounding.csv'), index=False, sep=',')

