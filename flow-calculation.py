import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from time import time
import scipy.io as sio
# ##每个数据包依次计算

start = time()

save_path = r'C:\Users\Lenovo\Desktop\新建文件夹 (2)'
pfile = str('C:/Users/Lenovo/Desktop/新建文件夹 (2)/') + str('data04-2.mat')

# 通过标签索引，提取数据
data = pd.DataFrame(sio.loadmat(pfile)['data'])
lk = pd.DataFrame(sio.loadmat(pfile)['lk'])
lc = pd.DataFrame(sio.loadmat(pfile)['lc'])
trajectories_last = pd.DataFrame(sio.loadmat(pfile)['trajectories_last'])
xdata = pd.DataFrame(sio.loadmat(pfile)['xdata'])  # 跟驰样本
ydata = pd.DataFrame(sio.loadmat(pfile)['ydata'])

data.columns = ['id', 'frame', 'localx', 'localy', 'length', 'width', 'speedx', 'speedy',
                'accx', 'accy', 'class', 'lane', 'precedingid', 'speedr', 'gap', 'yaw',
                'maneuver']
trajectories_last.columns = ['id', 'maneuver', 'startline', 'endline', 'startframe', 'endframe',
                             'num', 'lanechangeframe', 'class']
xdata.columns = data.columns
ydata.columns = data.columns
lk.columns = trajectories_last.columns
lc.columns = trajectories_last.columns

# 计算
sample = lc   # 分别计算跟驰样本和换道样本
v_list = []
K_list = []
Q_list = []
for i in tqdm(range(len(sample))):
    sample_id = sample.at[i, 'id']
    sample_start = sample.at[i, 'startframe']
    sample_end = sample.at[i, 'endframe']


    # ##计算车流量
    a1 = np.where((data['frame'] >= sample_start) & (data['frame'] <= sample_end))   # 找到时间范围内在车道上的车id
    idata = data.iloc[a1[0], :]
    idx1 = idata['id'].unique()
    y = 0
    for k in range(len(idx1)):
        iidata = (idata[idata['id'] == idx1[k]]).reset_index(drop=True)
        x1 = (iidata['localy']).values[0]
        x2 = (iidata['localy']).values[-1]
        x = (x1 - 400) * (x2 - 400)  # 判断时间范围内该id有没有经过中心线，如果经过，流量+1
        if x < 0:
            y = y + 1

    y1 = round(y / 7, 3)  # 时间范围内相同方向内平均每个车道的交通流量
    y2 = round((y * 3600) / ((sample_end - sample_start) * 0.1), 3)  # 通过除以时间，得到以单位为h的交通流量
    y3 = round(y2 / 7, 3)  # 单个车道内的流量
    z = np.array([y, y1, y2, y3])
    Q_list.append(z)


    # ## 计算车密度和平均速度
    data_d = data  # 与样本车行方向相同的所有id
    frame_v_mean_list = []
    frame_K_list = []
    for j in range(sample_start, sample_end + 1):  # 遍历各个frame
        data_frame_j = (data_d[data_d['frame'] == j]).reset_index(drop=True)
        j_v_mean = np.mean(data_frame_j['speedy'])  # 在此frame下的所有车速的均值
        j_K = len(data_frame_j)  # 在此frame下的密度K
        frame_v_mean_list.append(j_v_mean)
        frame_K_list.append(j_K)
    sample_v = round(np.mean(frame_v_mean_list), 3)  # 此样本对应的平均车速
    sample_K = round(np.mean(frame_K_list), 3)  # 此样本对应的平均密度
    v_list.append(sample_v)
    K_list.append(sample_K)


id_num = pd.DataFrame(Q_list)
id_num.columns = ['id_num', 'Q(veh/lane)',	'Q(veh/h)',	'Q(veh/h/ln)']
v_df = pd.DataFrame({'v(m/s)': v_list})
v_df['V(km/h)'] = round(v_df['v(m/s)'] * 3.6, 3)  # 所有车辆的平均速度
K_df = pd.DataFrame({'K(veh/800m)': K_list})
K_df['K(veh/km)'] = round(K_df['K(veh/800m)'] / 0.8, 3)
K_df['K(veh / km / ln)'] = K_df[['K(veh/km)']]/7  # 单个车道的车流量密度

df = pd.concat([id_num, v_df, K_df], axis=1, sort=False)
m = pd.concat([lc, df], axis=1)
m.to_csv(os.path.join(save_path, 'lane-change-QVK.csv'))


