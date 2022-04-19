import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import os

path1 = r'E:\PythonProjects1\Lane change prediciton\lane-change-prediction-NGSIM'

data1 = pd.read_csv(os.path.join(path1, 'car-following-QKV(original).csv'), parse_dates=True, encoding='utf-8')
data2 = pd.read_csv(os.path.join(path1, 'lane-change-QKV.csv'), parse_dates=True, encoding='utf-8')

data = data1


k = data[['K(veh/km/ln)']].values
v = data[['V(km/h)']].values
q = data[['Q(veh/h/ln)']].values

# kmeans clustering traffic flow level
x = np.concatenate([k, v], axis=1)
kmeans = KMeans(n_clusters=4, random_state=0)
y_pred = kmeans.fit_predict(k)  # 根据什么参数来label
sil_coeff = silhouette_score(k, y_pred, metric='euclidean')
center = kmeans.cluster_centers_
kc_list = sorted([i for i in center[:, 0]])
print(kc_list)

def my(_x):
    if _x == kc_list[0]:
        return "A"
    elif _x == kc_list[1]:
        return "B"
    elif _x == kc_list[2]:
        return "C"
    elif _x == kc_list[3]:
        return "D"


rrr = pd.DataFrame([my(x) for x in center[y_pred]])
data['Traffic'] = rrr.values

# data.to_csv(os.path.join(path1, 'lane_change_QKV.csv'), index=False, sep=',')

x = data[['K(veh/km/ln)', 'V(km/h)']]
x1 = data[data[['Traffic']].values == 'A']
v1 = np.mean(x1[['V(km/h)']].values)






# x1 = x[(data['Traffic'] == 'low traffic')]
# x2 = x[(data['Traffic'] == 'high traffic')]
#
# s1 = plt.scatter(x1[:, 0], x1[:, 1], color='#0000FF')  # 分别画出低交通流和高交通流的图片
# s2 = plt.scatter(x2[:, 0], x2[:, 1], color='#FF8C00')
#
# plt.rcParams["font.family"] = "serif"
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# # plt.ylim(2, 160)
# plt.xlabel('交通流密度 (veh/km)', fontsize=16, fontproperties='SimHei')
# plt.ylabel('交通流平均速度 (km/h)', fontsize=16, fontproperties='SimHei')
# # plt.legend(loc='upper right', frameon=False, labels=['Low traffic flow', 'High traffic flow'], prop={'size': 16})  # 显示标签
# plt.legend((s1, s2), ('低交通流', '高交通流'), loc='upper right', prop={'size': 16, 'family': 'SimHei'})  # 显示标签
# plt.tight_layout()
# plt.show()


