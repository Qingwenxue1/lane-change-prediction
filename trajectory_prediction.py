import time
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler


def create_dataset(_df, look_back=10):
    indexs = _df.index.drop_duplicates().tolist()  # 找到所有的序号.index找到行号，.drop_duplicates去掉重复
    scaler = MinMaxScaler(feature_range=(0, 1))

    dataset1 = scaler.fit_transform(_df.values[:, :-5])  # 归一化，并设置成原格式
    dataset2 = _df.values[:, -5:-2]
    dataset3 = scaler.fit_transform(_df.values[:, -2:])
    dataset = np.concatenate((dataset1, dataset2, dataset3), axis=1)
    dataset = pd.DataFrame(dataset)
    dataset.index = _df.index
    dataset.columns = _df.columns

    x, y = [], []
    for index in indexs:
        xdata = dataset[dataset.index == index].copy()
        data_adj = adjust_input(xdata.copy())  # adjust_inputs为一个定义的函数
        data_adj[:, -2:] = scaler.fit_transform(data_adj[:, -2:])

        for i in range(len(data_adj) - look_back):
            a = data_adj[i: (i + look_back), :]
            b = data_adj[(i + 1):(i + look_back + 1), -2:]
            x.append(a)
            y.append(b)
    x = pad_sequences(x, look_back)
    y = pad_sequences(y, look_back)
    return x, y, dataset


def adjust_input(part):  # 处理原始轨迹数据，将远点设置为[0,0]
    part = part.reset_index(drop=True)
    part['X'] = (part['X'] - part.at[0, 'X']).values
    part['Y'] = (part['Y'] - part.at[0, 'Y']).values
    if np.mean(part['maneuver']) == 3:  # 将左右换道的方向一致 都设置成正值
        part['X'] = part['X'] * (-1)
    # part1 = part.drop(columns=['maneuver', 'class', 'Traffic'])  # 根据输入变量调整
    part1 = part.drop(columns=['class', 'Traffic'])
    return np.array(part1)


def pad_sequences(sequence, max_length):  # 数组长度不足可以填补
    batch = len(sequence)
    vec = sequence[0].shape[1]
    res = np.zeros((batch, max_length, vec)).astype(np.float32)  # 由0填充的数组；astype数据类型
    for j in range(batch):
        length = sequence[j].shape[0]
        keep_part = np.array(sequence[j])
        res[j, -length:, :] = keep_part.copy()
    return res


def predict_model(_model, dataset):  # 单点预测并将预测真值处理成数组
    predict_y = []
    prediction = _model.predict(dataset)
    for j in np.arange(len(prediction)):
        predict_y.append(prediction[j, -1, :])
    predict_y = np.array(predict_y)
    print('predicted shape:', np.array(predict_y).shape)
    return predict_y


def predict_result(results):  # 预测真值处理成数组
    predict_data = []
    for j in np.arange(len(results)):
        predict_data.append(results[j, -1, :])
    predict_data = np.array(predict_data)
    print('predicted shape:', np.array(predict_data).shape)
    return predict_data


def plot_results(predicted_data, true_data):
    plt.figure(figsize=(8, 7))
    font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 25}
    font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 15}
    plt.plot(predicted_data[:, 1], predicted_data[:, 0], label='Prediction Data', linewidth=3)
    plt.plot(true_data[:, 1], true_data[:, 0], label='True Data', linewidth=3)
    plt.plot([0, true_data[-1:, 1]], [true_data[19, 0], true_data[19, 0]], '0.90', linestyle='--', linewidth=3,
             label='Lane separation line')
    plt.legend(prop=font1)
    plt.xlabel("Longitudinal trajectory of vehicle /m", font2)
    plt.ylabel("Lateral trajectory of vehicle /m", font2)
    plt.tick_params(labelsize=15)
    plt.show()


## 预测换道轨迹
path1 = r'E:\PythonProjects1\Lane change prediciton\lane-change-prediction-NGSIM'
df = pd.read_csv(os.path.join(path1, 'lane-change-prediction(trajectory3).csv'), encoding='gbk', index_col=0, header=0, parse_dates=[0], squeeze=True)

data = df
train_X, train_Y, number = create_dataset(data)

model = Sequential()
model.add(LSTM(input_dim=14, output_dim=50, return_sequences=True))
model.add(LSTM(input_dim=50, output_dim=100, return_sequences=True))
model.add(LSTM(input_dim=100, output_dim=200, return_sequences=True))
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(50))
model.add(Dense(output_dim=2))
start = time.time()
model.compile(loss='mean_squared_error', optimizer='Adam')  # learning rate怎么设置呢？？？
model.summary()
model.fit(train_X, train_Y, batch_size=32, nb_epoch=5, validation_split=0.1, verbose=2)
print('compilation time:', time.time() - start)



ids = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
# ids = [10, 12]
for id in ids:
    idata = data[data.index == id].copy()
    test_X, test_Y, state = create_dataset(idata)

    # # 采用训练的模型来预测数据
    predictions = predict_model(model, test_X)
    scaler = MinMaxScaler(feature_range=(0, 1))
    idata1 = adjust_input(idata)
    idata2 = idata1[:, -2:]

    scaler.fit_transform(idata2)
    prediction = scaler.inverse_transform(predictions)
    mse1 = mean_squared_error(prediction[:, -1:], idata2[10:, -1:])  # 纵向坐标的预测与真值之间的差值
    mse2 = mean_squared_error(prediction[:, :1], idata2[10:, :1])  # 横向坐标的预测与真值之间的差值
    mse = mean_squared_error(prediction[:, :], idata2[10:, :])  # 预测整体误差

    error = [mse1, mse2, mse]
    print(error)

    # if np.mean(idata[['maneuver']].values) == 3:
    #     prediction[:, :-1] = prediction[:, :-1] * (-1)
    #     idata2[:, :-1] = idata2[:, :-1] * (-1)
    #     plot_results(prediction, idata2)
    # else:
    #     plot_results(prediction, idata2)



# 输出数据
# prediction1=pd.DataFrame(prediction)
# prediction1.to_csv("data1.csv",index=False,sep=',')

# pickle.dump(model, open("lstm1.dat", "wb"))  #  保存模型
# model = pickle.load(open("rf2.pickle.dat", "rb"))  # 引用模型
