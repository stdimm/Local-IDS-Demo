import torch

import numpy as np
import pandas as pd
import torch.utils.data as Data
from filter_utils import one_hot_new, normalize_new

def data_set(TxtPath):
    # 读取KDD-cup网络安全数据,将标签数字化
    df = pd.read_csv(TxtPath, sep=' ')
    df.columns = [x for x in range(41)]

    # 42列无用，删去
    labels = df.iloc[:, 40]
    data = df
    # 标签编码
    cols = [1, 2, 3]
    #独热编码
    new_train_df = one_hot_new(data, cols)

    # 特征值归一化
    combined_data = normalize_new(new_train_df, new_train_df.columns)
    #combined_data = normalize_new(df, df.columns)

    #labels = label_list(labels)

    # 标签和特征转成numpy数组
    data = np.array(combined_data)
    labels = np.array(labels)


    # 转成torch.tensor类型
    labels = torch.from_numpy(labels)
    data = torch.from_numpy(data).float()

    dataset = Data.TensorDataset(data, labels)
    dataset.data = dataset.tensors[0]
    dataset.targets = dataset.tensors[1]
    labels = ['DOS', 'Normal', 'Probe', 'R2L', 'U2R']
    dataset.classes = labels
    dataset.classes_to_idx = {i: label for i, label in enumerate(labels)}


    return dataset