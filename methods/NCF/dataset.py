import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch.utils.data import TensorDataset, DataLoader

class NCFDataset(object):
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def load_dataset(self):
        # get the dataset path
        dataset_path = "./data/" + self.dataset_name
        train_data_path = dataset_path + ".train.rating"
        test_data_path = dataset_path + ".test.rating"
        test_neg_path = dataset_path + ".test.negative"

        # load training data
        train_data = pd.read_csv(
            train_data_path,
            sep='\t', header=None, names=['user', 'item'],
            usecols=[0, 1], dtype={0: np.int32, 1: np.int32})

        user_num = train_data['user'].max() + 1
        item_num = train_data['item'].max() + 1

        train_data = train_data.values.tolist()

        # load ratings as a dok matrix
        train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
        for x in train_data:
            train_mat[x[0], x[1]] = 1.0

        test_data = []
        with open(test_neg_path, 'r') as fr:
            line = fr.readline()
            while line != None and line != '':
                arr = line.split('\t')
                u = eval(arr[0])[0]
                test_data.append([u, eval(arr[0])[1], 1])
                for i in arr[1:]:
                    test_data.append([u, int(i), 0])
                line = fr.readline()
        return train_data, test_data, user_num, item_num, train_mat

    # get the training dataset
    # user_num is useless in here, I just add them for looking comfortable
    def get_train_instances(self, train_data, item_num, neg_num, train_mat):
        user_input, item_input, label = [], [], []
        for (u, i) in train_data:
            user_input.append(u)
            item_input.append(i)
            label.append(1)  # postive sample
            # negative instances
            for _ in range(neg_num):
                j = np.random.randint(item_num)
                while (u, j) in train_mat:
                    j = np.random.randint(item_num)
                # add neg samples
                user_input.append(u)
                item_input.append(j)
                label.append(0)
        return user_input, item_input, label

    def get_train_dataloader(self, batch_size, train_data, item_num, neg_num, train_mat):
        # get data
        user_input, item_input, label = self.get_train_instances(
            train_data, item_num, neg_num, train_mat)
        # get tensor data
        train_data = TensorDataset(torch.LongTensor(user_input),
            torch.LongTensor(item_input), torch.LongTensor(label))
        # get data loader
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
        return train_loader

    def get_test_dataloader(self, test_data, test_pos_neg_num):
        user_input, item_input, label = [], [], []
        for (u, i, l) in test_data:
            user_input.append(u)
            item_input.append(i)
            label.append(l)
        test_tensor_data = TensorDataset(torch.LongTensor(user_input),
            torch.LongTensor(item_input), torch.LongTensor(label))
        test_loader = DataLoader(test_tensor_data,
                batch_size=test_pos_neg_num, shuffle=False)
        return test_loader


