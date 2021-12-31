from pathlib import Path
import numpy as np 
import time
from datetime import datetime
from tqdm import tqdm
from collections import Counter
import random

class read_data:
    def __init__(self):
        self.phone2id = {}
        self.id2phone = {}
        self.trainBincount = None
        self.testBincount = None

    def read_data_by_sort_time(self, root='../16ue_20211223/', balance=True):
        # 按时间排序：前百分之70作为训练集，后百分之30作为验证集
        # 如果数据不一样，只要修改文件的提取方式即可
        p = Path(root)
        phone_data = {}
        file_count = [] # 统计文件数量
        count = 0
        for smi in p.iterdir():
            smi_name = smi.name
            if phone_data.get(smi_name) is None:
                phone_data[smi_name] = {} # 键保存时间好排序，值保存文件路径
            for dat in smi.iterdir():
                count += 1
                time = dat.stem.split('_')[-1]
                time = datetime.strptime(time, '%Y-%m-%d-%H-%M-%S')
                phone_data[smi_name][time] = dat
            file_count.append(count)
            count = 0

        treshold = int(np.percentile(file_count, 90)) # 这里不能保证为整数

        count = 0
        train_path, test_path, temp_train_label, temp_test_label = [], [], [], []
        for smi, time in phone_data.items():
            sorted_time_file = []
            self.phone2id[smi] = count

            if balance: 
                # 只能先随机完数据，再按时间排序
                file_len = len(time)
                file_len = file_len if file_len <= treshold else treshold
                time = {i:time[i] for i in random.sample(time.keys(), file_len)}

            for i in sorted(time): # 这里time是一个字典{time1:datfile1, time2:datfile2, ...}
                sorted_time_file.append(time[i])

            phone_data[smi] = sorted_time_file # sorted_time是按时间排序好的文件
            train_len = int(len(sorted_time_file) * 0.7)
            test_len = len(sorted_time_file) - train_len

            train_path.extend(sorted_time_file[:train_len]) # 70% be train 
            test_path.extend(sorted_time_file[train_len:])
            temp_train_label.extend([count]*train_len)
            temp_test_label.extend([count]*test_len)
            count += 1

        x_train, x_test, y_train, y_test = [], [], [], []
        self.id2phone = {id_:phone for phone, id_ in self.phone2id.items()}

        def get_data(path_list, label_list):
            x, y = [], []
            for idx, file in enumerate(path_list):
                data = np.fromfile(file, dtype=np.int16)
                L = len(data) // 8192
                data = data[: L*8192].reshape(-1, 8192)
                x.append(data)
                y.extend([label_list[idx]]*L)
            return np.concatenate(x), np.array(y)

        x_train, y_train = get_data(train_path, temp_train_label)
        x_test, y_test = get_data(test_path, temp_test_label)
        self.trainBincount = np.bincount(y_train)
        self.testBincount = np.bincount(y_test)
        return x_train, x_test, y_train, y_test


    def read_data_from_16_source(self, root='../16ue_20211223'):
        # 数据读取方式为：按天数划分，前几天作为训练集，最后一天作为验证集
        #%Y%m%d%H%M%S
        p = Path(root)
        phone_data = {}
        for smi in p.iterdir():
            smi_name = smi.name
            if phone_data.get(smi_name) is None:
                phone_data[smi_name] = {}
            for dat in smi.iterdir():
                time = dat.stem.split('_')[-1].split('.')[0] # extract date
                time = time.split('-')[0:3]
                time = ''.join(time)
                time = datetime.strptime(time, '%Y%m%d')
                if phone_data[smi_name].get(time) is None:
                    phone_data[smi_name][time] = []        
                phone_data[smi_name][time].append(dat)
        del_phone = []
        for phone, value in phone_data.items():
            if len(value.keys()) <= 1:
                del_phone.append(phone)
        for name in del_phone:
            del phone_data[name]
        x_train, x_test, y_train, y_test = [], [], [], []
        count = 0
        phone2id = {}
        for phone_name, value in tqdm(phone_data.items()):
            phone2id[phone_name] = count
            sort_list = sorted(value.keys())
            for train_time in sort_list:
                for file_path in value.get(train_time):
                    data = np.fromfile(file_path, np.int16)
                    L = len(data) // 8192
                    data = data[: L*8192].reshape(-1, 8192)
                    if train_time not in sort_list[-1:]:
                        x_train.append(data)
                        y_train.extend([count]*L)
                    else:
                        x_test.append(data)
                        y_test.extend([count]*L)
            count += 1
        x_train = np.concatenate(x_train)
        x_test = np.concatenate(x_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        self.phone2id = phone2id
        self.id2phone = {value:key for key, value in self.phone2id.items()}
        self.trainBincount = np.bincount(y_train)
        self.testBincount = np.bincount(y_test)
        return x_train, x_test, y_train, y_test

    
if '__main__' == __name__:
    read = read_data()
    #read.read_data_from_16_source()
    read.read_data_by_sort_time()
    #print(read.phone2id)
    print(read.id2phone)
    print(read.trainBincount)
    print(read.testBincount)
