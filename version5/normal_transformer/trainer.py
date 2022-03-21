import torch
from pathlib import Path
import pickle
from utils import save2pkl
import random
import numpy as np

class Trainer:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.device = kwargs.get("device")
        self.args = kwargs.get("args")
        self.model = kwargs.get("model").to(self.device)
        self.optimizer = kwargs.get("optimizer")
        self.scheduler = kwargs.get("scheduler")
        self.criterion = kwargs.get("criterion")
        self.epochs = kwargs.get("epochs")
        self.train_loader = kwargs.get("train_loader")
        self.test_loader = kwargs.get("test_loader")
        self.model_save_name = kwargs.get('model_save_name')
        self.augmentation_rate = kwargs.get('augmentation_rate')
        self.window_len = kwargs.get('window_len')
        self.true_label, self.pre_label = [], []

    def train_step(self):
        self.model.train()
        self.train_loss = 0
        self.train_acc_num = 0
        for feature, label in self.train_loader:
            feature = feature.to(self.device).reshape(-1, 2, self.args.embed_dim, 64)
            label = label.to(self.device)

            self.optimizer.zero_grad()
            preds = self.model(feature)
            loss = self.criterion(preds, label)
            loss.backward()
            self.optimizer.step()
            self.train_acc_num += (preds.argmax(1) == label).sum()
            self.train_loss += loss.item() / len(self.train_loader)
        return self.train_loss, self.train_acc_num

    def test_step(self):
        self.model.eval()
        self.test_loss = 0
        self.test_acc_num = 0
        with torch.no_grad():
            for feature, label in self.test_loader:
                feature = feature.to(self.device).reshape(-1, 2, self.args.embed_dim, 64)

                self.true_label.extend(label)
                label = label.to(self.device)
                preds = self.model(feature)
                self.test_loss += self.criterion(preds, label) / len(self.test_loader)
                self.test_acc_num += ((preds.argmax(1) == label).sum())
                self.pre_label.extend(preds.argmax(1).cpu())
        return self.test_loss, self.test_acc_num

    def aug_train_step(self):
        self.model.train()
        self.train_acc_num = 0
        self.train_loss = 0
        for feature, label in self.train_loader:
            feature = feature.to(self.device)
            label = label.to(self.device)
            for _ in range(self.augmentation_rate):
                self.optimizer.zero_grad()
                start = random.choice(range(self.augmentation_rate))
                #aug_feature = feature[:, start : start+self.window_len]
                aug_feature = feature[:, start : start+self.window_len].reshape(-1, 2, self.args.embed_dim, 60)
                preds = self.model(aug_feature)
                loss = self.criterion(preds, label)
                loss.backward()
                self.train_loss += loss.item()
                self.optimizer.step()
                self.train_acc_num += (preds.argmax(1) == label).sum()

                feature.cpu()
                label.cpu()

        return self.train_loss, torch.div(self.train_acc_num, self.augmentation_rate)

    def aug_test_step(self):
        self.model.eval()
        self.test_loss = 0
        self.test_acc_num = 0
        self.length = None
        with torch.no_grad():
            for feature, label in self.test_loader:
                feature = feature.to(self.device)
                label = label.to(self.device)
                output = torch.zeros(feature.shape[0], self.augmentation_rate).to(self.device)
                start = 0
                for _ in range(self.augmentation_rate):
                    #aug_feature = feature[:, start:start + self.window_len]
                    aug_feature = feature[:, start:start + self.window_len].reshape(-1, 2, self.args.embed_dim, 60)
                    preds = self.model(aug_feature)
                    self.test_loss += self.criterion(preds, label).item()
                    preds = torch.argmax(preds, 1)
                    output[:, _] = preds.reshape(-1, )
                    start += 1

                    feature.cpu()
                    label.cpu()
                output = output.cpu().numpy().astype(np.int16)
                rank = np.zeros(output.shape[0])
                for i in range(output.shape[0]):
                    rank[i] = np.argsort(np.bincount(output[i, :]))[-1]

                rank = torch.LongTensor(rank)
                self.test_acc_num += (rank.cpu() == label.cpu()).sum()
                self.true_label.extend(label.cpu())
                self.pre_label.extend(rank.cpu())
        return self.test_loss / self.augmentation_rate, self.test_acc_num

    def train(self):
        self.best_score = -float('inf')
        train_loss_list, test_loss_list, train_acc_list, test_acc_list = [], [], [], []
        path = 'exp'/Path(self.kwargs.get('exp'))
        if not path.exists():
            path.mkdir(parents=True) # 创建多级子目录

        for epoch in range(self.epochs):
            train_loss, train_acc_num = self.train_step()
            #train_loss, train_acc_num = self.aug_train_step()
            test_loss, test_acc_num = self.test_step()
            #test_loss, test_acc_num = self.aug_test_step()
            train_acc = train_acc_num/len(self.train_loader.dataset)
            test_acc = test_acc_num/len(self.test_loader.dataset)
            self.scheduler.step(test_loss)


            if self.args.local_rank == 0:
                print(f'Epoch:{epoch:2} | Train Loss:{train_loss:6.4f} | Train Acc:{train_acc:6.4f} | Test Loss:{test_loss:6.4f} | Test Acc:{test_acc:6.4f}')

            if self.args.local_rank == 0:
                if test_acc > self.best_score:
                    self.best_score = test_acc
                    self.model_save(self.model_save_name)

            # 保存准确率等数据

            if self.args.local_rank == 0:
                train_loss_list.append(train_loss)
                train_acc_list.append(train_acc)
                test_loss_list.append(test_loss)
                test_acc_list.append(test_acc)
        if self.args.local_rank == 0:
            save2pkl(train_loss_list, str(path/'train_loss.pkl'))
            save2pkl(train_acc_list, str(path/'train_acc.pkl'))
            save2pkl(test_loss_list, str(path/'test_loss.pkl'))
            save2pkl(test_acc_list, str(path/'test_acc.pkl'))
            save2pkl(self.true_label, str(path/'true_labels.pkl'))
            save2pkl(self.pre_label, str(path/'pre_labels.pkl'))

    def model_save(self, file_name):
        path = Path('./models')
        if not path.exists():
            path.mkdir()
        path = str(path / Path(file_name))
        torch.save(self.model.state_dict(), path)
