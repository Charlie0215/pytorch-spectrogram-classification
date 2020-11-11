import torch
import torch.nn as nn
from utils.utils import accuracy, scale_minmax, poly_learning_decay
import os

class BaseTrainer(object):
    def __init__(self, 
                model=None, 
                optimizer=None, 
                scheduler=None, 
                loss=None,
                train_dataloader=None, 
                test_dataloader=None, 
                use_gpu=None, 
                config=None
        ):
        
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.loss = loss
        self.iterations = 0
        self.config = config

        self.device = torch.device('cuda:0' if use_gpu else 'cpu')

    def run(self, epochs=1):
        self.pre_acc = self.validation()
        self.best_epoch = 0
        for i in range(1, epochs + 1):
            self.train(i)
            acc = self.validation()
            print("acc: {}".format(acc))
            state = {"model_state": self.model.state_dict()}
            save_path = os.path.join(self.config.weight_save_dir, 'epoch_{}.pkl'.format(i))
            print(save_path)
            torch.save(state, save_path)
            if acc >= self.pre_acc:
                self.pre_acc = acc
                self.best_epoch = i
                state = {"model_state": self.model.state_dict()}
                torch.save(state, self.config.weight_load_dir)
                print('save_best_weight')
    
    def train(self, epoch):
        print('############# training {} #############'.format(len(self.train_dataloader)*self.config.batch_size))
        for batch_id, data in enumerate(self.train_dataloader):
            spc, label = data
            spc = spc.to(self.device)
            label = label.to(self.device)
            pred = self.model(spc)
            self.optimizer.zero_grad()
            loss = self.loss(pred, label)
            loss.backward()
            self.optimizer.step()

            new_lr = poly_learning_decay(
                self.optimizer, self.iterations, self.config.epoch, len(self.train_dataloader), self.config.lr
            )
            self.iterations += 1

            if batch_id % 100 == 0:
                print(
                    'iter: {} | epoch: {}/{} | loss: {:.4f} | best loss/epoch: {}/{}'.format(
                        batch_id, epoch, self.config.epoch, loss.item(), 
                        self.pre_acc, self.best_epoch,
                ))
        
    def validation(self):
        print('############# validation {} #############'.format(len(self.test_dataloader)))
        self.model.eval()
        acc = 0
        for batch_id, data in enumerate(self.test_dataloader):
            spc, label = data
            spc = spc.to(self.device)
            label = label.to(self.device)
            pred = self.model(spc)
            acc += accuracy(pred, label)
        acc = acc / len(self.test_dataloader)
        self.model.train()
        return acc
        

class multiheadTrainer(object):
    def __init__(self, 
                model=None, 
                optimizer=None, 
                scheduler=None, 
                loss=None,
                train_dataloader=None, 
                test_dataloader=None, 
                use_gpu=None, 
                config=None
        ):
        
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.loss = loss
        self.iterations = 0
        self.config = config

        self.device = torch.device('cuda:0' if use_gpu else 'cpu')

    def run(self, epochs=1):
        self.pre_acc = self.validation()
        self.best_epoch = 0
        for i in range(1, epochs + 1):
            self.train(i)
            acc = self.validation()
            print("acc: {}".format(acc))
            state = {"model_state": self.model.state_dict()}
            save_path = os.path.join(self.config.weight_save_dir, 'epoch_{}.pkl'.format(i))
            print(save_path)
            torch.save(state, save_path)
            if acc >= self.pre_acc:
                self.pre_acc = acc
                self.best_epoch = i
                state = {"model_state": self.model.state_dict()}
                torch.save(state, self.config.weight_load_dir)
                print('save_best_weight')
    
    def train(self, epoch):
        print('############# training {} #############'.format(len(self.train_dataloader)*self.config.batch_size))
        for batch_id, data in enumerate(self.train_dataloader):
            spc, label = data
            spc = [x.to(self.device) for x in spc]
            label = label.to(self.device)
            pred = self.model(spc)
            self.optimizer.zero_grad()
            loss = self.loss(pred, label)
            loss.backward()
            self.optimizer.step()

            new_lr = poly_learning_decay(
                self.optimizer, self.iterations, self.config.epoch, len(self.train_dataloader), self.config.lr
            )
            self.iterations += 1
            if batch_id % 100 == 0:
                print(
                    'iter: {} | epoch: {}/{} | loss: {:.4f} | best loss/epoch: {}/{}'.format(
                        batch_id, epoch, self.config.epoch, loss.item(), 
                        self.pre_acc, self.best_epoch,
                ))
        
    def validation(self):
        print('############# validation {} #############'.format(len(self.test_dataloader)))
        self.model.eval()
        acc = 0
        for batch_id, data in enumerate(self.test_dataloader):
            spc, label = data
            spc = [x.to(self.device) for x in spc]
            label = label.to(self.device)
            pred = self.model(spc)
            acc += accuracy(pred, label)
        acc = acc / len(self.test_dataloader)
        self.model.train()
        return acc