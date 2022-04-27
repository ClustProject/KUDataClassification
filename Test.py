# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 23:23:45 2022

@author: jiyoon
"""

import pandas as pd
import numpy as np
import main_classificaiton as mc
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import os
import random
from sklearn.metrics import classification_report
#%%

# seed 고정
random_seed = 42

torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

#%%

# Case 1. w/o data representation & LSTM model 
config1 = {
        'with_representation': False, # classification에 사용되는 representation이 있을 경우 True, 아닐 경우 False
        'model': 'LSTM', # classification에에 활용할 알고리즘 정의, {'LSTM', 'GRU', 'CNN_1D', 'LSTM_FCNs', 'FC_layer'} 중 택 1

        'parameter': {
            'window_size' : 128, # input time series data를 windowing 하여 자르는 길이(size)
            'num_layers' : 2, # recurrnet layers의 수, Default : 1
            'hidden_size' : 64, # hidden state의 벡터차원 수
            'attention' : False, # True일 경우 attention layer를 추가
            'dropout' : 0.1, # If non-zero, introduces a Dropout layer on the outputs of each RNN layer except the last layer, with dropout probability equal to dropout. Default: 0
            'bidirectional' : True, # 모델의 양방향성 여부
            'batch_size' : 64, #batch size
            'device': 'cuda', # 학습 환경, ["cuda", "cpu"] 중 선택
            'num_epochs' : 150 # 학습 시 사용할 epoch 수
            }
}

# Case 2. w/o data representation & GRU model 
config2 = {
        'with_representation': False, # classification에 사용되는 representation이 있을 경우 True, 아닐 경우 False
        'model': 'GRU', # classification에에 활용할 알고리즘 정의, {'LSTM', 'GRU', 'CNN_1D', 'LSTM_FCNs', 'FC_layer'} 중 택 1

        'parameter': {
            'window_size' : 128, # input time series data를 windowing 하여 자르는 길이(size)
            'num_layers' : 2, # recurrnet layers의 수, Default : 1
            'hidden_size' : 64, # hidden state의 벡터차원 수
            'attention' : False, # True일 경우 attention layer를 추가
            'dropout' : 0.1, # If non-zero, introduces a Dropout layer on the outputs of each RNN layer except the last layer, with dropout probability equal to dropout. Default: 0
            'bidirectional' : True, # 모델의 양방향성 여부
            'batch_size' : 64, #batch size
            'device': 'cuda', # 학습 환경, ["cuda", "cpu"] 중 선택
            'num_epochs' : 150 # 학습 시 사용할 epoch 수
            }
}

# Case 3. w/o data representation & CNN_1D model 
config3 = {
        'with_representation': False, # classification에 사용되는 representation이 있을 경우 True, 아닐 경우 False
        'model': 'CNN_1D', # classification에에 활용할 알고리즘 정의, {'LSTM', 'GRU', 'CNN_1D', 'LSTM_FCNs', 'FC_layer'} 중 택 1

        'parameter': {
            'window_size' : 128, # input time series data를 windowing 하여 자르는 길이(size)
            'output_channels' : 64, # convolution channel size of output
            'drop_out' : 0.1, # If non-zero, introduces a Dropout layer on the outputs of each RNN layer except the last layer, with dropout probability equal to dropout. Default: 0
            'kernel_size' : 3, # convolutional filter size
            'stride' : 1, # stride of the convolution. Default = 1 
            'padding' : 0, # padding added to both sides of the input. Default = 0
            'batch_size' : 64, # batch size
            'device': 'cuda', # 학습 환경, ["cuda", "cpu"] 중 선택
            'num_epochs' : 150 # 학습 시 사용할 epoch 수
            }
}

# Case 4. w/o data representation & LSTM_FCNs model 
config4 = {
        'with_representation': False, # classification에 사용되는 representation이 있을 경우 True, 아닐 경우 False
        'model': 'LSTM_FCNs', # classification에에 활용할 알고리즘 정의, {'LSTM', 'GRU', 'CNN_1D', 'LSTM_FCNs', 'FC_layer'} 중 택 1

        'parameter': {
            'window_size' : 128, # input time series data를 windowing 하여 자르는 길이(size)
            'num_layers' : 1, # recurrnet layers의 수, Default : 1
            'lstm_drop_out' : 0.4, # If non-zero, introduces a Dropout layer on the outputs of each RNN layer except the last layer, with dropout probability equal to dropout. Default: 0
            'fc_drop_out' : 0.1, # If non-zero, introduces a Dropout layer on the outputs of each RNN layer except the last layer, with dropout probability equal to dropout. Default: 0
            'batch_size' : 256, # batch size
            'device': 'cuda', # 학습 환경, ["cuda", "cpu"] 중 선택
            'num_epochs' : 150 # 학습 시 사용할 epoch 수
            }
}

# Case 5. w data representation & fully-connected layers 
config5 = {
        'with_representation': True, # classification에 사용되는 representation이 있을 경우 True, 아닐 경우 False
        'model': 'FC', # classification에에 활용할 알고리즘 정의, {'RNN', 'LSTM', 'GRU', 'CNN_1D', 'FC'} 중 택 1

        'parameter': {
            'window_size' : 128, # input time series data를 windowing 하여 자르는 길이(size)
            'drop_out' : 0.1, # If non-zero, introduces a Dropout layer on the outputs of each RNN layer except the last layer, with dropout probability equal to dropout. Default: 0
            'batch_size' : 64, # batch size
            'bias': True, # bias [True, False]
            'device': 'cuda', # 학습 환경, ["cuda", "cpu"] 중 선택
            'num_epochs' : 150 # 학습 시 사용할 epoch 수
            }
}

#%%
# Raw data 
data_dir = './data'

train_x = pickle.load(open(os.path.join(data_dir, 'X_train.pkl'), 'rb'))
train_y = pickle.load(open(os.path.join(data_dir, 'y_train.pkl'), 'rb'))
test_x =  pickle.load(open(os.path.join(data_dir, 'X_test.pkl'), 'rb'))
test_y = pickle.load(open(os.path.join(data_dir, 'y_test.pkl'), 'rb'))

train_data = {'x' : train_x, 'y' : train_y}
test_data = {'x' : test_x, 'y' : test_y}

print(train_x.shape)  #shape : (num_of_instance x input_dims x window_size) = (7352, 9, 128)
print(train_y.shape) #shape : (num_of_instance) = (7352, )
print(test_x.shape)  #shape : (num_of_instance x input_dims x window_size) = (2947, 9, 128)
print(test_y.shape)  #shape : (num_of_instance) = (2947, )

#%%

# Case 4. w/o data representation & LSTM_FCNs
config = config4
data_classification = mc.Classification(config, train_data, test_data)
train_loader, valid_loader, test_loader   = data_classification.get_loaders(train_data, test_data, window_size=128, 
                                                                            batch_size=256, with_representation=True)
 
tmp = next(iter(train_loader))

 
pred, prob = data_classification.getResult()

# test_loader : shuffle = False 
print(pred[:5]) # shape : (2947, )
print(prob[:5]) # shape : (2947, 6)

print(pd.DataFrame(classification_report(test_data['y'], pred, output_dict=True)))
