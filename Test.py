import torch
import pickle
import random
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

import main_classificaiton as mc

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
        'model': 'LSTM', # classification에에 활용할 알고리즘 정의, {'LSTM', 'GRU', 'CNN_1D', 'LSTM_FCNs', 'FC_layer'} 중 택 1
        "training": True,  # 학습 여부, 저장된 학습 완료 모델 존재시 False로 설정
        "best_model_path": './ckpt/lstm.pt',  # 학습 완료 모델 저장 경로
        'parameter': {
            'input_size': 9,
            'num_classes': 6,
            'num_layers' : 2, # recurrnet layers의 수, Default : 1
            'hidden_size' : 64, # hidden state의 벡터차원 수
            'dropout' : 0.1, # If non-zero, introduces a Dropout layer on the outputs of each RNN layer except the last layer, with dropout probability equal to dropout. Default: 0
            'bidirectional' : True, # 모델의 양방향성 여부
            'batch_size' : 64, #batch size
            'device': 'cuda', # 학습 환경, ["cuda", "cpu"] 중 선택
            'num_epochs' : 10, # 학습 시 사용할 epoch 수
            'lr': 0.0001
            }
}

# Case 2. w/o data representation & GRU model 
config2 = {
        'model': 'GRU', # classification에에 활용할 알고리즘 정의, {'LSTM', 'GRU', 'CNN_1D', 'LSTM_FCNs', 'FC_layer'} 중 택 1
        "training": True,  # 학습 여부, 저장된 학습 완료 모델 존재시 False로 설정
        "best_model_path": './ckpt/gru.pt',  # 학습 완료 모델 저장 경로
        'parameter': {
            'input_size': 9,
            'num_classes': 6,
            'num_layers' : 2, # recurrnet layers의 수, Default : 1
            'hidden_size' : 64, # hidden state의 벡터차원 수
            'dropout' : 0.1, # If non-zero, introduces a Dropout layer on the outputs of each RNN layer except the last layer, with dropout probability equal to dropout. Default: 0
            'bidirectional' : True, # 모델의 양방향성 여부
            'batch_size' : 64, #batch size
            'device': 'cuda', # 학습 환경, ["cuda", "cpu"] 중 선택
            'num_epochs' : 10, # 학습 시 사용할 epoch 수
            'lr': 0.0001
            }
}

# Case 3. w/o data representation & CNN_1D model 
config3 = {
        'model': 'CNN_1D', # classification에에 활용할 알고리즘 정의, {'LSTM', 'GRU', 'CNN_1D', 'LSTM_FCNs', 'FC_layer'} 중 택 1
        "training": True,  # 학습 여부, 저장된 학습 완료 모델 존재시 False로 설정
        "best_model_path": './ckpt/cnn_1d.pt',  # 학습 완료 모델 저장 경로
        'parameter': {
            'input_size': 9,
            'num_classes': 6,
            'seq_len': 128,
            'output_channels' : 64, # convolution channel size of output
            'drop_out' : 0.1, # If non-zero, introduces a Dropout layer on the outputs of each RNN layer except the last layer, with dropout probability equal to dropout. Default: 0
            'kernel_size' : 3, # convolutional filter size
            'stride' : 1, # stride of the convolution. Default = 1 
            'padding' : 0, # padding added to both sides of the input. Default = 0
            'batch_size' : 64, # batch size
            'device': 'cuda', # 학습 환경, ["cuda", "cpu"] 중 선택
            'num_epochs' : 10, # 학습 시 사용할 epoch 수
            'lr': 0.0001
            }
}

# Case 4. w/o data representation & LSTM_FCNs model 
config4 = {
        'model': 'LSTM_FCNs', # classification에에 활용할 알고리즘 정의, {'LSTM', 'GRU', 'CNN_1D', 'LSTM_FCNs', 'FC_layer'} 중 택 1
        "training": True,  # 학습 여부, 저장된 학습 완료 모델 존재시 False로 설정
        "best_model_path": './ckpt/lstm_fcn.pt',  # 학습 완료 모델 저장 경로
        'parameter': {
            'input_size': 9,
            'num_classes': 6,
            'num_layers' : 1, # recurrnet layers의 수, Default : 1
            'lstm_drop_out' : 0.4, # If non-zero, introduces a Dropout layer on the outputs of each RNN layer except the last layer, with dropout probability equal to dropout. Default: 0
            'fc_drop_out' : 0.1, # If non-zero, introduces a Dropout layer on the outputs of each RNN layer except the last layer, with dropout probability equal to dropout. Default: 0
            'batch_size' : 256, # batch size
            'device': 'cuda', # 학습 환경, ["cuda", "cpu"] 중 선택
            'num_epochs' : 10, # 학습 시 사용할 epoch 수
            'lr': 0.0001
            }
}

# Case 5. w data representation & fully-connected layers 
config5 = {
        'model': 'FC', # classification에에 활용할 알고리즘 정의, {'RNN', 'LSTM', 'GRU', 'CNN_1D', 'FC'} 중 택 1
        "training": True,  # 학습 여부, 저장된 학습 완료 모델 존재시 False로 설정
        "best_model_path": './ckpt/gru.pt',  # 학습 완료 모델 저장 경로
        'parameter': {
            'input_size': 64,
            'num_classes': 6,
            'drop_out' : 0.1, # If non-zero, introduces a Dropout layer on the outputs of each RNN layer except the last layer, with dropout probability equal to dropout. Default: 0
            'batch_size' : 64, # batch size
            'bias': True, # bias [True, False]
            'device': 'cuda', # 학습 환경, ["cuda", "cpu"] 중 선택
            'num_epochs' : 10, # 학습 시 사용할 epoch 수
            'lr': 0.0001
            }
}

#%%
# Raw data 
data_dir = './data'

train_x = pickle.load(open('./data/x_train.pkl', 'rb'))
train_y = pickle.load(open('./data/y_train.pkl', 'rb'))
test_x = pickle.load(open('./data/x_test.pkl', 'rb'))
test_y = pickle.load(open('./data/y_test.pkl', 'rb'))

train_data = {'x': train_x, 'y': train_y}
test_data = {'x': test_x, 'y': test_y}

config = config4
data_cls = mc.Classification(config, train_data, test_data)
model = data_cls.build_model()  # 모델 구축

if config["training"]:
    best_model = data_cls.train_model(model)  # 모델 학습
    data_cls.save_model(best_model, best_model_path=config["best_model_path"])  # 모델 저장

pred, prob, acc = data_cls.pred_data(model, best_model_path=config["best_model_path"])  # representation 도출
print(acc)


train_x = pd.read_csv('./data/ts2vec_repr_train.csv')
train_y = pickle.load(open('./data/y_train.pkl', 'rb'))

test_x = pd.read_csv('./data/ts2vec_repr_test.csv')
test_y = pickle.load(open('./data/y_test.pkl', 'rb'))

train_data = {'x': train_x, 'y': train_y}
test_data = {'x': test_x, 'y': test_y}

config = config5
data_cls = mc.Classification(config, train_data, test_data)
model = data_cls.build_model()  # 모델 구축

if config["training"]:
    best_model = data_cls.train_model(model)  # 모델 학습
    data_cls.save_model(best_model, best_model_path=config["best_model_path"])  # 모델 저장

pred, prob, acc = data_cls.pred_data(model, best_model_path=config["best_model_path"])  # representation 도출