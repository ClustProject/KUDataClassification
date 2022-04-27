from models.CNN.main import *
from models.RNN.main import *
from models.FC_Layer.main import *
from models.LSTM_FCNs.main import *
import numpy as np
import torch

import warnings
warnings.filterwarnings('ignore')


class Classification():
    def __init__(self, config, train_data, test_data):
        """
        :param config: config 
        :type config: dictionary
        
        example
                    # Case 1. w/o data representation & LSTM model 
                    config1 = {
                            'with_representation': False, # classification에 사용되는 representation이 있을 경우 True, 아닐 경우 False
                            'model': 'LSTM', # classification에에 활용할 알고리즘 정의, {'RNN', 'LSTM', 'GRU', 'CONV_1D', 'FC_layer'} 중 택 1

                            'parameter': {
                                'window_size' : 50, # input time series data를 windowing 하여 자르는 길이(size)
                                'num_layers' : 2, # recurrnet layers의 수, Default : 1
                                'hidden_size' : 64, # hidden state의 벡터차원 수
                                'attention' : False, # True일 경우 attention layer를 추가
                                'dropout' : 0.2, # If non-zero, introduces a Dropout layer on the outputs of each RNN layer except the last layer, with dropout probability equal to dropout. Default: 0
                                'bidirectional' : True, # 모델의 양방향성 여부
                                'batch_size' : 64 #batch size
                                'data_dir : './data'
                                'device' : torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Detect if we have a GPU available
                                'num_epochs' : 200 # 학습 시 사용할 epoch 수
                                }
                    }
            
        """
        self.config = config
        self.train_data = train_data
        self.test_data = test_data
        
        self.with_representation = config['with_representation']
        self.input_size = self.train_data['x'].shape[1]
        if self.with_representation == True:
            self.representation_size = self.train_data['x'].shape[1]
        
        self.num_classes = len(np.unique(self.train_data['y']))
        
        self.model = config['model']
        self.parameter = config['parameter']
        

        self.train_loader, self.valid_loader, self.test_loader = self.get_loaders(train_data = self.train_data, 
                                                                                  test_data = self.test_data,
                                                                                  window_size = self.parameter['window_size'],
                                                                                  batch_size=  self.parameter['batch_size'],
                                                                                  with_representation = self.with_representation)

    def getResult(self):
        """
        getResult by classification model and data representation
        return: test set accuracy
        rtype: float
        """
        if self.with_representation == False:
            if self.model == 'LSTM_FCNs':
                result = self.LSTM_FCNs()
            elif self.model == 'LSTM':
                result = self.RNN()
            elif self.model == 'GRU':
                result = self.RNN() 
            elif self.model == 'CNN_1D':
                result = self.CNN_1D()
            else:
                print('Choose the model to use')

        elif self.with_representation == True:
            if self.model == 'FC':
                result = self.FC()
            else:
                print('Define which model to use')
        
        return result

    def RNN(self):
        RNN = RNN_fit(self.config, self.train_loader, self.valid_loader, self.test_loader, self.input_size, self.num_classes)
        best_model = RNN.train_RNN()
        result = RNN.test_RNN(best_model)
        return result
    
    def CNN_1D(self):
        CNN_1D = CNN_1D_fit(self.config, self.train_loader, self.valid_loader, self.test_loader, self.input_size, self.num_classes)
        best_model = CNN_1D.train_CNN_1D()
        result = CNN_1D.test_CNN_1D(best_model)
        return result

    def LSTM_FCNs(self):
        LSTM_FCNs = LSTM_FCNs_fit(self.config, self.train_loader, self.valid_loader, self.test_loader, self.input_size, self.num_classes)
        best_model = LSTM_FCNs.train_LSTM_FCNs()
        result = LSTM_FCNs.test_LSTM_FCNs(best_model)
        return result

    def FC(self):
        FC = FC_fit(self.config, self.train_loader, self.valid_loader, self.test_loader, self.representation_size, self.num_classes)
        best_model = FC.train_FC()
        result = FC.test_FC(best_model)
        return result
    
    def get_loaders(self, train_data, test_data, window_size, batch_size, with_representation):
        # data_dir에 있는 train/test 데이터 불러오기
        x = train_data['x']
        y = train_data['y']
        x_test = test_data['x']
        y_test = test_data['y']
        
        if np.min(y)!= 0:
            min_num = np.min(y)
            print('Set y values to zero')
            y = y - min_num
            y_test = y_test - min_num
            
        else:
            pass
                 
        # train data를 시간순으로 8:2의 비율로 train/validation set으로 분할
        n_train = int(0.8 * len(x))
        n_valid = len(x) - n_train
        n_test = len(x_test)
        x_train, y_train = x[:n_train], y[:n_train]
        x_valid, y_valid = x[n_train:], y[n_train:]

        # train/validation/test 데이터를 window_size 시점 길이로 분할
        datasets = []
        for set in [(x_train, y_train, n_train), (x_valid, y_valid, n_valid), (x_test, y_test, n_test)]:
            if with_representation == False:
                T = set[0].shape[-1]
                windows = np.split(set[0][:, :, :window_size * (T // window_size)], (T // window_size), -1)
                windows = np.concatenate(windows, 0)
                labels = set[1] # 묶여있는 window 관측치 하나마다 y_label 값이 하나씩 달려있는 dataset 이므로
                datasets.append(torch.utils.data.TensorDataset(torch.Tensor(windows), torch.Tensor(labels)))

            elif with_representation == True:
                labels = set[1] # 묶여있는 window 관측치 하나마다 y_label 값이 하나씩 달려있는 dataset 이므로
                representation = set[0]
                representation = np.array(representation)
                datasets.append(torch.utils.data.TensorDataset(torch.Tensor(representation), torch.Tensor(labels)))

        # train/validation/test DataLoader 구축
        trainset, validset, testset = datasets[0], datasets[1], datasets[2]
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

        return train_loader, valid_loader, test_loader  