import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from models.train_model import Train_Test
from models.lstm_fcn import LSTM_FCNs
from models.rnn import RNN_model
from models.cnn_1d import CNN_1D
from models.fc import FC

import warnings
warnings.filterwarnings('ignore')


class Classification():
    def __init__(self, config, train_data, test_data):
        """
        Initialize Classification class and prepare dataloaders for training and testing

        :param config: config
        :type config: dictionary

        :param train_data: train data with X and y
        :type train_data: dictionary

        :param test_data: test data with X and y
        :type test_data: dictionary
        
        example
            >>> config1 = {
                    'model': 'LSTM', # classification에에 활용할 알고리즘 정의, {'LSTM', 'GRU', 'CNN_1D', 'LSTM_FCNs', 'FC_layer'} 중 택 1
                    'training': True,  # 학습 여부, 저장된 학습 완료 모델 존재시 False로 설정
                    'best_model_path': './ckpt/lstm.pt',  # 학습 완료 모델 저장 경로
                    'parameter': {
                        'input_size': 9,  # 데이터의 변수 개수, int
                        'num_classes': 6,  # 분류할 class 개수, int
                        'num_layers': 1,  # recurrent layers의 수, int(default: 1, 범위: 1 이상)
                        'hidden_size': 64,  # hidden state의 차원, int(default: 64, 범위: 1 이상)
                        'dropout': 0.1,  # dropout 확률, float(default: 0.1, 범위: 0 이상 1 이하)
                        'bidirectional': True,  # 모델의 양방향성 여부, bool(default: True)
                        'num_epochs': 100,  # 학습 epoch 횟수, int(default: 100, 범위: 1 이상)
                        'batch_size': 512,  # batch 크기, int(default: 512, 범위: 1 이상, 컴퓨터 사양에 적합하게 설정)
                        'lr': 0.0001,  # learning rate, float(default: 0.001, 범위: 0.1 이하)
                        'device': 'cuda'  # 학습 환경, ["cuda", "cpu"] 중 선택
                    }
                }
            >>> data_cls = mc.Classification(config, train_data, test_data)
            >>> model = data_cls.build_model()  # 모델 구축
            >>> if config["training"]:
            >>>     best_model = data_cls.train_model(model)  # 모델 학습
            >>>     data_cls.save_model(best_model, best_model_path=config["best_model_path"])  # 모델 저장
            >>> pred, prob, acc = data_cls.pred_data(model, best_model_path=config["best_model_path"])  # class 예측
        """

        self.config = config
        self.model = config['model']
        self.parameter = config['parameter']

        self.train_data = train_data
        self.test_data = test_data

        # load dataloder
        self.train_loader, self.valid_loader, self.test_loader = self.get_loaders(train_data=self.train_data,
                                                                                  test_data=self.test_data,
                                                                                  batch_size=self.parameter['batch_size'])

        # build trainer
        self.trainer = Train_Test(self.config, self.train_loader, self.valid_loader, self.test_loader)

    def build_model(self):
        """
        Build model and return initialized model for selected model_name

        :return: initialized model
        :rtype: model
        """

        # build initialized model
        if self.model == 'LSTM':
            init_model = RNN_model(
                rnn_type='lstm',
                input_size=self.parameter['input_size'],
                num_classes=self.parameter['num_classes'],
                hidden_size=self.parameter['hidden_size'],
                num_layers=self.parameter['num_layers'],
                bidirectional=self.parameter['bidirectional'],
                device=self.parameter['device']
            )
        elif self.model == 'GRU':
            init_model = RNN_model(
                rnn_type='gru',
                input_size=self.parameter['input_size'],
                num_classes=self.parameter['num_classes'],
                hidden_size=self.parameter['hidden_size'],
                num_layers=self.parameter['num_layers'],
                bidirectional=self.parameter['bidirectional'],
                device=self.parameter['device']
            )
        elif self.model == 'CNN_1D':
            init_model = CNN_1D(
                input_channels=self.parameter['input_size'],
                num_classes=self.parameter['num_classes'],
                input_seq=self.parameter['seq_len'],
                output_channels=self.parameter['output_channels'],
                kernel_size=self.parameter['kernel_size'],
                stride=self.parameter['stride'],
                padding=self.parameter['padding'],
                drop_out=self.parameter['drop_out']
            )
        elif self.model == 'LSTM_FCNs':
            init_model = LSTM_FCNs(
                input_size=self.parameter['input_size'],
                num_classes=self.parameter['num_classes'],
                num_layers=self.parameter['num_layers'],
                lstm_drop_p=self.parameter['lstm_drop_out'],
                fc_drop_p=self.parameter['fc_drop_out']
            )
        elif self.model == 'FC':
            init_model = FC(
                representation_size=self.parameter['input_size'],
                num_classes=self.parameter['num_classes'],
                drop_out=self.parameter['drop_out'],
                bias=self.parameter['bias']
            )
        else:
            print('Choose the model correctly')
        return init_model

    def train_model(self, init_model):
        """
        Train model and return best model

        :param init_model: initialized model
        :type init_model: model

        :return: best trained model
        :rtype: model
        """

        print("Start training model")

        # train model
        init_model = init_model.to(self.parameter['device'])

        dataloaders_dict = {'train': self.train_loader, 'val': self.valid_loader}
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(init_model.parameters(), lr=self.parameter['lr'])

        best_model = self.trainer.train(init_model, dataloaders_dict, criterion, self.parameter['num_epochs'], optimizer)
        return best_model

    def save_model(self, best_model, best_model_path):
        """
        Save the best trained model

        :param best_model: best trained model
        :type best_model: model

        :param best_model_path: path for saving model
        :type best_model_path: str
        """

        # save model
        torch.save(best_model.state_dict(), best_model_path)

    def pred_data(self, init_model, best_model_path):
        """
        Predict class based on the best trained model
        :param init_model: initialized model
        :type model: model

        :param best_model_path: path for loading the best trained model
        :type best_model_path: str

        :return: predicted classes
        :rtype: numpy array

        :return: prediction probabilities
        :rtype: numpy array

        :return: test accuracy
        :rtype: float
        """

        print("\nStart testing data\n")

        # load best model
        init_model.load_state_dict(torch.load(best_model_path))

        # get prediction and accuracy
        pred, prob, acc = self.trainer.test(init_model, self.test_loader)
        return pred, prob, acc
    
    def get_loaders(self, train_data, test_data, batch_size):
        """
        Get train, validation, and test DataLoaders
        
        :param train_data: train data with X and y
        :type train_data: dictionary

        :param test_data: test data with X and y
        :type test_data: dictionary

        :param batch_size: batch size
        :type batch_size: int

        :return: train, validation, and test dataloaders
        :rtype: DataLoader
        """

        x = train_data['x']
        y = train_data['y']
        x_test = test_data['x']
        y_test = test_data['y']
        
        # class의 값이 0부터 시작하지 않으면 0부터 시작하도록 변환
        if np.min(y) != 0:
            min_num = np.min(y)
            print('Set start class as zero')
            y = y - min_num
            y_test = y_test - min_num
        else:
            pass

        # train data를 시간순으로 8:2의 비율로 train/validation set으로 분할
        n_train = int(0.8 * len(x))
        x_train, y_train = x[:n_train], y[:n_train]
        x_valid, y_valid = x[n_train:], y[n_train:]

        # train/validation/test 데이터셋 구축
        datasets = []
        for dataset in [(x_train, y_train), (x_valid, y_valid), (x_test, y_test)]:
            x_data = np.array(dataset[0])
            y_data = dataset[1]
            datasets.append(torch.utils.data.TensorDataset(torch.Tensor(x_data), torch.Tensor(y_data)))

        # train/validation/test DataLoader 구축
        trainset, validset, testset = datasets[0], datasets[1], datasets[2]
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
        return train_loader, valid_loader, test_loader