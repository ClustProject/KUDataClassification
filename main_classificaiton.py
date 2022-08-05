import torch
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

from models.train_model import Train_Test
from models.lstm_fcn import LSTM_FCNs
from models.rnn import RNN_model
from models.cnn_1d import CNN_1D
from models.fc import FC

import warnings
warnings.filterwarnings('ignore')


class Classification():
    def __init__(self, config):
        """
        Initialize Classification class

        :param config: config
        :type config: dictionary

        example (training)
            >>> model_name = 'LSTM'
            >>> model_params = config.model_config[model_name]
            >>> data_cls = mc.Classification(model_params)
            >>> best_model = data_cls.train_model(train_x, train_y, valid_x, valid_y)  # 모델 학습
            >>> data_cls.save_model(best_model, best_model_path=model_params["best_model_path"])  # 모델 저장
        
        example (testing)

        """

        self.model_name = config['model']
        self.parameter = config['parameter']

        # build trainer
        self.trainer = Train_Test(config)

    def build_model(self):
        """
        Build model and return initialized model for selected model_name

        :return: initialized model
        :rtype: model
        """

        # build initialized model
        if self.model_name == 'LSTM':
            init_model = RNN_model(
                rnn_type='lstm',
                input_size=self.parameter['input_size'],
                num_classes=self.parameter['num_classes'],
                hidden_size=self.parameter['hidden_size'],
                num_layers=self.parameter['num_layers'],
                bidirectional=self.parameter['bidirectional'],
                device=self.parameter['device']
            )
        elif self.model_name == 'GRU':
            init_model = RNN_model(
                rnn_type='gru',
                input_size=self.parameter['input_size'],
                num_classes=self.parameter['num_classes'],
                hidden_size=self.parameter['hidden_size'],
                num_layers=self.parameter['num_layers'],
                bidirectional=self.parameter['bidirectional'],
                device=self.parameter['device']
            )
        elif self.model_name == 'CNN_1D':
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
        elif self.model_name == 'LSTM_FCNs':
            init_model = LSTM_FCNs(
                input_size=self.parameter['input_size'],
                num_classes=self.parameter['num_classes'],
                num_layers=self.parameter['num_layers'],
                lstm_drop_p=self.parameter['lstm_drop_out'],
                fc_drop_p=self.parameter['fc_drop_out']
            )
        elif self.model_name == 'FC':
            init_model = FC(
                representation_size=self.parameter['input_size'],
                num_classes=self.parameter['num_classes'],
                drop_out=self.parameter['drop_out'],
                bias=self.parameter['bias']
            )
        else:
            print('Choose the model correctly')
        return init_model

    def train_model(self, train_x, train_y, valid_x, valid_y):
        """
        Train model and return best model

        :param train_x: input train data 
        :type train_x: numpy array

        :param train_y: target train data 
        :type train_y: numpy array
        
        :param valid_x: input validation data 
        :type valid_x: numpy array

        :param valid_y: target validation data 
        :type valid_y: numpy array

        :return: best trained model
        :rtype: model
        """

        print(f"Start training model: {self.model_name}")

        # build train/validation dataloaders
        train_loader = self.get_dataloader(train_x, train_y, self.parameter['batch_size'], shuffle=True)
        valid_loader = self.get_dataloader(valid_x, valid_y, self.parameter['batch_size'], shuffle=False)

        # build initialized model
        init_model = self.build_model()
        
        # train model
        dataloaders_dict = {'train': train_loader, 'val': valid_loader}
        best_model = self.trainer.train(init_model, dataloaders_dict)
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

    def pred_data(self, test_x, test_y, best_model_path):
        """
        Predict target class based on the best trained model

        :param test_x: input test data
        :type test_x: numpy array

        :param test_y: target test data
        :type test_y: numpy array

        :param best_model_path: path for loading the best trained model
        :type best_model_path: str

        :return: actual and predicted classes
        :rtype: DataFrame

        :return: test accuracy
        :rtype: float
        """

        print(f"Start testing model: {self.model_name}")

        # build test dataloader
        test_loader = self.get_dataloader(test_x, test_y, self.parameter['batch_size'], shuffle=False)

        # build initialized model
        init_model = self.build_model()

        # load best model
        init_model.load_state_dict(torch.load(best_model_path))

        # get predicted classes
        pred_data = self.trainer.test(init_model, test_loader)

        # class의 값이 0부터 시작하지 않으면 0부터 시작하도록 변환
        if np.min(test_y) != 0:
            print('Set start class as zero')
            test_y = test_y - np.min(test_y)

        # calculate performance metrics
        acc = accuracy_score(test_y, pred_data)
        
        # merge true value and predicted value
        pred_df = pd.DataFrame()
        pred_df['actual_value'] = test_y
        pred_df['predicted_value'] = pred_data
        return pred_df, acc
    
    def get_dataloader(self, x_data, y_data, batch_size, shuffle):
        """
        Get DataLoader
        
        :param x_data: input data
        :type x_data: numpy array

        :param y_data: target data
        :type y_data: numpy array

        :param batch_size: batch size
        :type batch_size: int

        :param shuffle: shuffle for making batch
        :type shuffle: bool

        :return: dataloader
        :rtype: DataLoader
        """

        # class의 값이 0부터 시작하지 않으면 0부터 시작하도록 변환
        if np.min(y_data) != 0:
            print('Set start class as zero')
            y_data = y_data - np.min(y_data)

        # torch dataset 구축
        dataset = torch.utils.data.TensorDataset(torch.Tensor(x_data), torch.Tensor(y_data))

        # DataLoader 구축
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return data_loader