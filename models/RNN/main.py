import torch.nn as nn
import torch
import torch.optim as optim
from models.train_model import Train_Test

import numpy as np
import time
import copy

class RNN_model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, bidirectional, rnn_type='rnn', device='cuda'):
        super(RNN_model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.rnn_type = rnn_type
        self.num_directions = 2 if bidirectional == True else 1
        self.device = device
        
        # rnn_type에 따른 recurrent layer 설정
        if self.rnn_type == 'rnn':
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        elif self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        
        # bidirectional에 따른 fc layer 구축
        # bidirectional 여부에 따라 hidden state의 shape가 달라짐 (True: 2 * hidden_size, False: hidden_size)
        self.fc = nn.Linear(self.num_directions * hidden_size, self.num_classes)

    def forward(self, x):
        # data dimension: (batch_size x input_size x seq_len) -> (batch_size x seq_len x input_size)로 변환
        x = torch.transpose(x, 1, 2)
        
        # initial hidden states 설정
        h0 = torch.zeros(self.num_directions * self.num_layers, x.size(0), self.hidden_size).to(self.device)
        
        # 선택한 rnn_type의 RNN으로부터 output 도출
        if self.rnn_type in ['rnn', 'gru']:
            out, _ = self.rnn(x, h0)  # out: tensor of shape (batch_size, seq_length, hidden_size)
        else:
            # initial cell states 설정
            c0 = torch.zeros(self.num_directions * self.num_layers, x.size(0), self.hidden_size).to(self.device)
            out, _ = self.rnn(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        out = self.fc(out[:, -1, :])
        return out
    
    
class RNN_fit():
    def __init__(self, config, train_loader, valid_loader, test_loader, input_size, num_classes):
        self.config = config
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.input_size = input_size
        self.num_classes = num_classes

        self.with_representation = config['with_representation']
        self.model = self.config['model']
        self.parameter = self.config['parameter']
        
        
    def train_RNN(self):
        # representation feauture 유무 및 사용 알고리즘 모델 선언
        if self.with_representation == False:
            if self.model == 'RNN':
                model = RNN_model(input_size = self.input_size, 
                                hidden_size = self.parameter['hidden_size'],
                                num_layers = self.parameter['num_layers'], 
                                num_classes = self.num_classes, 
                                bidirectional = self.parameter['bidirectional'], 
                                rnn_type='rnn',
                                device = self.parameter['device'])
                        
            elif self.model == 'LSTM':
                model = RNN_model(input_size = self.input_size, 
                                hidden_size = self.parameter['hidden_size'],
                                num_layers = self.parameter['num_layers'], 
                                num_classes = self.num_classes, 
                                bidirectional = self.parameter['bidirectional'], 
                                rnn_type='lstm',
                                device = self.parameter['device'])

            elif self.model == 'GRU':
                model = RNN_model(input_size = self.input_size, 
                                hidden_size = self.parameter['hidden_size'],
                                num_layers = self.parameter['num_layers'], 
                                num_classes = self.num_classes, 
                                bidirectional = self.parameter['bidirectional'], 
                                rnn_type='gru',
                                device = self.parameter['device'])
            else:
                print('Please check out our chosen model')
        else:
            print('Please check whether representation rules are used')
            
        model = model.to(self.parameter['device'])
        dataloaders_dict = {'train': self.train_loader, 'val': self.valid_loader}
        criterion = nn.CrossEntropyLoss()
        optimizer=optim.Adam(model.parameters(), lr=0.0001)
        
        trainer = Train_Test(self.config, self.train_loader, self.valid_loader, self.test_loader, self.input_size, self.num_classes)
        
        best_model, val_acc_history = trainer.train(model, dataloaders_dict, criterion, self.parameter['num_epochs'], optimizer)
        return best_model
        
    def test_RNN(self, best_model):
        trainer = Train_Test(self.config, self.train_loader, self.valid_loader, self.test_loader, self.input_size, self.num_classes)
        result = trainer.test(best_model, self.test_loader)

        return result
    