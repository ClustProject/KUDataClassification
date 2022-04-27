import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
from models.train_model import Train_Test


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class LSTM_FCNs(nn.Module):
    def __init__(self, *, n_time = 128, num_classes, input_size, num_lstm_out=64, num_layers,
                conv1_nf=128, conv2_nf=128, conv3_nf=16, lstm_drop_p=0.8, fc_drop_p=0.3):
        super(LSTM_FCNs, self).__init__()
        self.n_time = n_time
        self.num_classes = num_classes
        # self.max_seq_len = max_seq_len
        self.num_features = input_size

        self.num_lstm_out = num_lstm_out
        self.num_lstm_layers = num_layers

        self.conv1_nf = conv1_nf
        self.conv2_nf = conv2_nf
        self.conv3_nf = conv3_nf

        self.lstm_drop_p = lstm_drop_p
        self.fc_drop_p = fc_drop_p

        self.lstm = nn.LSTM(input_size=self.num_features, 
                            hidden_size=self.num_lstm_out,
                            num_layers=self.num_lstm_layers,
                            batch_first=True)
        
        self.conv1 = nn.Conv1d(self.num_features, self.conv1_nf, 8)
        self.conv2 = nn.Conv1d(self.conv1_nf, self.conv2_nf, 5)
        self.conv3 = nn.Conv1d(self.conv2_nf, self.conv3_nf, 3)

        self.bn1 = nn.BatchNorm1d(self.conv1_nf)
        self.bn2 = nn.BatchNorm1d(self.conv2_nf)
        self.bn3 = nn.BatchNorm1d(self.conv3_nf)

        self.se1 = SELayer(self.conv1_nf)  # ex 128
        self.se2 = SELayer(self.conv2_nf)  # ex 256

        self.relu = nn.ReLU()
        self.lstmDrop = nn.Dropout(self.lstm_drop_p)
        self.convDrop = nn.Dropout(self.fc_drop_p)

        self.fc = nn.Linear(self.conv3_nf+self.num_lstm_out, self.num_classes)
    
    def forward(self, x):
        # input x should be in size [B,T,F] , where B = Batch size
        #                                           T = Time sampels
        #                                           F = features

        x = x.permute(0, 2, 1)

        x1, (ht,ct) = self.lstm(x)
        x1 = x1[:,-1,:]

        x2 = x.transpose(2,1)
        x2 = self.convDrop(self.relu(self.bn1(self.conv1(x2))))
        x2 = self.se1(x2)
        x2 = self.convDrop(self.relu(self.bn2(self.conv2(x2))))
        x2 = self.se2(x2)
        x2 = self.convDrop(self.relu(self.bn3(self.conv3(x2))))
        x2 = torch.mean(x2,2)
        
        x_all = torch.cat((x1,x2),dim=1)
        x_out = self.fc(x_all)
        x_out = F.log_softmax(x_out, dim=1)

        return x_out
   
class LSTM_FCNs_fit():
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
        

    def train_LSTM_FCNs(self):
        # representation feauture 유무 및 사용 알고리즘 모델 선언
        if self.with_representation == False:
            model = LSTM_FCNs(num_classes = self.num_classes,
                              input_size = self.input_size,
                              num_layers = self.parameter['num_layers'],
                              lstm_drop_p = self.parameter['lstm_drop_out'],
                              fc_drop_p = self.parameter['fc_drop_out']
                            )
        else:
            print('Please check whether representation rules are used')
            
        model = model.to(self.parameter['device'])
        dataloaders_dict = {'train': self.train_loader, 'val': self.valid_loader}
        criterion = nn.CrossEntropyLoss()
        optimizer=optim.Adam(model.parameters(), lr=0.0001)
        
        trainer = Train_Test(self.config, self.train_loader, self.valid_loader, self.test_loader, self.input_size, self.num_classes)
        
        best_model, val_acc_history = trainer.train(model, dataloaders_dict, criterion, self.parameter['num_epochs'], optimizer)
        return best_model
        
    def test_LSTM_FCNs(self, best_model):
        trainer = Train_Test(self.config, self.train_loader, self.valid_loader, self.test_loader, self.input_size, self.num_classes)
        result = trainer.test(best_model, self.test_loader)

        return result
    