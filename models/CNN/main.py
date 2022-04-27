import torch.nn as nn
import torch.optim as optim
from models.train_model import Train_Test

def calculate_output_length(input_seq, kernel_size, stride=1, padding=0, dilation=1):
    return (input_seq + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

class CNN_1D(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding, drop_out, input_seq, num_classes):
        super(CNN_1D, self).__init__()
        # 첫 번째 1-dimensional convolution layer 구축
        self.layer1 = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size = kernel_size, stride = stride, padding = padding),
            nn.ReLU(),
            nn.AvgPool1d(2)
        )
        
        next_seq = calculate_output_length(input_seq, kernel_size, stride, padding, dilation=1)
        next_seq = next_seq // 2
        
        # 두 번째 1-dimensional convolution layer 구축
        self.layer2 = nn.Sequential(
            nn.Conv1d(32, output_channels, kernel_size = kernel_size, stride = stride, padding = padding),
            nn.ReLU(),
            nn.AvgPool1d(2)
        )
        
        next_seq = calculate_output_length(next_seq, kernel_size, stride, padding, dilation=1)
        next_seq = next_seq // 2
        
        # fully-connected layer 구축
        self.dropout = nn.Dropout(drop_out)
        self.fc = nn.Linear(output_channels * next_seq, num_classes) # 이부분은 hyperparameter에 따라 계산을 해줘야 함


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class CNN_1D_fit():
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
                
    def train_CNN_1D(self):
        # representation feauture 유무 및 사용 알고리즘 모델 선언
        if self.with_representation == False:
            if self.model == 'CNN_1D':
                model = CNN_1D(input_channels = self.input_size,
                                output_channels = self.parameter['output_channels'],
                                kernel_size = self.parameter['kernel_size'],
                                stride = self.parameter['stride'],
                                padding = self.parameter['padding'],
                                drop_out = self.parameter['drop_out'],
                                input_seq = self.parameter['window_size'],
                                num_classes = self.num_classes
                                )
            else:
                print('Please check out our chosen model')
        else:
            print('Please Check whether representation rules are used')
            
        model = model.to(self.parameter['device'])
        # summary(model)
        dataloaders_dict = {'train': self.train_loader, 'val': self.valid_loader}
        criterion = nn.CrossEntropyLoss()
        optimizer=optim.Adam(model.parameters(), lr=0.0001)
        
        trainer = Train_Test(self.config, self.train_loader, self.valid_loader, self.test_loader, self.input_size, self.num_classes)
        
        best_model, val_acc_history = trainer.train(model, dataloaders_dict, criterion, self.parameter['num_epochs'], optimizer)
        return best_model
        
    def test_CNN_1D(self, best_model):
        trainer = Train_Test(self.config, self.train_loader, self.valid_loader, self.test_loader, self.input_size, self.num_classes)
        result = trainer.test(best_model, self.test_loader)
        
        return result