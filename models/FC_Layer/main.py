import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from models.train_model import Train_Test



class FC(nn.Module):
    def __init__(self, representation_size, drop_out, num_classes, bias):
        super(FC, self).__init__()
        self.fc1 = nn.Linear(representation_size, 32, bias = bias)
        self.fc2 = nn.Linear(32, num_classes, bias = bias)
        self.layer = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            nn.Dropout(drop_out),
            self.fc2
        )

    def forward(self, x):
        x = self.layer(x)

        return x


class FC_fit():
    def __init__(self, config, train_loader, valid_loader, test_loader, representation_size, num_classes):
        self.config = config
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.representation_size = representation_size
        self.num_classes = num_classes

        self.with_representation = config['with_representation']
        self.model = self.config['model']
        self.parameter = self.config['parameter']
        
        
    def train_FC(self):
        # representation feauture 유무 및 사용 알고리즘 모델 선언
        if self.with_representation == True:
            if self.model == 'FC':
                model = FC(representation_size = self.representation_size,
                           drop_out = self.parameter['drop_out'],
                           num_classes = self.num_classes,
                           bias = self.parameter['bias']
                          )  
            else:
                print('Please check out our chosen model')
        else:
            print('Please Check whether representation rules are used')
            
        model = model.to(self.parameter['device'])
        dataloaders_dict = {'train': self.train_loader, 'val': self.valid_loader}
        criterion = nn.CrossEntropyLoss()
        optimizer=optim.Adam(model.parameters(), lr=0.0001)
        
        trainer = Train_Test(self.config, self.train_loader, self.valid_loader, self.test_loader, self.representation_size, self.num_classes)
        
        best_model, val_acc_history = trainer.train(model, dataloaders_dict, criterion, self.parameter['num_epochs'], optimizer)
        return best_model
        
    def test_FC(self, best_model):
        trainer = Train_Test(self.config, self.train_loader, self.valid_loader, self.test_loader, self.representation_size, self.num_classes)
        result = trainer.test(best_model, self.test_loader)
        
        return result