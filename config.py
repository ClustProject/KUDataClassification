model_config = {
    'LSTM' : {  # Case 1. LSTM model (w/o data representation)
            'model': 'LSTM', 
            'best_model': 'lstm.pt',  # 학습 완료 모델 저장 경로
            'parameter': {
                'input_size': 9,  # 데이터의 변수 개수, int
                'num_classes': 6,  # 분류할 class 개수, int
                'num_layers': 2,  # recurrent layers의 수, int(default: 2, 범위: 1 이상)
                'hidden_size': 64,  # hidden state의 차원, int(default: 64, 범위: 1 이상)
                'dropout': 0.1,  # dropout 확률, float(default: 0.1, 범위: 0 이상 1 이하)
                'bidirectional': True,  # 모델의 양방향성 여부, bool(default: True)
                'num_epochs': 150,  # 학습 epoch 횟수, int(default: 150, 범위: 1 이상)
                'batch_size': 64,  # batch 크기, int(default: 64, 범위: 1 이상, 컴퓨터 사양에 적합하게 설정)
                'lr': 0.0001,  # learning rate, float(default: 0.001, 범위: 0.1 이하)
                'device': 'cuda'  # 학습 환경, ["cuda", "cpu"] 중 선택
            }
    },
    'GRU' : {  # Case 2. GRU model (w/o data representation)
        'model': 'GRU',
        'best_model': 'gru.pt',  # 학습 완료 모델 저장 경로
        'parameter': {
            'input_size': 9,  # 데이터의 변수 개수, int
            'num_classes': 6,  # 분류할 class 개수, int
            'num_layers': 2,  # recurrent layers의 수, int(default: 2, 범위: 1 이상)
            'hidden_size': 64,  # hidden state의 차원, int(default: 64, 범위: 1 이상)
            'dropout': 0.1,  # dropout 확률, float(default: 0.1, 범위: 0 이상 1 이하)
            'bidirectional': True,  # 모델의 양방향성 여부, bool(default: True)
            'num_epochs': 150,  # 학습 epoch 횟수, int(default: 150, 범위: 1 이상)
            'batch_size': 64,  # batch 크기, int(default: 64, 범위: 1 이상, 컴퓨터 사양에 적합하게 설정)
            'lr': 0.0001,  # learning rate, float(default: 0.001, 범위: 0.1 이하)
            'device': 'cuda'  # 학습 환경, ["cuda", "cpu"] 중 선택
        }
    },
    'CNN_1D' : {  # Case 3. CNN_1D model (w/o data representation)
        'model': 'CNN_1D', 
        'best_model': 'cnn_1d.pt',  # 학습 완료 모델 저장 경로
        'parameter': {
            'input_size': 9,  # 데이터의 변수 개수, int
            'num_classes': 6,  # 분류할 class 개수, int
            'seq_len': 128,  # 데이터의 시간 길이, int
            'output_channels': 64, # convolution layer의 output channel, int(default: 64, 범위: 1 이상, 2의 지수로 설정 권장)
            'kernel_size': 3, # convolutional layer의 filter 크기, int(default: 3, 범위: 3 이상, 홀수로 설정 권장)
            'stride': 1, # convolution layer의 stride 크기, int(default: 1, 범위: 1 이상)
            'padding': 0, # padding 크기, int(default: 0, 범위: 0 이상)
            'drop_out': 0.1, # dropout 확률, float(default: 0.1, 범위: 0 이상 1 이하)
            'num_epochs': 150,  # 학습 epoch 횟수, int(default: 150, 범위: 1 이상)
            'batch_size': 64,  # batch 크기, int(default: 64, 범위: 1 이상, 컴퓨터 사양에 적합하게 설정)
            'lr': 0.0001,  # learning rate, float(default: 0.0001, 범위: 0.1 이하)
            'device': 'cuda'  # 학습 환경, ["cuda", "cpu"] 중 선택
        }
    },
    'LSTM_FCNs' : {  # Case 4. LSTM_FCNs model (w/o data representation)
        'model': 'LSTM_FCNs', 
        'best_model': 'lstm_fcn.pt',  # 학습 완료 모델 저장 경로
        'parameter': {
            'input_size': 9,  # 데이터의 변수 개수, int
            'num_classes': 6,  # 분류할 class 개수, int
            'num_layers': 1,  # recurrent layers의 수, int(default: 1, 범위: 1 이상)
            'lstm_drop_out': 0.4, # LSTM dropout 확률, float(default: 0.4, 범위: 0 이상 1 이하)
            'fc_drop_out': 0.1, # FC dropout 확률, float(default: 0.1, 범위: 0 이상 1 이하)
            'num_epochs': 150, # 학습 epoch 횟수, int(default: 150, 범위: 1 이상)
            'batch_size': 64,  # batch 크기, int(default: 64, 범위: 1 이상, 컴퓨터 사양에 적합하게 설정)
            'lr': 0.0001,  # learning rate, float(default: 0.0001, 범위: 0.1 이하)
            'device': 'cuda'  # 학습 환경, ["cuda", "cpu"] 중 선택
        }
    },
    'FC' : {  # Case 5. fully-connected layers (w/ data representation)
        'model': 'FC', 
        "best_model": 'fc.pt',  # 학습 완료 모델 저장 경로
        'parameter': {
            'input_size': 64,  # 데이터의 변수 개수(representation 차원), int
            'num_classes': 6,  # 분류할 class 개수, int
            'drop_out': 0.1, # dropout 확률, float(default: 0.1, 범위: 0 이상 1 이하)
            'bias': True, # bias 사용 여부, bool(default: True)
            'num_epochs': 150, # 학습 epoch 횟수, int(default: 150, 범위: 1 이상)
            'batch_size': 64,  # batch 크기, int(default: 64, 범위: 1 이상, 컴퓨터 사양에 적합하게 설정)
            'lr': 0.0001,  # learning rate, float(default: 0.0001, 범위: 0.1 이하)
            'device': 'cuda'  # 학습 환경, ["cuda", "cpu"] 중 선택
        }
    }
}
