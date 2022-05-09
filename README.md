# Time Series Classification

## 1. Without data representation

- 원본 시계열 데이터를 입력으로 활용하는 time series classification에 대한 설명
- 입력 데이터 형태 : (num_of_instance x input_dims x seq_len) 차원의 다변량 시계열 데이터(multivariate time-series data)
<br>

**Time series classification 사용 시, 설정해야하는 값**
* **model** : ['LSTM', 'GRU', 'CNN_1D', 'LSTM_FCNs'] 중 선택
* **training** : 모델 학습 여부, [True, False] 중 선택, 학습 완료된 모델이 저장되어 있다면 False 선택
* **best_model_path** : 학습 완료된 모델을 저장할 경로

* **시계열 분류 모델 hyperparameter :** 아래에 자세히 설명.
  * LSTM hyperparameter 
  * GRU hyperparameter 
  * 1D-CNN hyperparameter
  * LSTM_FCNs hyperparameter
<br>

#### 시계열 분류 모델 hyperparameter <br>

#### 1. LSTM & GRU (w/o data representation)
- **input_size** : 데이터의 변수 개수, int
- **num_classes** : 분류할 class 개수, int
- **num_layers** : recurrent layers의 수, int(default: 2, 범위: 1 이상)
- **hidden_size** : hidden state의 차원, int(default: 64, 범위: 1 이상)
- **dropout** : dropout 확률, float(default: 0.1, 범위: 0 이상 1 이하)
- **bidirectional** : 모델의 양방향성 여부, bool(default: True)
- **num_epochs** : 학습 epoch 횟수, int(default: 150, 범위: 1 이상)
- **batch_size** : batch 크기, int(default: 64, 범위: 1 이상, 컴퓨터 사양에 적합하게 설정)
- **lr** : learning rate, float(default: 0.0001, 범위: 0.1 이하)
- **device** : 학습 환경, (default: 'cuda', ['cuda', 'cpu'] 중 선택)
<br>

#### 2. 1D CNN (w/o data representation)
- **input_size** : 데이터의 변수 개수, int
- **num_classes** : 분류할 class 개수, int
- **seq_len** : 데이터의 시간 길이, int
- **output_channels** : convolution layer의 output channel, int(default: 64, 범위: 1 이상, 2의 지수로 설정 권장)
- **kernel_size** : convolutional layer의 filter 크기, int(default: 3, 범위: 3 이상, 홀수로 설정 권장)
- **stride** : convolution layer의 stride 크기, int(default: 1, 범위: 1 이상)
- **padding** : padding 크기, int(default: 0, 범위: 0 이상)
- **dropout** : dropout 확률, float(default: 0.1, 범위: 0 이상 1 이하)
- **num_epochs** : 학습 epoch 횟수, int(default: 150, 범위: 1 이상)
- **batch_size** : batch 크기, int(default: 64, 범위: 1 이상, 컴퓨터 사양에 적합하게 설정)
- **lr** : learning rate, float(default: 0.0001, 범위: 0.1 이하)
- **device** : 학습 환경, (default: 'cuda', ['cuda', 'cpu'] 중 선택)
<br>

#### 3.	LSTM-FCNs (w/o data representation)
- **input_size** : 데이터의 변수 개수, int
- **num_classes** : 분류할 class 개수, int
- **num_layers** : recurrent layers의 수, int(default: 1, 범위: 1 이상)
- **lstm_drop_out** : LSTM dropout 확률, float(default: 0.4, 범위: 0 이상 1 이하)
- **fc_drop_out** : FC dropout 확률, float(default: 0.1, 범위: 0 이상 1 이하)
- **num_epochs** : 학습 epoch 횟수, int(default: 150, 범위: 1 이상)
- **batch_size** : batch 크기, int(default: 64, 범위: 1 이상, 컴퓨터 사양에 적합하게 설정)
- **lr** : learning rate, float(default: 0.0001, 범위: 0.1 이하)
- **device** : 학습 환경, (default: 'cuda', ['cuda', 'cpu'] 중 선택)
 

<br><br>
## 2. With data representation
- 원본 시계열 데이터를 representation vector로 변환한 데이터를 입력으로 활용하는 time series classification에 대한 설명
- 입력 데이터 형태 : (num_of_instance x input_dims) 차원의 다변량 시계열 데이터(multivariate time-series data)
<br>

**Time series classification 사용 시, 설정해야하는 값**
* **model** : 'FC' 선택
* **training** : 모델 학습 여부, [True, False] 중 선택, 학습 완료된 모델이 저장되어 있다면 False 선택
* **best_model_path** : 학습 완료된 모델을 저장할 경로

* **시계열 분류 모델 hyperparameter :** 아래에 자세히 설명.
  * FC hyperparameter 
<br>

#### 데이터 표상 분류 모델 hyperparameter <br>

#### 1. LSTM & GRU (w/o data representation)
- **input_size** : 데이터의 변수 개수, int
- **num_classes** : 분류할 class 개수, int
- **dropout** : dropout 확률, float(default: 0.1, 범위: 0 이상 1 이하)
- **bias** : bias 사용 여부, bool(default: True)
- **num_epochs** : 학습 epoch 횟수, int(default: 150, 범위: 1 이상)
- **batch_size** : batch 크기, int(default: 64, 범위: 1 이상, 컴퓨터 사양에 적합하게 설정)
- **lr** : learning rate, float(default: 0.0001, 범위: 0.1 이하)
- **device** : 학습 환경, (default: 'cuda', ['cuda', 'cpu'] 중 선택)
<br>
