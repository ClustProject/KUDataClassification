# Time Series Classification
시계열 데이터 분류
<br><br><br>
## 1. Without data representation

- DataFrame 형태의 시계열 데이터를 입력으로 활용하는 time series classification에 대한 설명.
- 입력 데이터 형태 : TXP (P>=2) 차원의 다변량 시계열 데이터(multivariate time-series data)
<br>
<br>

**time series classification 사용 시, 설정해야하는 값**

* **시계열 분류 모델 :**
  * LSTM
  * GRU
  * 1D CNN 


* **시계열 분류 모델 hyperparameter :** 아래에 자세히 설명.
  * LSTM hyperparameter 
  * GRU hyperparameter 
  * 1D CNN  hyperparameter 
<br>

```c
python time series classification.py --model='lstm' \
                                     --attention=False \
                                     --hidden_size=20 \
                                     --num_layers=2 \
                                     --dropout=0.1 \
                                     --bidirectional=False \
```
<br><br>

#### 시계열 분류 모델 hyperparameter <br>

#### 1. LSTM & GRU
- **attention** : If True, adds an attention layer to RNN. Default: False
- **hidden_size** : The number of features in the hidden state h
- **num_layers** : The number of recurrent layers. E.g., setting num_layers=2 would mean stacking two RNNs together to form a stacked RNN, with the second RNN taking in outputs of the first RNN and computing the final results. Default: 1
- **dropout** : If non-zero, introduces a Dropout layer on the outputs of each RNN layer except the last layer, with dropout probability equal to dropout. Default: 0
- **bidirectional** : If True, becomes a bidirectional RNN. Default: False
- **bias** : If False, then the layer does not use bias weights b_ih and b_hh. Default: True
 
 
 #### 2. 1D CNN
- **num_layers** : Number of convolutional layers.
- **activation** : Type of activation functions to be used. Default : relu
- **dropout** : If non-zero, introduces a Dropout layer on the outputs of each CNN layer except the last layer, with dropout probability equal to dropout. Default: 0
- **batch_norm** : If True, applies Batch Normalization after CNN layers. Default: False
- **kernel_size** : Size of the convolving kernel
- **stride** : Stride of the convolution. Default: 1
- **padding** : Padding added to both sides of the input. Default: 0
- **dilation** : Spacing between kernel elements. Default: 1
- **bias** : If True, adds a learnable bias to the output. Default: True
 

<br><br>
## 2. With data representation
- 일정한 형식의 representation을 입력으로 활용하는 classification에 대한 설명.
- 입력 데이터 형태 : P (P>=2) 차원 벡터<br>


```c
python time series classification with data representation.py --model='fc' \
                                                              --num_layers=2 \
                                                              --activation=relu \
                                                              --dropout=0.2 \
                                                              --batch_norm=True
```
<br><br>


**time series classification 사용 시, 설정해야하는 값**

* **분류 모델 :**
  * FC layers (Fully Connected layers)



* **분류 모델 hyperparameter :** 아래에 자세히 설명.
  * FC layers (Fully Connected layers)


#### 분류 모델 hyperparameter <br>

#### 1. FC layers
- **num_layers** : The number of linear layers.
- **activation** : Type of activation functions to be used. Default : relu
- **dropout** : If non-zero, introduces a Dropout layer on the outputs of each CNN layer except the last layer, with dropout probability equal to dropout. Default: 0
- **batch_norm** : If True, applies Batch Normalization after CNN layers. Default: False
- **bias** : If True, adds a learnable bias to the output. Default: True
 




