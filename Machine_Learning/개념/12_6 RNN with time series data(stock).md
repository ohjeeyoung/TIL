# RNN with time series data(stock)

## Time series data

> 시간에 따라서 값이 변하는 데이터

ex. 주식

#### Many to one

<img width="1000" alt="스크린샷 2020-12-14 오전 4 05 24" src="https://user-images.githubusercontent.com/62995632/102021222-9b64f780-3dc1-11eb-86fd-fcc4c3589fa8.png">

7일 동안의 data를 가지고 있다면 8일째의 값이 궁금한 것

이전 것들이 다음 값에 영향을 미친다는 것이 time series의 기본 전제

<img width="1000" alt="스크린샷 2020-12-14 오전 4 07 45" src="https://user-images.githubusercontent.com/62995632/102021256-f0a10900-3dc1-11eb-9a5e-9596d628f4b8.png">

input_dim = 5(open, high, low, volume, close), seq_len = 7(7일동안의 데이터), output_dim = 1(8일째 알고싶은 값)

#### Reading data

<img width="1000" alt="스크린샷 2020-12-14 오전 4 09 14" src="https://user-images.githubusercontent.com/62995632/102021284-247c2e80-3dc2-11eb-94d6-916bcf4d62b7.png">

```python
timesteps = seq_length = 7
data_dim = 5
output_dim = 1

# Open, High, Low, Close, Volume
xy = np.loadtxt('data-02-stock-daily.csv', delimiter=',')
xy = xy[::-1]   # reverse order (chronically ordered)
xy = MinMaxScaler(xy)
x = xy
y = xy[:, [-1]] # Close as label

dataX = []
dataY = []
for i in range(0, len(y) - seq_length):
    _x = x[i: i + seq_length]
    _y = y[i + seq_length]  # Next close price
    print(_x, "->", y)
    dataX.append(_x)
    dataY.append(_y)
```

#### Training and test datasets

```python
# split to train and testing
train_size = int(len(dataY) * 0.7)
test_size = len(dataY) - train_size
trainX, testX = np.array(dataX[0:train_size]), np.array(dataX[train_size:len(dataX)])
trainY, testY = np.array(dataY[0:train_size]), np.array(dataY[train_size:len(dataY)])

# input placeholders
X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
Y = tf.placeholder(tf.float32, [None, 1])
```

#### LSTM and Loss

```python
cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, state_is_tupe=True)
outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
Y_pred = tf.contrib.layers.fully_connected(
    outputs[:, -1], output_dim, activation_fn=None)
    # We use the last cell's output

# cost/loss
loss = tf.reduce_sum(tf.square(Y_pred - Y)) # sum of the squares
# optimizer
optimizer = tf.train.AdamOptimizer(0.01)
train = optimizer.minimize(loss)
```

fully connected를 1회 수행한 뒤 y 값 구함

y_dim = 

#### Training and Results

```python
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    _, l = sess.run([train, loss], feed_dict={X: trainX. Y: trainY})
    print(i, l)


testPredict = sess.run(Y_pred, feed_dict={X: testX})

import matplotlib.pyplot as plt
plt.plot(testY)
plt.plot(testPredict)
plt.show()
```

학습이 끝나면 y 값이 무엇일지 예측 -> 화면에다가 plot

예측과 결과가 거의 다 일치한다.

#### Exercise

- Implement stock prediction using linear regression only
- Improve results using more features such as keywords and/or sentiments in top news

#### Other RNN applications

- Language Modeling
- Speech Recognition
- Machine Translation
- Conversation Modeling/Question Answering
- Image/Video Captioning
- Image/Music/Dance Generation
