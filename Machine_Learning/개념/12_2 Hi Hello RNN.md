## Hi Hello RNN

#### Teach RNN 'hihello'

<img width="1000" alt="스크린샷 2020-12-14 오전 2 05 58" src="https://user-images.githubusercontent.com/62995632/102018516-fa6e4080-3db0-11eb-9f92-ed82dc4327c4.png">

일반적인 forward net으로 구현하기에는 h가 나왔을 때 한번은 i, 한번은 e를 예측해야하므로 쉽지 않다.

#### One-hot encoding

<img width="1000" alt="스크린샷 2020-12-14 오전 2 07 46" src="https://user-images.githubusercontent.com/62995632/102018541-2c7fa280-3db1-11eb-9a19-34d96be91278.png">

#### Teach RNN 'hihello'

<img width="1000" alt="스크린샷 2020-12-14 오전 2 09 17" src="https://user-images.githubusercontent.com/62995632/102018566-62bd2200-3db1-11eb-814d-de5ee7bbfe06.png">

input_dim=5, seq=6, hidden_dim=5(one-hot 때문), batch=1(문자열 하나)

#### Creating rnn cell

```python
# RNN model
rnn_cell = rnn_cell.BasicRNNCell(rnn_size)

rnn_cell = rnn_cell.BasicLSTMCell(rnn_size)
rnn_cell = rnn_cell.GRUCell(rnn_size)
```

rnn_size = 5

#### Execute RNN

<img width="1000" alt="스크린샷 2020-12-14 오전 2 13 12" src="https://user-images.githubusercontent.com/62995632/102018671-f7c01b00-3db1-11eb-884b-9450170b7d29.png">

```python
# RNN model
rnn_cell = rnn_cell.BasicRNNCell(rnn_size)

outputs, _states = tf.nn.dynamic_rnn(rnn_cell, X.initial_state=initial_state, dtype=tf.float32)
```

#### RNN parameters

```python
hidden_size = 5   # output from the LSTM
input_dim = 5     # one-hot size
batch_size = 1    # one sentence
sequence_length   # /ihello/ == 6
```

#### Data creation

```python

idx2char = ['h', 'i', 'e', 'l', 'o']    # h=0, i=1, e=2, l=3, o=4
x_data = [[0, 1, 0, 2, 3, 3]]           # hihell
x_one_hot = [[[1, 0, 0, 0, 0],          # h 0
              [0, 1, 0, 0, 0],          # i 1
              [1, 0, 0, 0, 0],          # h 0
              [0, 0, 1, 0, 0],          # e 2
              [0, 0, 0, 1, 0],          # l 3
              [0, 0, 0, 1, 0]]]         # l 3

y_data = [[1, 0, 2, 3, 3, 4]]           # ihello
X = tf.placeholder(tf.float32, [None, sequence_length, input_dim])  # X one-hot
Y = tf.placeholder(tf.int32, [None, sequence_length])   # Y label
```

x_data를 그대로 사용하지 않고, one-hot으로 바꿔서 input

#### Feed to RNN

```python
cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(cell, X, initial_state=initial_state, dtype=tf.float32)
```

#### Cost: sequence_loss

```python
# [batch_size, sequence_length]
y_data = tf.constant([[1, 1, 1]]) # true

# [batch_size, sequence_length, emb_dim]
prediction = tf.constant([[[0.2, 0.7], [0.6, 0.2], [0.2, 0.9]]], dtype=tf.float32)

# [batch_size * sequence_length]
weights = tf.constant([[1, 1, 1]], dtype=tf.float32)

sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=prediction, targets=y_data, weights=weights)
sess.run(tf.global_variables_initializer())
print("Loss: ", sequence_loss.eval())
```

결과: Loss: 0.596759

```python
# [batch_size, sequence_length]
y_data = tf.constant([[1, 1, 1]]) # true

# [batch_size, sequence_length, emb_dim]
prediction1 = tf.constant([[[0.3, 0.7], [0.3, 0.7], [0.3, 0.97]]], dtype=tf.float32)
prediction2 = tf.constant([[[0.1, 0.9], [0.1, 0.9], [0.1, 0.9]]], dtype=tf.float32)

# [batch_size * sequence_length]
weights = tf.constant([[1, 1, 1]], dtype=tf.float32)

sequence_loss1 = tf.contrib.seq2seq.sequence_loss(prediction1, y_data, weights)
sequence_loss2 = tf.contrib.seq2seq.sequence_loss(prediction2, y_data, weights)

sess.run(tf.global_variables_initializer())
print("Loss1: ", sequence_loss1.eval(), "Loss2: ", sequence_loss2.eval())
```

결과

Loss1: 0.513015

Loss2: 0.371101

실제의 label과 가까워질수록 loss가 작아짐

```python
outputs, _states = tf.nn.dynamic_rnn(cell, X, initial_state=initial_state, dtype=tf.float32)
weights = tf.ones([batch_size, sequence_length])

sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)
```

#### Train

```python
prediction = tf.argmax(outputs, axis=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        l, _ = sess.run([loss, train], feed_dict={X: x_one_hot, Y: y_data})
        result = sess.run(prediction, feed_dict={X: x_one_hot})
        print(i, "loss:", l, "prediction: ", result, "true Y: ", y_data)
        
        # print char using dic
        result_str = [idx2char[c] for c in np.squeeze(result)]
        print("\tPrediction str: ", ''.join(result_str))
```

#### Results

<img width="1000" alt="스크린샷 2020-12-14 오전 2 36 54" src="https://user-images.githubusercontent.com/62995632/102019166-402d0800-3db5-11eb-94c6-6222b39cf0b7.png">

초기에는 부정확하지만 끝에는 원하는 값을 구한 것을 알 수 있음
