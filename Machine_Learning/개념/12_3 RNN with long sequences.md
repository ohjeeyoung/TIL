## RNN with long sequences

#### Manual data creation

앞에서는 선언하는 것을 손으로 작업했음

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
```

#### Better data creation

프로그램이 자동으로 하도록 해보자

```python
sample = "if you want you"
idx2char = list(set(sample))  # index -> char
char2idx = {c: i for i, c in enumerate(idx2char)}   # char -> idx

sample_idx = [char2idx[c] for c in sample]  # char to index
x_data = [sample_idx[:-1]]  # X data sample (0 ~ n-1) hello: hell
y_data = [sample_idx[1:]]   # Y label sample (1 ~ n)  hello: ello

X = tf.placeholder(tf.int32, [None, sequence_length]) # X data
Y = tf.placeholder(tf.int32, [None, sequence_length]) # Y label

X_one_hot = tf.one_hot(X, num_classes)  # one hot: 1 -> 0 1 0 0 0 0 0 0 0 0
```

#### Hyper parameters

```python
# hyper parameters
dic_size = len(char2idx)  # RNN input size(one hot size)
rnn_hidden_size = len(char2idx) # RNN output size
num_classes = len(char2idx)   # final output size(RNN or softmax, etc.)
batch_size = 1  # one sample data, one batch
sequence_length = len(sample) - 1 # number of lstm unfolding (unit #)
```

#### LSTM and Loss

```python
X = tf.placeholder(tf.int32, [None, sequence_length])   # X data
Y = tf.placeholder(tf.int32, [None, sequence_length])   # Y label

X_one_hot = tf.one_hot(X, num_classes)  # one hot: 1 -> 0 1 0 0 0 0 0 0 0 0

cell = tf.contrib.rnn.BasicLSTMCell(num_units=rnn_hidden_size, state_is_tuple=True)
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(cell, X_one_hot, initial_state=initial_state, dtype=tf.float32)

weights = tf.ones([batch_size, sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)
prediction = tf.argmax(outputs, axis=2)
```

#### Training and Results

```python
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(3000):
        l, _ = sess.run([loss, train], feed_dict={X: x_data, Y: y_data})
        result = sess.run(prediction, feed_dict={X: x_data})
        # print char using dic
        result_str = [idx2char[c] for c in np.squeeze(result)]
        print(i, "loss:", l, "Prediction:", ''.join(result_str))
```

<img width="1000" alt="스크린샷 2020-12-14 오전 2 55 22" src="https://user-images.githubusercontent.com/62995632/102019602-d2cea680-3db7-11eb-8367-99fd4f89c73e.png">


#### Really long sentence?

<img width="1000" alt="스크린샷 2020-12-14 오전 2 56 11" src="https://user-images.githubusercontent.com/62995632/102019630-f4c82900-3db7-11eb-8525-6277aaa3e0e0.png">

```python
sentence = ("if you want to build a ship, don't drup up people together to "
            "collect wood and don't assign them tasks and work, but rather "
            "teach them to long for the endless immensity of the sea.")
```

한번에 주는 것은 너무 많으므로 잘라서 줌

#### Making dataset

```python
char_set = list(set(sentence))
char_dic = {w: i for i, w in enumerate(char_set)}

dataX = []
dataY = []

for i in range(0, len(sentence) - seq_length):
    x_str = sentence[i:i + seq_length]
    y_str = sentence[i + 1: i seq_length + 1]
    print(i, x_str, '->', y_str)

    x = [char_dic[c] for c in x_str]    # x str to index
    y = [char_dic[c] for c in y_str]    # y str to index

    dataX.append(x)
    dataY.append(y)
```

#### RNN parameters

```python
char_set = list(set(sentence))
char_dic = {w: i for i, w in enumerate(char_set)}

data_dim = len(char_set)
hidden_size = len(char_set)
num_classes = len(char_set)
seq_length = 10 # Any arbitrary number

batch_size = len(dataX) # 여기서는 169
```

#### LSTM and Loss

```python
X = tf.placeholder(tf.int32, [None, sequence_length])   # X data
Y = tf.placeholder(tf.int32, [None, sequence_length])   # Y label

X_one_hot = tf.one_hot(X, num_classes)  # one hot: 1 -> 0 1 0 0 0 0 0 0 0 0

cell = tf.contrib.rnn.BasicLSTMCell(num_units=rnn_hidden_size, state_is_tuple=True)
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(cell, X_one_hot, initial_state=initial_state, dtype=tf.float32)

weights = tf.ones([batch_size, sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

prediction = tf.argmax(outputs, axis=2)
```

#### Exercise

- Run long sequence RNN
- Why it does not work?
