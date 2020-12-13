## RNN with long sequences

Stacked RNN + Softmax layer

#### Wide & Deep

<img width="1000" alt="스크린샷 2020-12-14 오전 3 20 17" src="https://user-images.githubusercontent.com/62995632/102020255-51791300-3dbb-11eb-9048-b59695cfc8ed.png">

앞의 것이 제대로 작동하지 않은 이유는 문장은 긴데 RNN이 하나라 Wide & Deep 하지 않았음

#### Stacked RNN

<img width="1000" alt="스크린샷 2020-12-14 오전 3 20 11" src="https://user-images.githubusercontent.com/62995632/102020256-52aa4000-3dbb-11eb-9cac-129200c7fe41.png">

층을 쌓는다.

```python
X = tf.placeholder(tf.int32, [None, seq_length])   # X data
Y = tf.placeholder(tf.int32, [None, seq_length])   # Y label

# One-hot encoding
X_one_hot = tf.one_hot(X, num_classes)
print(X_one_hot)  # check out the shape

# Make a lstm cell with hidden_size (each unit output vector size)
cell = rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)
cell = rnn.MultiRNNCell([cell] * 2, state_is_tuple=True)

# outputs: unfolding size x hidden size, state = hidden size
outputs, _states = tf.nn.dynamic_rnn(cell, X_one_hot, dtype=tf.float32)
```

기본 셀을 만들어놓고, multiRNNCell에서 [cell] * (원하는 높이)로 계산하면 됨

#### Softmax

뒤에 Softmax를 붙여서 계산

<img width="1000" alt="스크린샷 2020-12-14 오전 3 26 34" src="https://user-images.githubusercontent.com/62995632/102020386-407cd180-3dbc-11eb-8a93-a909dd2a6b30.png">

RNN 자체가 하나로 보는 것을 펼친 것이기 때문에 stack처럼 하나로 쭉 쌓아서 한번에 softmax 처리해도 됨

reshape을 두번 함

```python
# (optional) softmax layer
X_for_softmax = tf.reshape(outputs, [-1, hidden_size])

softmax_w = tf.get_variable("softmax_w", [hidden_size, num_classes])
softmax_b = tf.get_variable("softmax_b", [num_classes])
outputs = tf.matmul(X_for_softmax, softmax_w) + softmax_b
outputs = tf.reshape(outputs, [batch_size, seq_length, num_classes])  # RNN's output shape과 같다
```

#### Loss

```python
# reshape out for sequence_loss
outputs = tf.reshape(outputs, [batch_size, seq_length, num_classes])

# All weights are 1 (equal weights)
weights = tf.ones([batch_size, seq_length])

sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
mean_loss = tf.reduce_mean(sequence_loss)

train_op = tf.train.AdamOptimizer(learning_rate=0.1).minimize(mean_loss)
```

#### Training and print results

```
sess = tf.Session()
sess.run(tf.global_variables_initializer()

for i in range(500):
    _, l, results = sess.run([train_op, mean_loss, outputs], feed_dict={X: dataX, Y: dataY})
    
    for j, result in enumerate(results):
        index = np.argmax(result, axis=1)
        print(i, j, ''.join([char_set[t] for t in index]), 1)
```

<img width="567" alt="스크린샷 2020-12-14 오전 3 40 56" src="https://user-images.githubusercontent.com/62995632/102020691-32c84b80-3dbe-11eb-8cb2-c71418445cd3.png">

```python
# Let's print the last char of each result to check it works
results = sess.run(outputs, feed_dict={X: dataX})
for j, result in enumerate(results):
    index = np.argmax(result, axis=1)
    if j is 0:  # print all for the first result to make a sentence
        print(''.join([char_set[t] for t in index]), end='')
    else:
        print(char_set[index[-1]], end='')
```

굉장히 긴 문장도 잘 학습한 것을 알 수 있음

#### char-rnn

아주 긴 셰익스피어같은 작품도 학습할 수 있음

#### char/word rnn(char/word level n to n model)
