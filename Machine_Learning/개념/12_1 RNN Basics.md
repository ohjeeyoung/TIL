## RNN Basics

#### RNN in TensorFlow

기존의 NN과 차이점: output이 다음에 cell로 연결된다.

```python
cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_size)
...
outputs, _states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)
```

1. cell 만들기, cell에서 나가는 Ouput size 정하기
2. 만들었던 cell을 넘겨주고 원하는 입력값 x를 넘기면 dynamic_rnn은 output 출력과 마지막 state의 값을 낸다.

두 부분으로 나누었을 때의 장점은 원하는 결과가 안나올 때 cell을 마음대로 바꿔서 해볼 수 있음

```python
cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size)
```

#### One node: 4(input-dim) in 2(hidden_size)

<img width="1000" alt="스크린샷 2020-12-14 오전 1 30 42" src="https://user-images.githubusercontent.com/62995632/102017738-16231800-3dac-11eb-8a9e-23fe906cd387.png">

원하는 단어를 벡터로 나타내는 가장 좋은 방법은 one-hot encoding

input dimension은 h,e,l,o 4개이므로 4

<img width="1000" alt="스크린샷 2020-12-14 오전 1 34 53" src="https://user-images.githubusercontent.com/62995632/102017833-947fba00-3dac-11eb-8a51-a8b812c060e6.png">

output의 dimension은 마음대로 -> 우리가 정한 hidden_size에 따라 출력의 dim이 정해짐

```python
# One cell RNN input_dim(4) -> output_dim(2)
hidden_size = 2
cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size)

x_data = np.array([[[1,0,0,0]]], dtype=np.float32)
outputs, _status = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)

sess.run(tf.global_variables_initializer())
pp.pprint(outputs.eval())
```

결과: array([[[-0.42409304, 0.64651132]]])

#### Unfolding to n sequences

<img width="1000" alt="스크린샷 2020-12-14 오전 1 41 37" src="https://user-images.githubusercontent.com/62995632/102017967-854d3c00-3dad-11eb-8e68-4d552766c778.png">

RNN은 하나처럼 보이지만 fold를 쪼개서 series data를 구할 수 있음

sequence_length = 한번에 sequence를 몇번 할 것인지 cell을 몇번 펼칠 것인지

입력 데이터의 모양에 따라 sequence_length이 결정됨

x의 shape = (1,5(sequence_length),4(input dim))

y의 shape = (1,5(sequence_length),2(hidden dim))

<img width="1000" alt="스크린샷 2020-12-14 오전 1 43 35" src="https://user-images.githubusercontent.com/62995632/102017996-cba29b00-3dad-11eb-9973-c27977232885.png">

sequence 만드는 법 -> 직접 hello를 입력해 sequence size = 5

```python
# One cell RNN input_dim(4) -> output_dim(2). sequence: 5
hidden_size = 2
cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size)
x_data = np.array([[h, e, l, l, o]], dtype=np.float32)
print(x_data.shape)
pp.pprint(x_data)
outputs, states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)
sess.run(tf.global_variables_initializer())
pp.pprint(outputs.eval())
```

#### Batching input

<img width="1000" alt="스크린샷 2020-12-14 오전 1 48 55" src="https://user-images.githubusercontent.com/62995632/102018119-8cc11500-3dae-11eb-8d6f-3ea1bc00c5c9.png">

학습을 시킬 때 문자열 하나씩 학습시키면 비효율적이므로 sequence를 여러개 줌 -> batch_size

x의 shape = (3(batch_size),5(sequence_length),4(input dim))

y의 shape = (3(batch_size),5(sequence_length),2(hidden dim))

<img width="1000" alt="스크린샷 2020-12-14 오전 1 52 32" src="https://user-images.githubusercontent.com/62995632/102018193-140e8880-3daf-11eb-9d6f-cdc5ab34a52d.png">

<img width="1000" alt="스크린샷 2020-12-14 오전 1 52 43" src="https://user-images.githubusercontent.com/62995632/102018194-153fb580-3daf-11eb-8413-8ff10d16c4b6.png">

```python
# One cell RNN input_dim(4) -> output_dim(2). sequence: 5, batch 3
# 3 batches 'hello', 'eolll', 'lleel'
x_data = np.array([[h, e, l, l, o],
                   [e, o, l, l, l],
                   [l, l, e, e, l]], dtype=np.float32)
pp.pprint(x_data)

cell = rnn.BasicLSTMCell(num_units=2, state_is_tuple=True)
outputs, _states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)
sess.run(tf.global_variables_initializer())
pp.pprint(outputs.eval())
```
