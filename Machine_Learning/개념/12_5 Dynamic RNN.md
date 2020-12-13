## Dynamic RNN

#### Different sequence length

<img width="1000" alt="스크린샷 2020-12-14 오전 3 53 32" src="https://user-images.githubusercontent.com/62995632/102020974-f4339080-3dbf-11eb-9925-9fb9d7544ddb.png">

문자열 길이가 그때그때 달라짐

RNN은 가변하는 sequence를 받아들여야함

이전에는 <padding>을 삽입했음 -> 하지만 weight 때문에 값이 계산되어 나올 수 밖에 없음

<img width="1000" alt="스크린샷 2020-12-14 오전 3 54 29" src="https://user-images.githubusercontent.com/62995632/102020998-1d542100-3dc0-11eb-9a01-2ec7984354c9.png">

각각의 batch에 문자열 또는 sequence 길이를 정의하도록 함

빈자리는 0으로 해서 loss가 헷갈리지 않도록 함 -> dynamic RNN의 장점

#### Dynamic RNN

<img width="1000" alt="스크린샷 2020-12-14 오전 3 58 44" src="https://user-images.githubusercontent.com/62995632/102021075-b08d5680-3dc0-11eb-8198-6558bbe100da.png">

```python
# 3 batches 'hello', 'eolll', 'lleel'
x_data = np.array([[[...]]], dtype=np.float32)

hidden_size = 2
cell = rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
outputs, _states = tf.nn.dynamic_rnn(cell, x_data, sequence_length=[5,3,4], dtype=tf.float32)
sess.run(tf.global_variables_initializer())
print(outputs.eval())
```
