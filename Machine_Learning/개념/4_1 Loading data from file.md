## Loading data from file

```python
# 점수 예측 프로그램
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
tf.set_random_seed(777) # for reproducibility

xy = np.loadtxt('data-01-test-score.csv', delimiter=",", dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

# Make sure the shape and data are OK
print(x_data.shape, x_data, len(x_data))
print(y_data.shape, y_data)

# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis
hypothesis = tf.matmul(X, W) + b

# Simplified cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

# Launch the graph in a session
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())
# Set up feed_dict variables inside the loop.
for step in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
    if step % 10 == 0:
        print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)
    
# Ask my score
print("Your score will be ", sess.run(hypothesis, feed_dict={X: [[100, 70, 101]]}))
print("Other scores will be ", sess.run(hypothesis,feed_dict={X: [[60, 70, 110], [90, 100, 80]]}))
```

## Queue Runners

파일의 크기가 굉장히 크거나 파일이 여러개여서 numpy를 이용하기 어려울 때

<img width="1000" alt="스크린샷 2020-12-08 오후 7 51 27" src="https://user-images.githubusercontent.com/62995632/101474656-cf3db880-398e-11eb-89b6-f6fc9f6ddc2d.png">

1. 가지고 있는 파일 리스트
```python
filename_queue = tf.train.string_input_producer(['data-01-test-score.csv', 'data-02-test-score.csv', ... ], shuffle=False, name='filename_queue')
```

2. 파일을 읽어올 reader 정의
```python
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)
```

3. value를 어떻게 parsing 할 것인지 decode_csv로 가져옴
```python
record_defaults = [[0.], [0.], [0.], [0.]]
xy = tf.decode_csv(value, record_defaults=record_defaults)
```

## tf.train.batch
펌프처럼 원하는 batch size 만큼만 데이터를 끌어들여서 계산
```python
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

filename_queue = tf.train.string_input_producer(['data-01-test-score.csv'], shuffle=False, name='filename_queue')

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

# Default values, in case of empty columns. Also specifies the type of the
# decoded result.
record_defaults = [[0.], [0.], [0.], [0.]]
xy = tf.decode_csv(value, record_defaults=record_defaults)

# collect batches of csv in
train_x_batch, train_y_batch = tf.train.batch([xy[0:-1], xy[-1:]], batch_size=10)

# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis
hypothesis = tf.matmul(X, W) + b

# Simplified cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

# Start populating the filename queue.
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

for step in range(2001):
    x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_batch, Y: y_batch})
    if step % 10 == 0:
        print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)

coord.request_stop()
coord.join(threads)
```

## shuffle_batch
batch를 섞고 싶을 때
```python
# min_after_dequeue defines how big a buffer we will randomly sample
#   from -- biger means better shuffling but slower start up and more
#   memory used.
# capacity must be larger than min_after_dequeue and the amout larger
#   determines the maximum we will prefetch.    Recommendation:
#   min_after_dequeue + (num_threads + a small safety margin) * batch_size

min_after_dequeue = 10000
capacity = min_after_dequeue + 3 * batch_size
example_batch, label_batch = tf.train.shuffle_batch([example, label], batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue)
```
