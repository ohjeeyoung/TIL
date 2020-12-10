## Training and Test datasets

training set과 test set을 나누는 것이 중요

```python
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

x_data = [[1, 2, 1], [1, 3, 2], [1, 3, 4], [1, 5, 5], [1, 7, 5], [1, 2, 5], [1, 6, 6], [1, 7, 7]]
y_data = [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]]

# Evaluation our model using this test dataset
x_test = [[2, 1, 1], [3, 1, 2], [3, 3, 4]]
y_test = [[0, 0, 1], [0, 0, 1], [0, 0, 1]]

X = tf.placeholder("float", [None, 3])
Y = tf.placeholder("float", [None, 3])
W = tf.Variable(tf.random_normal([3, 3]))
b = tf.Variable(tf.random_normal([3]))

hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))   # axis = 축
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Correct prediction Test model
prediction = tf.arg_max(hypothesis, 1)
is_correct = tf.equal(prediction, tf.arg_max(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# Launch graph
with tf.Session() as sess:
    # Initialize Tensorflow variables
    sess.run(tf.global_variables_initializer())
    for step in range(201):
        cost_val, W_val, _ = sess.run([cost, W, optimizer], feed_dict={X: x_data, Y: y_data})
        print(step, cost_val, W_val)

    # predict
    print("Prediction:", sess.run(prediction, feed_dict={X: x_test}))
    # Calculate the accuracy
    print("Accuracy: ", sess.run(accuracy, feed_dict={X: x_test, Y: y_test}))
```

#### Learning rate: NaN!

learning rate이 제대로 설정되지 않을 때 발생할 수 있는 문제

<img width="1000" alt="스크린샷 2020-12-11 오전 4 08 58" src="https://user-images.githubusercontent.com/62995632/101818025-9eb07700-3b66-11eb-8119-3149d5bd6733.png">

1. Large learning rate: Overshooting.
2. Small learning rate: Many iterations until convergence and trapping in local minima.

-> 조금의 굴곡만 있어도 최소점이라고 판단해버릴 수도 있음

#### Non-normalized inputs

한쪽 방향으로 치우친 그래프가 그려지면 바깥으로 튕겨나갈 수도 있음

데이터의 값이 지나치게 들쑥날쑥 하는 경우

<img width="1000" alt="스크린샷 2020-12-11 오전 4 12 53" src="https://user-images.githubusercontent.com/62995632/101818411-29917180-3b67-11eb-8c7c-610145ab37a3.png">

#### Normalized inputs(min-max scale)

```python
xy = MinMaxScaler(xy)
print(xy)
```

제일 작은값을 0, 제일 큰 값을 1로 하고 나머지는 0과 1 사이로 고르게 만들 수 있다.

데이터가 지나치게 값이 들쑥날쑥하는 경우에는 min-max scale을 이용해 값을 고르게 만든다.
