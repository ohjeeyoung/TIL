## MNIST Dataset

손으로 쓰여있는 숫자를 판별하는 것

<img width="1000" alt="스크린샷 2020-12-11 오전 4 17 46" src="https://user-images.githubusercontent.com/62995632/101818881-d8ce4880-3b67-11eb-91ce-31283d4fd3cf.png">

작은 사각형의 개수만큼 placeholder를 준다.

```python
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# Reading data and set variables
from tensorflow.examples.tutorials.mnist import input_data

# Check out https://www.tensorflow.org/get_started/mnist/beginners for
# more information about the mnist dataset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

nb_classes = 10

# MNIST data image of shape 28 * 28 = 784
X = tf.placeholder(tf.float32, [None, 784])
# 0 - 9 digits recognition = 10 classes
Y = tf.placeholder(tf.float32, [None, nb_classes])

W = tf.Variable(tf.random_normal([784, nb_classes]))
b = tf.Variable(tf.random_normal([nb_classes]))

# Softmax
# Hypothesis(using softmax)
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Test model
is_correct = tf.equal(tf.arg_max(hypothesis, 1), tf.arg_max(Y, 1))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# Training epoch/batch
# 데이터가 많으면 한번에 학습이 어렵기 때문에 batch로 잘라서 학습

# parameters
training_epochs = 15    # 전체 데이터셋을 한번 학습시킨 것 -> 1 epoch
batch_size = 100

with tf.Session() as sess:
    # Initialize Tensorflow variables
    sess.run(tf.global_variables_initializer())
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c, _ = sess.run([cost, optimizer], feed_dict={X: batch_xs, Y: batch_ys})
            avg_cost += c / total_batch
        
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

# Report results on test dataset

# Test the model using test sets
print("Accuracy: ", accuracy.eval(session=sess, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))

```

#### Training epoch/batch
- one epoch = one forward pass and one backward pass of all the training examples

- batch size = the number of training examples in one forward/backward pass.
The higher the batch size, the more memory space you'll need.

- number of iterations = number of passes, each pass using [batch size] number of examples.
To be clear, one pass = one forward pass + one backward pass(we do not count the 
forward pass and backward pass as two different passes).

Examples: if you have 1000 training examples, and your batch size is 500, then it will take 2 iterations to complete 1 epoch.

1 epoch는 전체 training set을 한번 도는 것

batch는 한번에 모든 training set을 볼 수 없으므로 batch size만큼 잘라서 training을 진행

10000개의 training example이 있고 batch size가 500이라면 1 epoch 하는데 2번 반복이 필요함

#### Sample image show and prediction

랜덤 이미지를 테스트 해보기

```python
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt
import random

# Get one and predict
r = random.randint(0, mnist.test.num_examples - 1)
print("Label:", sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
print("Prediction:", sess.run(tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r + 1]}))

plt.imshow(mnist.test.images[r:r + 1].reshape(28, 28), cmap='Greys', interpolation='nearest')
plt.show()
```
