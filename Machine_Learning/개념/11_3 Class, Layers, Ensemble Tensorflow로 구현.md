## Class, Layers, Ensemble

#### Python Class

CNN을 파이썬 클래스로 효과적으로 관리할 수 있음

```python
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class Model:

    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()
    
    def _build_net(self):
        with tf.variable_scope(self.name):
            # input place holders
            self.X = tf.placeholder(tf.float32, [None, 784])
            # img 28x28x1 (black/white)
            X_img = tf.reshape(self.X, [-1, 28, 28, 1])
            self.Y = tf.placeholder(tf.float32, [None, 10])

            # L1 ImgIn shape=(?, 28, 28, 1)
            W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
            ...

def predict(self, x_test, keep_prop=1.0):
    return self.sess.run(self.logits, feed_dict={self.X: x_test, self.keep_prob: keep_prop})

def get_accuracy(self, x_test, y_test, keep_prop=1.0):
    return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test, self.keep_prob: keep_prop})

def train(self, x_data, y_data, keep_prop=0.7):
    return self.sess.run([self.cost, self.optimizer], feed_dict={self.X: x_data, self.Y: y_data, self.keep_prob: keep_porp})


# initialize
sess = tf.Session()
m1 = Model(sess, "m1")

sess.run(tf.global_variables_initializer())

print('Learning Started!')

# train the model
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        c, _ = m1.train(batch_xs, batch_ys)
        avg_cost += c / total_batch
```

#### tf.layers

<img width="1000" alt="스크린샷 2020-12-14 오전 12 20 49" src="https://user-images.githubusercontent.com/62995632/102016142-bd9b4d00-3da2-11eb-9f5e-7a433deceefd.png">

<img width="1000" alt="스크린샷 2020-12-14 오전 12 25 51" src="https://user-images.githubusercontent.com/62995632/102016170-efacaf00-3da2-11eb-9741-773b440b1804.png">

값이 여러개 일때 보기 쉽게 변수를 나타낼 수 있다.

## Ensemble

<img width="1000" alt="스크린샷 2020-12-14 오전 12 27 20" src="https://user-images.githubusercontent.com/62995632/102016208-24b90180-3da3-11eb-9623-e9789bdd38f1.png">

여러개의 모델을 training 시키고 test data가 들어오면 각각 예측을 시키고, 예측한 결과를 조합한 결과를 내놓음

#### Ensemble training

```python
models = []
num_models = 7
for m in range(num_models):
    models.append(Model(sess, "model" + str(m)))

sess.run(tf.global_variables_initializer())
print('Learning Started!')

# train my model
for epoch in range(training_epochs):
    avg_cost_list = np.zeros(len(models))
    total_batch = int(mnist.train.num_examples / batch_size)
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)

        # train each model
        for m_idx, m in enumerate(models):
            c, _ = m.train(batch_xs, batch_ys)
            avg_cost_list[m_idx] += c / total_batch
    
    print('Epoch:', '%04d'%(epoch + 1), 'cost = ', avg_cost_list)

print('Learning Finished!')
```

#### Ensemble prediction

<img width="1000" alt="스크린샷 2020-12-14 오전 12 35 59" src="https://user-images.githubusercontent.com/62995632/102016439-5a121f00-3da4-11eb-838d-74f9eb36c4a7.png">

모델에 대해서 softmax로 구현 된 label에 각각이 될 확률을 나타냄

여기서는 다 합해서 가장 큰 것을 골라서 예측하겠다. -> argmax

```python
# Test model and check accuracy
test_size = len(mnist.test.labels)
predictions = np.zeros(test_size * 10).reshape(test_size, 10)

for m_idx, m in enumerate(models):
    print(m_idx, 'Accuracy:', m.get_accuracy(mnist.test.images, mnist.test.labels))
    p = m.predict(mnist.test.images)
    predictions += p

ensemble_correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(mnist.test.labels, 1))
ensemble_accuracy = tf.reduce_mean(tf.cast(ensemble_correct_prediction, tf.float32))
print('Ensemble accuracy:', sess.run(ensemble_accuracy))
```

결과: Ensemble accuracy: 0.9952

#### Exercise

- Deep & Wide?
- CIFAR 10
- ImageNet
