## Minimize Cost

#### Hypothesis and Cost

![CodeCogsEqn (5)](https://user-images.githubusercontent.com/62995632/93719093-662e7c00-fbbb-11ea-9805-6a5063589840.gif)

![CodeCogsEqn](https://user-images.githubusercontent.com/62995632/93712521-5bf78800-fb91-11ea-8897-2533e31787c4.gif)


#### Simplified hypothesis

![CodeCogsEqn (6)](https://user-images.githubusercontent.com/62995632/93719105-73e40180-fbbb-11ea-945d-3491a7f98730.gif)

![CodeCogsEqn (1)](https://user-images.githubusercontent.com/62995632/93718554-27e38d80-fbb8-11ea-8bb2-b2e05e19d00b.gif)


#### Gradient descent algorithm
- Minimize cost function
- Gradient descent is used many minimization problems
- For a given cost function, cost(W,b), it will find W,b to minimize cost
- It can be applied to more general function: cost(w1,w2,...)

#### How it works
- Start with initial guesses

-> Start at 0,0(or any other value)
-> Keeping changing W and b a little bit to try and reduce cost(W,b)

- Each time you change the parameters, you select the gradient which reduces cost(W,b) the most possible
- Repeat
- Do so until you converge to a local minimum
- Has an interesting property

-> Where you start can determine which minimum you end up


#### Formal definition

![CodeCogsEqn (1)](https://user-images.githubusercontent.com/62995632/93718554-27e38d80-fbb8-11ea-8bb2-b2e05e19d00b.gif)
-> ![CodeCogsEqn (2)](https://user-images.githubusercontent.com/62995632/93718904-17ccad80-fbba-11ea-9170-d3bd28f3a5c4.gif)

![CodeCogsEqn (3)](https://user-images.githubusercontent.com/62995632/93718981-9de8f400-fbba-11ea-8c89-c2af5eaa2420.gif)
-> ![CodeCogsEqn (4)](https://user-images.githubusercontent.com/62995632/93719055-22d40d80-fbbb-11ea-8637-aab83e66f35f.gif)


#### Convex function

check cost function is convex -> then we can get the result

#### cost function 시각화

```python
import tensorflow as tf
import matplotlib.pyplot as plt

X = [1, 2, 3]
Y = [1, 2, 3]

W = tf.placeholder(tf.float32)
# Our hypothesis for linear model X * W
hypothesis = X * W

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))
# Launch the graph in a session
sess = tf.Session()
# Initializes global variables in the graph
sess.run(tf.global_variables_initializer())
# Variables for plotting cost function
W_val = []
cost_val = []

for i in range(-30, 50):
    feed_W = i * 0.1
    curr_cost, curr_W = sess.run([cost, W], feed_dict={W: feed_W})
    W_val.append(curr_W)
    cost_val.append(curr_cost)

# Show the cost function
plt.plot(W_val, cost_val)
plt.show()
```

#### Gradient descent

![CodeCogsEqn (4)](https://user-images.githubusercontent.com/62995632/93719055-22d40d80-fbbb-11ea-8637-aab83e66f35f.gif)

```python
# Minimize: Gradient Descent using derivative: W -= Learning_rate * derivative
learning_rate = 0.1
gradient = tf.reduce_mean((W * X - Y) * X)
descent = W - learning_rate * gradient
update = W.assign(descent)  # tense는 =로 바로 assign 불가
```

```python
import tensorflow as tf

x_data = [1,2,3]
y_data = [1,2,3]

W = tf.Variable(tf.random_normal([1]), name = 'weight')
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# Our hypothesis for linear model X * W
hypothesis = X * W

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize: Gradient Descent using derivative: W -= Learning_rate * derivative
learning_rate = 0.1
gradient = tf.reduce_mean((W * X - Y) * X)
descent = W - learning_rate * gradient
update = W.assign(descent)

# Launch the graph in a session
sess = tf.Session()
# Initializes global variables in the graph
sess.run(tf.global_variables_initializer())

for i in range(21):
    sess.run(update, feed_dict={X: x_data, Y: y_data})
    print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))
```

변수가 많아지고 복잡해지면 매번 Gradient Descent 함수 선언하기 어려우므로 아래 함수로 대체 가능
```python
# Minimize: Gradient Descent Magic
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)
```
```python
X = [1,2,3]
Y = [1,2,3]

W = tf.Variable(-3.0)

# Our hypothesis for linear model X * W
hypothesis = X * W

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize: Gradient Descent Magic
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

# Launch the graph in a session
sess = tf.Session()
# Initializes global variables in the graph
sess.run(tf.global_variables_initializer())

for step in range(100):
    print(step, sess.run(W))
    sess.run(train)
```

#### Optional: compute_gradient and apply_gradient

gradient를 임의로 조정하고 싶을 때

수식으로 주어진 것과 컴퓨터가 제공하는 값 비교해보기
```python
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

X = [1,2,3]
Y = [1,2,3]

# Set wrong model weights
W = tf.Variable(5.0)

# Linear model
hypothesis = X * W

# Manual gradient
gradient = tf.reduce_mean((W * X - Y) * X) * 2

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)

# Get gradients
gvs = optimizer.compute_gradients(cost)

# Apply gradients
apply_gradients = optimizer.apply_gradients(gvs)

# Launch the graph in a session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(100):
    print(step, sess.run([gradient, W, gvs]))
    sess.run(apply_gradients)
```
-> 컴퓨터가 제공하는 값과 수식으로 계산한 값이 동일하다는 것을 알 수 있다.
