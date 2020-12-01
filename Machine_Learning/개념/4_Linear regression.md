## TensorFlow Mechanics

1. Build graph using TF operations


![CodeCogsEqn (5)](https://user-images.githubusercontent.com/62995632/93719093-662e7c00-fbbb-11ea-9805-6a5063589840.gif)

```python
x_train = [1, 2, 3]
y_train = [1, 2, 3]

W = tf.Variable(tf.random_normal([1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

# Our hypothesis XW+b
hypothesis = x_train * W + b
```


![CodeCogsEqn](https://user-images.githubusercontent.com/62995632/93712521-5bf78800-fb91-11ea-8897-2533e31787c4.gif)

```python
# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - y_train))
```

reduce_mean: average calculator

```python
t = [1. ,2. ,3. ,4.]
tf.reduce_mean(t) ==> 2.5
```

GradientDescent
```python
# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)
```

2. Run/update graph (operation)
-> sess.run(op, feed_dict={x: x_data})

```python
# Launch the graph in a session
sess = tf.Session()

# Initializes global variables in the graph
sess.run(tf.global_variables_initializer())

# Fit the line
for step in range(2001):
  sess.run(train)
  if step % 20 == 0:
    print(step, sess.run(cost), sess.run(W), sess.run(b))
```

3. update variables in the graph(and return values)

#### Placeholder
처음에 선언하지 않아도 학습하고 싶은 데이터를 후에 던져줄 수 있음
```python
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
...

for step in range(2001):
  cost_val, W_val, b_val, _ = sess.run([cost, W, b, train], feed_dict={X:[1,2,3], Y:[1,2,3]})
  
  if step % 20 == 0:
    print(step, cost_val, W_val, b_val)
```
