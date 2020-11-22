## TensorFlow

- TensorFlow is an open source software library for numerical computation using data flow graphs.
- Python!

Data Flow Graph
- Nodes in the graph represent mathematical operations
- Edges represent the multidimensional data arrays(tensors) communicated between them

```python
# Hello, TensorFlow 출력하기

import tensorflow as tf

hello = tf.constant("Hello, TensorFlow!")
sess = tf.Session()
print(sess.run(hello))
```

TensorFlow Mechanics

Computational Graph

1. Build graph using TensorFlow operations
```python
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)
```
2. feed data and run graph(operation) 

-> sess.run(op), sess.run(op, feed_dict={x:x_data})
3. update variables in the graph(and return values)
```python
sess = tf.Session()

print("sess.run(node1, node2): ", sess.run([node1, node2]))
print("sess.run(node3): ", sess.run(node3))
```

Placeholder

```python
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b

print(sess.run(adder_node, feed_dict={a:3, b:4.5}))
print(sess.run(adder_node, feed_dict={a:[1,3], b:[2,4]}))
```

Tensor Ranks

|Rank|Math entity|Python example|
|----|-----------|--------------|
|0|Scalar(magnitude only)|s = 483|
|1|Vector(magnitude and direction)|v = [1.1, 2.2, 3.3]|
|2|Matrix(table of numbers)|m = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]|
|3|3-Tensor(cube of numbers)|t = [[[2], [4], [6]], [8], [10], [12]], [[14], [16], [18]]]|
|n|n-Tensor(you get the idea)|...|


Tensor Shapes

|Rank|Shape|Dimension number|Example|
|----|-----|----------------|-------|
|0|[]|0-D|A 0-D tensor. A scalar.|
|1|[D0]|1-D|A 1-D tensor with shape [5]|
|2|[D0, D1]|2-D|A 2-D tensor with shape [3,4]|
|3|[D0, D1, D2]|3-D|A 3-D tensor with shape [1,4,3]|
|n|[D0, D1, ..., Dn-1]|n-D|A tensor with shape [D0,D1,...,Dn-1]|


Tensor Types

|Data type|Python type|Description|
|---------|-----------|-----------|
|DT_FLOAT|tf.float32|32 bits floating point|
|DT_DOUBLE|tf.float64|64 bits floating point|
|DT_INT8|tf.int8|8 bits signed integer|
|DT_INT16|tf.int16|16 bits signed integer|
|DT_INT32|tf.int32|32 bits signed integer|
|DT_INT64|tf.int64|64 bits signed integer|
