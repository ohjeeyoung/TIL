## NN dropout and model ensemble

## Overfitting

대략 좋은 curve로 하는 것이 더 fit 하지만 더 정확하게 하겠다고 막 구부리는 것

#### Am I overfitting?

> overfitting 확인법

- Very high accuracy on the training dataset(eg:0.99)
- Poor accuracy on the test data set(0.85)

training dataset에서는 정확도가 매우 높고 test dataset에서는 낮게 나올 때

#### Solutions for overfitting

- More training data!
- Reduce the number of features
- Regularizations

## Regularization

> 그래프를 지나치게 구부리지 말자

<img width="1000" alt="스크린샷 2020-12-12 오전 5 28 55" src="https://user-images.githubusercontent.com/62995632/101951838-fc15f800-3c3a-11eb-8941-2dd9f016e8c4.png">

람다의 값에 따라 중요도가 달라진다.

## Regularization: Dropout

> A Simple Way to Prevent Neural Networks from Overfitting[Srivastava et al.2014]

<img width="1000" alt="스크린샷 2020-12-12 오전 5 32 05" src="https://user-images.githubusercontent.com/62995632/101952075-69c22400-3c3b-11eb-96e3-b17e18475b0f.png">

Neural Network로 엮어놓은 것을 임의적으로 끊어버림 -> 몇개의 노드를 죽임

<img width="1000" alt="스크린샷 2020-12-12 오전 5 32 15" src="https://user-images.githubusercontent.com/62995632/101952078-6a5aba80-3c3b-11eb-9d35-e3a2b0400b32.png">

몇개로만 학습을 시키고 마지막에 전체로 예측을 하면 더 잘될 수도 있음

#### TensorFlow implementation

```python
dropout_rate = tf.placeholder("float")
_L1 = tf.nn.relu(tf.add(tf.matmul(X, W1), B1))
L1 = tf.nn.dropout(_L1, dropout_rate)

# TRAIN:
sess.run(optimizer, feed_dict={X: batch_xs, Y: batch_ys, dropout_rate: 0.7})
# 학습때는 0.7만 참여하도록

# EVALUATION:
print("Accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels, dropout_rate: 1})
# 평가때는 전부 참여해야하므로 dropout_rate는 1
```

-> 주의할 점: 학습할 때만 dropout해서 하는 것. 실전에는 모두 불러와야 함

## What is Ensemble?

<img width="1000" alt="스크린샷 2020-12-12 오전 5 38 33" src="https://user-images.githubusercontent.com/62995632/101952679-624f4a80-3c3c-11eb-968b-0a33da0b5e6e.png">

기계가 많고, 학습할 것이 많을 때 사용

독립된 각각을 학습시킨 뒤 나중에 모두 합침

2% ~ 4,5% 정도 성능이 향상됨
