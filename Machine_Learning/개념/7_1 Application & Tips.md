## Application & Tips: Learning rate, data preprocessing, overfiting

## Learning rate: overshooting

Gradient Descent를 사용할 때 우리가 기울기 앞에 a(learning rate)를 곱해주었다.

```python
# Minimize error using cross entropy
learning_rate = 0.001
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), reduction_indices=1))  # Cross entropy
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) # Gradient Descent
```

learning rate을 지나치게 크게 줄 경우: 바깥으로 튕겨나갈 수 있음(overshooting)

#### Small learning rate: takes too long, stops at local minimum

### Try several learning rates

- Observe the cost function
- Check it goes down in a reasonable rate

## Data(X) preprocessing for gradient descent

data 값에 큰 차이가 있을 경우에 nomalize 할 필요가 있음

-> zero-centered data(중심이 0으로) 많이 사용

-> nomalized data(어떤 범위안에 항상 들어가도록)

#### Standardization

x에 평균을 빼고 분산을 나누어 구함

![daum_equation_1607515992567](https://user-images.githubusercontent.com/62995632/101628586-63794f80-3a63-11eb-87b0-dcd63c77bdd2.png)

```python
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
```

## Overfitting

주어진 데이터에만 지나치게 맞춰져서 잘맞는 모델일 때

실제적으로 사용하거나 실제 데이터를 사용했을 때 안맞는 경우가 생김

- Our model is very good with training data set (with memorization)
- Not good at test dataset or in real use

model1과 model2 중 model1이 더 좋은 case

model2의 경우 가지고 있는 데이터에만 지나치게 맞춰져 실제로 사용할 때 부정확 함

<img width="1000" alt="스크린샷 2020-12-09 오후 9 16 01" src="https://user-images.githubusercontent.com/62995632/101628945-ec908680-3a63-11eb-94c0-7a9612f1752b.png">

#### Solution for overfiting

- More training data!
- Reducing the number of features
- Regularization(일반화)

#### Regularization

- Let's not have too big numbers in the weight

cost를 최소화시키는 것이 목표였으나 뒤에 추가 식을 적어 그래프가 지나치게 기울이는 것을 막고 펴준다.

regularization strength가 클수록 중요하게 생각하는 것, 작을수록 신경 안써도 되는 것

<img width="1000" alt="스크린샷 2020-12-09 오후 9 20 05" src="https://user-images.githubusercontent.com/62995632/101629204-54df6800-3a64-11eb-8816-c866cbe169c4.png">

```python
l2reg = 0.001 * tf.reduce_sum(tf.square(W))
```

Tensorflow를 사용할 때는 기존의 cost 함수에다가 위의 식을 더해줌

#### Summary

1. Learning rate
2. Data preprocessing
3. Overfitting
- More training data
- Regularization

