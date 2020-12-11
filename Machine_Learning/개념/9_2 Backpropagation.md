##

#### How can we learn W1, W2, B1, B2 from training data?

-> Gradient Descent Algorithm 사용

cost 함수가 밥그릇 모양으로 생겼다면 어느점에서 시작해도 global minimal한 점에 도달하게 됨

-> 미분을 계산해야함(기울기)

#### Derivation

layer가 늘어날수록 기울기를 구하기 어려워짐

-> Backpropagation을 이용해 해결

예측값과 실제값을 비교해 나오는 오류(cost)를 뒤에서부터 앞으로 돌려서 뭘 조정해야하는지 확인

## Back propagation(chain rule)

<img width="1000" alt="스크린샷 2020-12-12 오전 2 36 29" src="https://user-images.githubusercontent.com/62995632/101935730-db41a880-3c22-11eb-8fea-ba41dcc9c4d3.png">

f = wx + b, g = wx, f = g + b

neural network 형태로 나타낸 것

w, x b가 f에 미치는 영향을 구하고 싶음 -> 미분값

1. forward로 값을 넣어서 계산함
2. 식들을 미리 미분해둠
3. g와 b는 1로 미분계수가 나옴
4. 나머지 w와 x를 미분할 때는 chain rule을 사용해서 구함

-> 미분의 의미: 그 값이 1 변화할 때 f가 얼만큼 변하는지

ex. f에 대한 b의 미분이 1이므로 b의 값이 1 변할 때 f의 값도 1 변화한다.

![스크린샷 2020-12-12 오전 2 39 17](https://user-images.githubusercontent.com/62995632/101935999-47bca780-3c23-11eb-8e4c-4aeda12a746b.png)

layer가 많아도 똑같이 뒤로 미분하면서 구할 수 있음

#### Sigmoid

<img width="1000" alt="스크린샷 2020-12-12 오전 2 41 31" src="https://user-images.githubusercontent.com/62995632/101936191-95391480-3c23-11eb-87c0-db9bf1770e9c.png">

미분값만 알고 있으면 아무리 복잡한 식도 구할 수 있음

#### Back propagation in TensorFlow(TensorBoard)

<img width="1000" alt="스크린샷 2020-12-12 오전 2 43 13" src="https://user-images.githubusercontent.com/62995632/101936545-05e03100-3c24-11eb-8ba9-8dd8cb2e8017.png">

<img width="1000" alt="스크린샷 2020-12-12 오전 2 44 15" src="https://user-images.githubusercontent.com/62995632/101936548-0678c780-3c24-11eb-8d9e-1424debf52e5.png">

```python
hypothesis = tf.sigmoid(tf.matmul(L2, W2) + b2)
# cost function
cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y)*tf.log(1-hypothesis))
```

각각을 그래프로 만든 이유는 tensorflow가 미분을 하기 위해서(backpropagation)

tensorflow를 이용하면 미분을 직접하지 않아도 됨
