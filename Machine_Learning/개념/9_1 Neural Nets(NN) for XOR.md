## Neural Nets(NN) for XOR

#### One logistic regression unit cannot separate XOR

하나의 logistic regression으로는 XOR을 해결할 수 없음

#### Multiple logistic regression units

여러개를 합치면 풀 수 있었음

#### Neural Nets(NN)

"No one on earth had found a viable way to train"

하지만 각각의 W, b를 구할 수가 없었다. -> 너무 복잡하게 얽혀있기 때문

## XOR using NN

#### Neural Net

x1, x2를 넣어 y1, y2를 구하고 그 값을 다시 넣어서 최종 y를 구한다.

<img width="1000" alt="스크린샷 2020-12-11 오후 10 16 47" src="https://user-images.githubusercontent.com/62995632/101907955-aae81300-3bfe-11eb-8cda-6145a4596586.png">

<img width="1000" alt="스크린샷 2020-12-11 오후 10 20 51" src="https://user-images.githubusercontent.com/62995632/101908564-9a846800-3bff-11eb-931a-8ebb89d1000a.png">

<img width="1000" alt="스크린샷 2020-12-11 오후 10 21 07" src="https://user-images.githubusercontent.com/62995632/101908566-9bb59500-3bff-11eb-86e1-a51ad89ac973.png">

<img width="1000" alt="스크린샷 2020-12-11 오후 10 22 42" src="https://user-images.githubusercontent.com/62995632/101908568-9c4e2b80-3bff-11eb-9612-3ecb32a3014c.png">

<img width="1000" alt="스크린샷 2020-12-11 오후 10 23 36" src="https://user-images.githubusercontent.com/62995632/101908572-9d7f5880-3bff-11eb-83c7-0894728cde23.png">

x1, x2에 따라 원하는 값이 잘 나오는 것을 볼 수 있음

#### Forward propagation

합쳐서 나타내면 다음과 같음

<img width="1000" alt="스크린샷 2020-12-11 오후 10 25 42" src="https://user-images.githubusercontent.com/62995632/101908703-d28bab00-3bff-11eb-9777-c1b6385f37e7.png">

- Can you find another W and b for the XOR?

## NN

multinoimial classification과 유사 -> 벡터 3개를 하나로 합쳐서 3*3 형태로 나타내어 계산했었음

두개를 하나로 합쳐서 계산할 수 있음

<img width="1000" alt="스크린샷 2020-12-11 오후 10 27 49" src="https://user-images.githubusercontent.com/62995632/101908980-3910c900-3c00-11eb-9ff4-59f8558a54a4.png">

수식으로 바꿀 때

<img width="1000" alt="스크린샷 2020-12-11 오후 10 28 28" src="https://user-images.githubusercontent.com/62995632/101908988-3a41f600-3c00-11eb-962a-49f7b9d71e2b.png">

```python
# NN
K = tf.sigmoid(tf.matmul(X, W1) + b1)
hypothesis = tf.sigmoid(tf.matmul(K, W2) + b2)
```

- How can we learn W1, W2, B1, B2 from training data?
