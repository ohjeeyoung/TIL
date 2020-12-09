#### Where is sigmoid?

y의 예측값이 2.0, 1.0, 0.1 등 다양하게 나오는 것이 아니라 0~1 사이로 나오는 것이 목표

<img width="1000" alt="스크린샷 2020-12-09 오후 5 47 18" src="https://user-images.githubusercontent.com/62995632/101606536-d6280200-3a46-11eb-8464-514f0d6b7d1a.png">

## Sigmoid?

p가 0~1 사이의 값이고 p의 합이 1이 될 때 우리는 p를 확률로 볼 수 있고, 이를 가능하게 하는 것을 softmax라고 한다.

<img width="1000" alt="스크린샷 2020-12-09 오후 5 47 44" src="https://user-images.githubusercontent.com/62995632/101606542-d7f1c580-3a46-11eb-9f70-d615741599c7.png">

나와있는 n개의 값을 softmax에 넣으면 p로 변환
1. p가 0~1 사이의 값
2. p의 sum이 1

<img width="1000" alt="스크린샷 2020-12-09 오후 5 48 14" src="https://user-images.githubusercontent.com/62995632/101606547-d922f280-3a46-11eb-94cf-8dfcf0c977a4.png">


그 중에 하나만 골라서 말할 때는 one-hot encoding을 이용해 제일 큰 값을 1로 보고 고름

<img width="1000" alt="스크린샷 2020-12-09 오후 5 48 48" src="https://user-images.githubusercontent.com/62995632/101606550-d9bb8900-3a46-11eb-85a1-1dc57125eb57.png">

## Cost function

L이 label(Y 실제값), S(y)는 softmax(예측값) 사이에 차이가 얼마나 나는지 cross-entropy를 사용해 구함

<img width="1000" alt="스크린샷 2020-12-09 오후 5 53 16" src="https://user-images.githubusercontent.com/62995632/101607052-767e2680-3a47-11eb-836d-90936a55223b.png">

## Cross-entropy cost function

예측값에 대해서 cross-entropy를 이용해 예측값의 정확도를 구함

<img width="1000" alt="스크린샷 2020-12-09 오후 5 58 27" src="https://user-images.githubusercontent.com/62995632/101607820-5a2eb980-3a48-11eb-8ffb-c6cd9b6059a8.png">

![스크린샷 2020-12-09 오후 5 58 47](https://user-images.githubusercontent.com/62995632/101607823-5ac75000-3a48-11eb-8656-81bd5dba9fda.png)

## Logistic cost VS cross entropy

많은 형태의 training data가 있을 때는 어떤 것을 사용할지 생각해보아야 함

C와 D function은 결국 같은 것

H(x) = S, y = L

<img width="1000" alt="스크린샷 2020-12-09 오후 6 02 49" src="https://user-images.githubusercontent.com/62995632/101608349-f1940c80-3a48-11eb-91ea-d28eee5b3533.png">

## Cost function

여러개의 training set이 있을 때는 전체의 거리를 차이를 구한 다음 그 합을 갯수로 나누어 평균을 구함

<img width="1000" alt="스크린샷 2020-12-09 오후 6 03 57" src="https://user-images.githubusercontent.com/62995632/101608352-f2c53980-3a48-11eb-966e-f9d9b1db5e65.png">

#### Gradient descent

loss/cost function이 밑의 그래프처럼 생겼다고 가정할 때, 어떤 점을 선택해도 경사면(미분, 기울기)을 타고 내려가면 최소값에 도달할 수 있음

<img width="1000" alt="스크린샷 2020-12-09 오후 6 05 53" src="https://user-images.githubusercontent.com/62995632/101608564-33bd4e00-3a49-11eb-8124-93e467975e17.png">
