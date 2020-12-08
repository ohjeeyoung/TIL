## Regression(HCG)
> Regression: 값을 예측

- H(Hypothesis): 가설 세우기

![CodeCogsEqn (6)](https://user-images.githubusercontent.com/62995632/93719105-73e40180-fbbb-11ea-945d-3491a7f98730.gif)


- C(Cost): 학습 데이터와 가설을 세운 선이 얼마나 가깝고 먼지 측정한 뒤 평균을 낸 것

![CodeCogsEqn (1)](https://user-images.githubusercontent.com/62995632/93718554-27e38d80-fbb8-11ea-8bb2-b2e05e19d00b.gif)

- G(Gradient decent): cost를 최소화하는 weight를 찾아내기 위해 사용

기울기는 cost를 미분한 값으로 나타내고, a(알파)는 step의 size(learning rate)

![CodeCogsEqn (3)](https://user-images.githubusercontent.com/62995632/93718981-9de8f400-fbba-11ea-8c89-c2af5eaa2420.gif)

## (Binary) Classification
- Regression과의 차이점

Regression: 숫자를 예측

Classification: 정해진 카테고리를 고르는 것

ex)
- Spam Detection: Spam or Ham
- Facebook feed: show or hide
- Credit Card Fraudulent Transaction detection: legitimate/fraud
- + Radiology(Malignant tumor, Benign tumor), Finance, ...

#### 0, 1 encoding
- Spam Detection: Spam(1) or Ham(0)
- Facebook feed: show(1) or hide(0)
- Credit Card Fraudulent Transaction detection: legitimate(0) or fraud(1)

#### Pass(1)/Fail(0) based on study hours
-> Linear regression을 사용할 때의 문제점
- We know Y is 0 or 1

y값이 0 또는 1로 나와야하는 linear regression에서는 다른 값이 나올 수 있다.

H(x) = Wx + b

- Hypothesis can give values large than 1 or less than 0 

- 예를 들어 50시간의 공부 끝에 pass한 사람이 있다면 가설의 식과 오차가 많이 나기때문에 pass와 fail의 기준이 잘못 설정될 수 있음

-> Logistic Hypothesis
<img width="1000" alt="스크린샷 2020-12-08 오후 8 43 58" src="https://user-images.githubusercontent.com/62995632/101479864-3317af80-3996-11eb-97f4-875ffd8d2673.png">
z = Wx

H(x) = g(z)
 
![daum_equation_1607428000665](https://user-images.githubusercontent.com/62995632/101480064-7f62ef80-3996-11eb-8740-d4a4c1597ceb.png)
