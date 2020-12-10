## Deep Neural Nets for Everyone

#### Ultimate dream: thinking machine

뉴런이 단순하게 동작이 됨

input에 따라 들어온 값들을 계산한 뒤 합치면 그 값이 일정값 이상이면 반응을 하고 아니면 반응하지 않음

#### Activation Functions

뉴런을 본따서 만든 함수

<img width="1000" alt="스크린샷 2020-12-11 오전 5 18 45" src="https://user-images.githubusercontent.com/62995632/101825068-775ea780-3b70-11eb-9db4-2f5016134ac3.png">


#### Logistic regression units

여러개의 출력을 동시에 낼 수 있는 기계가 될 수 있음

<img width="1000" alt="스크린샷 2020-12-11 오전 5 21 19" src="https://user-images.githubusercontent.com/62995632/101825242-b7be2580-3b70-11eb-88cc-e4b4fd6e41f0.png">

#### (Simple) AND/OR problem: linearly separable?

<img width="1000" alt="스크린샷 2020-12-11 오전 5 23 44" src="https://user-images.githubusercontent.com/62995632/101825552-16839f00-3b71-11eb-9120-0b930ae1619f.png">

and/or까지는 해결했음

#### (Simple) XOR proble: linearly separable?

<img width="1000" alt="스크린샷 2020-12-11 오전 5 23 51" src="https://user-images.githubusercontent.com/62995632/101825558-18e5f900-3b71-11eb-9ed0-0dc617600ed9.png">

어떻게 선을 그어도 xor 구분이 불가능함 -> 정확도가 항상 50%

#### Perceptrons(1969)
- We need to use MLP, multilayer perceptrons(multilayer neural nets)
- No one on earth had found a viable way to train MLPs good enough to learn such simple functions.

한개가 아닌 여러개를 합쳐(MLP) 풀 수 있다.

하지만 각각의 W, b를 학습시키는 것이 불가능함

#### Backpropagation

각각의 W, b가 있으면 주어진 입력을 가지고 출력을 만들어냄

우리가 가진 값과 틀린 경우에는 W, b를 조절함

에러를 발견하면 다시 뒤로 보내서 조절하는 알고리즘

<img width="1000" alt="스크린샷 2020-12-11 오전 5 29 43" src="https://user-images.githubusercontent.com/62995632/101826114-e4267180-3b71-11eb-83ec-28e7eda3312a.png">

#### Convolutional Neural Networks

<img width="1000" alt="스크린샷 2020-12-11 오전 5 30 34" src="https://user-images.githubusercontent.com/62995632/101826251-1506a680-3b72-11eb-960c-29e6f48c90f9.png">

<img width="1000" alt="스크린샷 2020-12-11 오전 5 32 33" src="https://user-images.githubusercontent.com/62995632/101826421-5303ca80-3b72-11eb-8dd2-5bda4fadd4ec.png">

고양이가 이미지를 봤을 때 뉴런들 중 일부만 활성화가 되고 다른 그림을 주면 다른 형태의 뉴런만 활성화가 됨

동시에 그림 전체를 보는 것이 아니라 일부의 신경망이 활성화 되는 것

그림을 한번에 입력시키는 것이 아니라 그림을 잘라서 부분부분을 보낸 뒤 나중에 합치는 것

90% 이상의 정확도를 보임

#### A BIG problem

- Backpropagation just did not work well for normal neural nets with many layers

몇 개의 레이어에서는 잘 동작했지만 실제로 구하고 싶었던 복잡한 레이어에서는 작동하지 않음

에러를 뒤로 보낼 때 갈수록 약해져서 학습이 불가능해짐

- Other rising machine learning algorithms: SVM, RandomForest, etc.
- 1995 "Comparison of Learning Algorithms For Handwritten Digit Recognition" by LeCun et al.
found that this new approach worked better
