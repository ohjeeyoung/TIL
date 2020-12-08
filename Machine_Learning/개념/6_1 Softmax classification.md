## Softmax classification: Multinomial classification

> 여러개의 class가 있을 때 값을 예측하는 것

#### Logistic regression

분류해야 될 데이터가 ㅁ와 x 일때 Logistic classification을 학습시킨다는 것은 ㅁ와 x를 구분하는 것을 구하는 것

<img width="1000" alt="스크린샷 2020-12-09 오전 1 51 41" src="https://user-images.githubusercontent.com/62995632/101515009-3d9b6e80-39c1-11eb-9800-7dfc0e901a9f.png">

## Multinomial classification

<img width="1000" alt="스크린샷 2020-12-09 오전 1 54 13" src="https://user-images.githubusercontent.com/62995632/101515275-84896400-39c1-11eb-9ade-54700eddb369.png">

A, B, C 이렇게 3개로 분류가 될 때

<img width="1000" alt="스크린샷 2020-12-09 오전 1 54 34" src="https://user-images.githubusercontent.com/62995632/101515277-8521fa80-39c1-11eb-8ab1-c299d812f6d6.png">

A or not, B or not, C or not 처럼 binary classification을 이용해 구현할 수 있음

X가 주어졌을 때 위의 결과로 Y를 예측할 수 있음


앞에서 변수가 여러개일때 행렬 곱셈으로 구했음 -> 독립적으로 계산하면 번거롭고 계산이 많음

<img width="1000" alt="스크린샷 2020-12-09 오전 1 58 13" src="https://user-images.githubusercontent.com/62995632/101515800-198c5d00-39c2-11eb-8ac2-cd05acbc97bb.png">

-> W를 하나로 합쳐서 행렬곱셈

<img width="1000" alt="스크린샷 2020-12-09 오전 2 00 08" src="https://user-images.githubusercontent.com/62995632/101516229-915a8780-39c2-11eb-8526-9ca53976d858.png">

마지막 값이 우리가 구하려고 했던 가설 H(X)

하나의 벡터로 처리하면 한번에 계산 가능, 독립된 classification처럼 동작하게 됨

sigmoid를 각각 적용해서 계산 해야함 
