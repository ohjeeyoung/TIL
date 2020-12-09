## Application & Tips: Learning and test data sets

### Performance evaluation: is this good?

#### Evaluation using training set?

- 100% correct(accuracy)
- Can memorize

데이터(training set)로 모델을 학습시킨 뒤, 다시 training set으로 물어본다면 거의 100% 완벽할 것이다.

매우 안좋은 방법이고, 시험을 본 뒤 똑같은 문제로 다시 시험을 보는 것과 같다.

#### Training and test sets

학습시키려는 것을 training data로 하고, 숨겨진 test set을 만든다.(training data와 test set 나누기)

training data으로 모델을 만든 뒤, test set으로 실험해본다.

예측값과 실제 모델을 통해 나온 값이 같은지 확인한다.

#### Training, validation and test sets

<img width="1000" alt="스크린샷 2020-12-10 오전 4 36 02" src="https://user-images.githubusercontent.com/62995632/101678383-3b104600-3aa1-11eb-8874-c285e59db839.png">

보통은 training과 testing set을 나누는 것이 일반적

a(alpha, running_rate)과 람다(regularlization 강화)를 튜닝할 필요가 있을 때, training set을 둘로 나누어
training set으로 학습시킨뒤 validation set으로 위의 값들을 조정함(모의시험)

Testing set은 숨겨져 있음

Validation set이 실전 모의고사 느낌

#### Online learning

data set이 굉장히 많아 한 번에 학습시키기 어려울 때 사용

data set이 100만개라 가정 -> 10만개씩 잘라서 model에 학습시킴(이때 학습시킨 데이터는 남아있어야 함)

#### MINIST Dataset

사람들이 손으로 적은 숫자를 정확하게 컴퓨터가 인식할 수 있는지 test하는 data set

- train-images-idx3-ubyte.gz: training set images (9912422 bytes)
- train-labels-idx1-ubyte.gz: training set labels (28881 bytes)
- t10k-images-idx3-ubyte.gz: test set images (1648877 bytes)
- t10k-labels-idx1-ubyte.gz: test set labels (4542 bytes)

위처럼 training set image와 그 결과(labels), test set image와 그 결과(labels)로 나뉘는 것을 알 수 있다.

## Accuracy
- How many of your predictions are correct?
- 95% ~ 99%?
- Check out the lab video
