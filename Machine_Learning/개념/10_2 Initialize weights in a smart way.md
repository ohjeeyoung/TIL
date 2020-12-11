## Initialize weights in a smart way

-> Hinton 교수님이 얘기했던 제대로 동작하지 않는 이유 중에 초기값을 멍청하게 줬다는 이유가 있었음

- We initialized the weights in a stupid way

우리가 항상 초기값을 random으로 줬기 때문에 같은 식에서도 결과가 조금씩 다르게 나옴

#### Set all initial weights to 0

<img width="1000" alt="스크린샷 2020-12-12 오전 4 56 37" src="https://user-images.githubusercontent.com/62995632/101949113-74c68580-3c36-11eb-9804-39c5593e8710.png">

w가 chain rule을 할 때 사용되는데 x의 기울기가 항상 0이 되고 그 뒤도 전부 0이 되어버림

#### Need to set the initial weight values wisely

- Not all 0's
- Challenging issue
- Hinton et al.(2006) "A Fast Learning Algorithm for Deep Belief Nets" - Restricted Boatman Machine(RBM)

## RBM Structure

<img width="1000" alt="스크린샷 2020-12-12 오전 5 00 27" src="https://user-images.githubusercontent.com/62995632/101949474-00d8ad00-3c37-11eb-9b70-b806dae28b87.png">

restriction이라고 불리는 이유는 앞뒤 layer들끼리만 연결이 되고 나머지는 연결되지 않았기 때문

목적은 입력을 재생산해 내는 것

#### Recreate Input
<img width="1000" alt="스크린샷 2020-12-12 오전 5 03 39" src="https://user-images.githubusercontent.com/62995632/101949774-780e4100-3c37-11eb-8cbe-e50d07d98a0a.png">

forward에서 앞에서 주었던 값과 backward에서 뒤의 결과에서 역연산을 해서 나온 앞의 값을 비교한다.

그 차가 최저가 되도록 weight 값을 조절한다.

en/decoder라고도 함.

#### How can we use RBM to initialize weights?

- Apply the RBM idea on adjacent two layers as a pre-training step
- Continue the first process to all layers
- This will set weights
- Example: Deep Belief Network -> Weight initialized by RBM

#### Pre-training

<img width="1000" alt="스크린샷 2020-12-12 오전 5 09 15" src="https://user-images.githubusercontent.com/62995632/101950258-4e094e80-3c38-11eb-9a78-fada53cb7e92.png">

인접한 두개의 layer를 en/decoder를 반복하면서 weight를 다 학습시킴

이를 전체로 반복시켜서 나온 것을 초기값으로 설정

#### Fine Tuning

<img width="1000" alt="스크린샷 2020-12-12 오전 5 11 54" src="https://user-images.githubusercontent.com/62995632/101950393-90cb2680-3c38-11eb-9ef8-ae9b7221b78b.png">

x 데이터를 넣고 실제 labels를 학습

이미 가지고 있는 weight이 훌륭해 빨리 학습이 되기 때문에 training이 아닌 tuning이라는 명칭을 붙임

#### Good news

<img width="1000" alt="스크린샷 2020-12-12 오전 5 13 00" src="https://user-images.githubusercontent.com/62995632/101950880-5e6df900-3c39-11eb-8f9c-c2fbd19002e3.png">

RBM을 굳이 안주고 매우 간단한 값으로 초기화를 해도 된다.

Xavier initialization: 하나의 node에 몇개 입력, 몇개 출력인지에 맞게 비례해서 초기값을 준다.

## Xavier/He initialization

- Makes sure the weights are 'just right', not too small, not too big
- Using number of input (fan_in) and output (fan_out)

```python
# Xavier initialization
# Glorot et al. 2010
W = np.random.randn(fan_in, fan_out)/np.sqrt(fan_in)

# He et al. 2015
W = np.random.randn(fan_in, fan_out)/np.sqrt(fan_in/2)
```

#### prettytensor implementation

<img width="1000" alt="스크린샷 2020-12-12 오전 5 17 39" src="https://user-images.githubusercontent.com/62995632/101950978-8a897a00-3c39-11eb-853d-504e7b6b6a15.png">

#### Activation functions and initialization on CIFAR-10

<img width="1000" alt="스크린샷 2020-12-12 오전 5 18 16" src="https://user-images.githubusercontent.com/62995632/101950982-8bbaa700-3c39-11eb-9408-b4bf3d70d9ea.png">

#### Still an active area of research

- We don't know how to initialize perfect weight values, yet
- Many new algorithms

-> Batch normalization

-> Layer sequential uniform variance

-> ...
