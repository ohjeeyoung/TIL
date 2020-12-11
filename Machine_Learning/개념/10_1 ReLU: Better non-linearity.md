## ReLU: Better non-linearity

#### NN for XOR

<img width="1000" alt="스크린샷 2020-12-12 오전 4 30 18" src="https://user-images.githubusercontent.com/62995632/101946805-d4229680-3c32-11eb-8258-d4a7e9200e42.png">

<img width="1000" alt="스크린샷 2020-12-12 오전 4 30 36" src="https://user-images.githubusercontent.com/62995632/101946809-d4bb2d00-3c32-11eb-8bc0-9ed7c3813809.png">

#### Let's go deep & wide!

<img width="1000" alt="스크린샷 2020-12-12 오전 4 32 15" src="https://user-images.githubusercontent.com/62995632/101946952-09c77f80-3c33-11eb-9db8-c28ccc336205.png">

layer가 늘어나면 늘어날수록 backpropagation을 이용해도 학습이 안됨

9_2의 chain rule을 이용해서 구하는데 우리가 구하려는 이전에도 layer가 여러개가 있으면
sigmoid 함수를 통과하면서 값이 무조건 0~1사이로 고정됨

chain rule을 적용하면 뒤로갈수록 굉장히 작은 값이 됨 -> 0에 가까워짐 -> 뒤의 layer가 앞에 별로 영향을 미치지 않게 됨

#### Vanishing gradient(NN winter2: 1986-2006)

> 경사기울기가 사라지는 문제

최종과 가까이에 걸쳐있는 것은 경사나 기울기는 나타나지만 끝으로 갈수록 경사도가 사라진다.

#### Geoffrey Hinton's summary of findings up to today

- We used the wrong type of non-linearity.

-> sigmoid 함수를 잘못 사용했다. -> 항상 1보다 작은 값을 곱하게 되기 때문

-> 그래서 만들어진 것이 ReLU

<img width="1000" alt="스크린샷 2020-12-12 오전 4 41 55" src="https://user-images.githubusercontent.com/62995632/101947869-7db65780-3c34-11eb-8358-f5d673cc2d57.png">

0보다 작으면 activate 되지 않으므로 종료, 0보다 크면 값에 비례해서 계속 올라감

## ReLU: Rectified Linear Unit

NN에서는 activate function으로 sigmoid 대신 ReLU를 사용

<img width="1000" alt="스크린샷 2020-12-12 오전 4 44 08" src="https://user-images.githubusercontent.com/62995632/101947996-afc7b980-3c34-11eb-9d26-52a594130926.png">

#### ReLU

기존의 layer를 sigmoid 대신 relu로 바꾸지만 마지막 layer에는 sigmoid를 사용한다.

-> 최종값은 0~1 사이로 나와야하기 때문

<img width="1000" alt="스크린샷 2020-12-12 오전 4 47 37" src="https://user-images.githubusercontent.com/62995632/101948329-2cf32e80-3c35-11eb-9914-4113dcb5e5c7.png">

-> Works very well

<img width="1000" alt="스크린샷 2020-12-12 오전 4 48 42" src="https://user-images.githubusercontent.com/62995632/101948513-7fcce600-3c35-11eb-9573-31505eb7c348.png">

#### Activation Functions

<img width="1000" alt="스크린샷 2020-12-12 오전 4 48 58" src="https://user-images.githubusercontent.com/62995632/101948505-7d6a8c00-3c35-11eb-9526-c5d1fd9c12b9.png">

- Leaky ReLU: 0 이하를 조금 살림
- ELU: max(0.1x, x)에서 0.1로 fix 시키지 말고 원하는 값을 줌

#### Activation functions on CIFAR-10

![스크린샷 2020-12-12 오전 4 51 47](https://user-images.githubusercontent.com/62995632/101948645-c1f62780-3c35-11eb-95d8-810181e1990a.png)

sigmoid는 아예 작동하지 않고, relu의 성능이 괜찮다.
