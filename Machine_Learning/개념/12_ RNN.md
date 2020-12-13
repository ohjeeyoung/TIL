# RNN

## Sequence data

- We don't understand one word only
- We understand based on the previous words + this word.(time series)
- NN/CNN cannot do this

우리가 사용하는 데이터들 중에서는 음성인식, 말하는 자연어를 보면 하나의 데이터가 아닌 sequence로 되어있음

하나의 단어만 이해해서 맥락을 이해하는 것이 아닌 그전에 얘기한 단어를 이해하며 문맥으로 파악하게 됨

series data를 NN/CNN은 할 수 없음

<img width="1000" alt="스크린샷 2020-12-14 오전 12 47 40" src="https://user-images.githubusercontent.com/62995632/102016788-fee12c00-3da5-11eb-9fe1-d58f4e0cde85.png">

series data에서 이전의 연산이 다음에도 영향을 미칠 수 있어야 함


## Recurrent Neural Network

<img width="1000" alt="스크린샷 2020-12-14 오전 12 49 51" src="https://user-images.githubusercontent.com/62995632/102016839-5da6a580-3da6-11eb-8652-f53e2ef2c6c5.png">

x가 있으면 RNN 연산을 통해 state를 계산하고 그것이 자기 입력이 되는 형태

각각의 RNN에서 y 값을 뽑아낼 수 있음

<img width="1000" alt="스크린샷 2020-12-14 오전 12 50 48" src="https://user-images.githubusercontent.com/62995632/102016849-731bcf80-3da6-11eb-9f2d-3a3809493766.png">

RNN은 state를 먼저 계산한다 -> 그 state를 통해 y 계산

이전의 state가 입력으로 이용된다

<img width="1000" alt="스크린샷 2020-12-14 오전 12 52 43" src="https://user-images.githubusercontent.com/62995632/102016894-b1b18a00-3da6-11eb-886f-d8f30ed5a179.png">

전체가 하나와 같다. 앞의 다른 것도 다 식이 같기 때문

#### (Vanilla) Recurrent Neural Network

<img width="1000" alt="스크린샷 2020-12-14 오전 12 55 40" src="https://user-images.githubusercontent.com/62995632/102016956-1c62c580-3da7-11eb-9106-27d595a831df.png">

WX를 많이 사용해서 계산

W에 따라 y나 h의 개수가 달려있음

<img width="1000" alt="스크린샷 2020-12-14 오전 12 56 21" src="https://user-images.githubusercontent.com/62995632/102016977-3bf9ee00-3da7-11eb-89c0-19988aa6f19a.png">

어떤 것이든 다 W의 값을 동일하게 계산

#### Character-level language model example

> 현재의 글씨가 있을 때 그 다음 글씨를 맞추는 것

<img width="1000" alt="스크린샷 2020-12-14 오전 1 02 00" src="https://user-images.githubusercontent.com/62995632/102017075-fc7fd180-3da7-11eb-84da-f56f2de1828c.png">

한글자씩 입력이 된다고 생각

문자의 일부만 입력이 되었을 때도 다음에 오는 것을 단어를 예측해줄 수 있기를 바람

<img width="1000" alt="스크린샷 2020-12-14 오전 1 04 53" src="https://user-images.githubusercontent.com/62995632/102017161-87f96280-3da8-11eb-901c-84099ff8da61.png">

one-hot encoding을 통해 들어온 값에 해당하는 자리에 1로 계산

위의 공식으로 계산을 하면 첫번째인 h가 입력되었을 때는 앞의 layer가 없으므로 그때는 초기화값 0이라고 생각

hidden layer에 W_hh, input layer에 W_xh를 곱한 값을 더해 다음값의 hidden layer 결과를 만든다.

y를 뽑아낼 때 W_hy에 곲해서 구할 수 있음 -> 여기서는 label이 [h, e, l, o]로 4가지 이므로 4가지 중에 하나로 나옴

softmax를 취하면 가장 큰 것을 선택하면 그 것에 해당하는 index

<img width="1000" alt="스크린샷 2020-12-14 오전 1 07 54" src="https://user-images.githubusercontent.com/62995632/102017218-e292be80-3da8-11eb-9731-50f83c4b9536.png">

원하는 것과 나오는 것이 다를 수 있음 -> cost는 softmax에 cost함수를 넣어 계산

#### RNN applications

https://github.com/TensorFlowKR/awesome_tensorflow_implementations

- Language Modeling
- Speech Recognition
- Machine Translation
- Conversation Modeling/Question Answering
- Image/Video Captioning
- Image/Music/Dance Generation

#### Recurrent Networks offer a lot of flexibility

RNN으로 여러가지 형태 구성 가능

<img width="1000" alt="스크린샷 2020-12-14 오전 1 14 19" src="https://user-images.githubusercontent.com/62995632/102017341-baf02600-3da9-11eb-9fea-c188a54c8dee.png">

- one to one(하나의 입력, 하나의 출력): Vanilla Neural Networks
- one to many(하나의 입력, '나는 모자를 쓰고 있네'같은 series data 출력): e.g. Image Captioning(image -> sequence of words)
- many to one(문자열 입력, 하나 출력): e.g. Sentiment Classification(sequence of words -> sentiment)
- many to many(문자열 입력, 문자열 출력): e.g. Machine Translation(seq of words -> seq of words)
- many to many(프레임 여러개, 설명 출력): e.g. Video classification on frame level

#### Multi-Layer RNN

<img width="1000" alt="스크린샷 2020-12-14 오전 1 18 29" src="https://user-images.githubusercontent.com/62995632/102017445-49fd3e00-3daa-11eb-899e-b95adab70f8f.png">

layer를 여러개 둘 수 있으면 더 복잡한 학습도 가능

#### Training RNNs is challenging

layer가 많아지다보면 학습에 어려움이 있고 이를 극복하기 위한 모델이 있음

- Several advanced models

-> Long Short Term Memory(LSTM)

-> GRU by Cho et al.2014
