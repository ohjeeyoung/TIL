## Convolutional Neural Networks

<img width="1000" alt="스크린샷 2020-12-13 오전 5 28 28" src="https://user-images.githubusercontent.com/62995632/101994216-0ea83400-3d04-11eb-8e30-63b51f0459bd.png">

고양이 실험에서 시작

고양이에게 그림을 보여줬더니 모든 뉴런들이 반응하는 것이 아니라 부분적으로 반응함

입력을 나눠갖는 것에 착안해서 시작 됨

<img width="1000" alt="스크린샷 2020-12-13 오전 5 30 40" src="https://user-images.githubusercontent.com/62995632/101994249-5af37400-3d04-11eb-8fca-c7b49118e424.png">

하나의 이미지가 있으면 이것을 잘라서 각각의 입력으로 넘기면 이 층을 convolutional layer라고 함

relu 함수를 중간에 넣어서 conv, pool 등 여러번 반복을 통해 마지막으로 Fully connected NN을 구성

그림의 차이를 labeling 하는 역할을 수행


### Start with an image (width * hight * depth)

<img width="1000" alt="스크린샷 2020-12-13 오전 5 32 00" src="https://user-images.githubusercontent.com/62995632/101994277-a6a61d80-3d04-11eb-85ed-9d3a59cecd52.png">

32 * 32 * 3 형태의 이미지가 있다고 가정

### Let's focus on a small area only (5 * 5 * 3)

<img width="1000" alt="스크린샷 2020-12-13 오전 5 33 38" src="https://user-images.githubusercontent.com/62995632/101994311-d9501600-3d04-11eb-962e-f4ebc858777e.png">

<img width="1000" alt="스크린샷 2020-12-13 오전 5 34 53" src="https://user-images.githubusercontent.com/62995632/101994334-12888600-3d05-11eb-9b6b-57ba53d49807.png">

전체 이미지를 보는 것이 아니라 이미지의 일부분만 우선 처리함 -> filter로 설명 -> 크기는 정의 가능

색깔은 항상 이미지의 depth와 같이 처리

### Get one number using the filter

<img width="1000" alt="스크린샷 2020-12-13 오전 5 38 52" src="https://user-images.githubusercontent.com/62995632/101994404-82970c00-3d05-11eb-8418-c5e1a0041dfc.png">

filter의 역할은 궁극적으로 한 값을 만들어내는 것 -> Wx + b를 이용 -> ReLU를 사용해도 됨

### Let's look at other areas with the same filter (w)

<img width="1000" alt="스크린샷 2020-12-13 오전 5 40 22" src="https://user-images.githubusercontent.com/62995632/101994429-b5410480-3d05-11eb-8038-d4b6acc11a2d.png">

w는 같은 값을 갖고 점차적으로 다른 이미지들도 확인

몇번 확인해야 하는지 구해야 함

- stride: 1

<img width="1000" alt="스크린샷 2020-12-13 오전 5 41 47" src="https://user-images.githubusercontent.com/62995632/101994451-e9b4c080-3d05-11eb-9f2e-3c2b0d7c63da.png">

옆으로 한칸씩 움직여서(stride) 5x5가 됨

stride 크기만큼 옆으로 움직임

- stride: 2

<img width="1000" alt="스크린샷 2020-12-13 오전 5 43 33" src="https://user-images.githubusercontent.com/62995632/101994504-30a2b600-3d06-11eb-8bd4-5d08a6c72216.png">

3*3의 output이 나옴

- output을 공식화

<img width="1000" alt="스크린샷 2020-12-13 오전 5 44 45" src="https://user-images.githubusercontent.com/62995632/101994531-516b0b80-3d06-11eb-84b3-8fb721ae7d97.png">

이미지와 filter size에 따라서 어떤 stride가 가능한지 알 수 있음

stride가 클수록 정보를 잃어버리기 쉬움

#### In practice

<img width="1000" alt="스크린샷 2020-12-13 오전 5 47 49" src="https://user-images.githubusercontent.com/62995632/101994583-c0e0fb00-3d06-11eb-8517-bb003d572828.png">

결과: 7x7 output

padding: 테두리에다가 0을 두르는 가상의 값이 있다고 생각함

1. 그림이 급격하게 작아지는 것을 방지
2. 모서리를 네트워크에 알려주고자 함

<img width="1000" alt="스크린샷 2020-12-13 오전 5 48 50" src="https://user-images.githubusercontent.com/62995632/101994629-27feaf80-3d07-11eb-8920-96ca289f2c39.png">

padding을 사용해 입력의 이미지(7x7)와 출력의 이미지(7x7) size가 같아지는 것을 일반적으로 사용함

### Swiping the entire image

<img width="1000" alt="스크린샷 2020-12-13 오전 5 52 36" src="https://user-images.githubusercontent.com/62995632/101994670-6bf1b480-3d07-11eb-81a0-151255201bbc.png">

filter 여러개를 동시에 적용시킴 -> 각각의 filter는 weight이 다르기 때문에 조금씩 다르게 나옴

activation maps(28, 28, 6(filter의 개수))

## Convolution layers

<img width="1000" alt="스크린샷 2020-12-13 오전 5 54 54" src="https://user-images.githubusercontent.com/62995632/101994722-c723a700-3d07-11eb-9c67-aacae0c697e8.png">

#### How many weight variables? How to set them?

앞의 예시에서는 빨간색은 5x5x3x6 weight의 개수, 노란색은 5x5x6x10번 사용
