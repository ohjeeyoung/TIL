# Max pooling and others

## Pooling layer(sampling)
<img width="1000" alt="스크린샷 2020-12-13 오전 6 02 01" src="https://user-images.githubusercontent.com/62995632/101994815-cfc8ad00-3d08-11eb-86df-27b841a8aee2.png">

<img width="1000" alt="스크린샷 2020-12-13 오전 6 02 10" src="https://user-images.githubusercontent.com/62995632/101994816-d0614380-3d08-11eb-964e-b09349866c5f.png">

<img width="1000" alt="스크린샷 2020-12-13 오전 6 02 30" src="https://user-images.githubusercontent.com/62995632/101994817-d0f9da00-3d08-11eb-9d44-c78af08bc7e6.png">

이미지에서 filter를 처리한 뒤 conv layer를 만들었음

한 layer만 뽑아내서 크기를 줄이는 reize(sampling)를 함

이렇게 각각의 layer를 하나씩 pooling 한 다음 다시 쌓기


#### MAX POOLING

<img width="1000" alt="스크린샷 2020-12-13 오전 6 06 23" src="https://user-images.githubusercontent.com/62995632/101994886-5a111100-3d09-11eb-8465-2b599b56d12b.png">

4x4 이미지가 있을 때 filter의 개념을 사용하는데 2x2를 사용한다고 하자.

위의 예시에서는 2x2 output이 나옴

그 값을 옮길 때 평균을 낼 수도 있고 여러가지 방법을 생각해볼 수 있다.

가장 큰 값을 고르는 MAX POOLING을 사용해 많이 계산한다.

첫번째칸에 6, 두번째에 8, 아래 첫번째에 3, 아래 두번째에 4

이것을 sampling이라고 부르는 이유는 전체값 중에서 하나만 뽑기 때문이다.

#### Fully Connected Layer(FC layer)

- Contains neurons that connect to the entire input volume, as in ordinary Neural Networks

conv, relu, pool 내가 원하는대로 쌓으면 된다.

마지막에 보통 pooling을 하면서 softmax classifier 형태로 나오게 된다.

#### [ConvNetJS demo: training on CIFAR-10]

http://cs.standford.edu/people/karpathy/convnetjs/demo/cifar10.html

레이어 동작하는 것 볼 수 있음
