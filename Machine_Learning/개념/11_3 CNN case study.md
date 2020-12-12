## CNN case study

#### Case Study: LeNet-5

<img width="1000" alt="스크린샷 2020-12-13 오전 6 18 26" src="https://user-images.githubusercontent.com/62995632/101995113-07385900-3d0b-11eb-955c-1868b1346b85.png">

#### Case Study: AlexNet

<img width="1000" alt="스크린샷 2020-12-13 오전 6 20 17" src="https://user-images.githubusercontent.com/62995632/101995141-48c90400-3d0b-11eb-8c41-503add8bd0e4.png">

96개의 filter 사용 -> depth = 96

<img width="1000" alt="스크린샷 2020-12-13 오전 6 21 26" src="https://user-images.githubusercontent.com/62995632/101995167-78780c00-3d0b-11eb-94ad-99b307dafb76.png">

<img width="1000" alt="스크린샷 2020-12-13 오전 6 22 58" src="https://user-images.githubusercontent.com/62995632/101995188-a8bfaa80-3d0b-11eb-82f7-f797b8c81875.png">

AlexNet: ReLU 처음 사용, dropout 사용, 7개의 CNN을 합쳐서 오류를 낮춤

#### Case Study: GoogLeNet

<img width="1000" alt="스크린샷 2020-12-13 오전 6 25 21" src="https://user-images.githubusercontent.com/62995632/101995250-fe945280-3d0b-11eb-86c2-20b9c604e0ed.png">

#### Case Study: ResNet

<img width="1000" alt="스크린샷 2020-12-13 오전 6 26 10" src="https://user-images.githubusercontent.com/62995632/101995264-1c61b780-3d0c-11eb-978e-6cd9d60e7f22.png">

<img width="1000" alt="스크린샷 2020-12-13 오전 6 27 23" src="https://user-images.githubusercontent.com/62995632/101995296-55019100-3d0c-11eb-87b2-5063ea757b74.png">

AlexNet의 경우 처음 8개의 layer, ResNet은 152개의 layer를 사용 -> 학습하기 어려울 것

<img width="1000" alt="스크린샷 2020-12-13 오전 6 29 37" src="https://user-images.githubusercontent.com/62995632/101995350-a6aa1b80-3d0c-11eb-9b4b-120b26319562.png">

<img width="1000" alt="스크린샷 2020-12-13 오전 6 29 29" src="https://user-images.githubusercontent.com/62995632/101995348-a578ee80-3d0c-11eb-8cfe-b65fa12459e8.png">

-> fast forward 방식을 사용함 -> 계산을 하다가 중간에 건너뛰는 방식 -> 전체적인 깊이는 깊지만 layer가 합쳐진다고 생각할 수 있음

#### Convolution Neural Networks for Sentence Classification

<img width="1000" alt="스크린샷 2020-12-13 오전 6 32 05" src="https://user-images.githubusercontent.com/62995632/101995403-ef61d480-3d0c-11eb-947f-c8a1624b0f9c.png">

이미지 뿐만 아니라 다른 것들에도 적용이 가능함

#### Case Study Bonus: DeepMind's AlphaGo
<img width="1000" alt="스크린샷 2020-12-13 오전 6 33 18" src="https://user-images.githubusercontent.com/62995632/101995426-1a4c2880-3d0d-11eb-814e-d256ac567642.png">
