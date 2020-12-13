# CNN Basics

## CNN

<img width="1000" alt="스크린샷 2020-12-13 오후 4 10 08" src="https://user-images.githubusercontent.com/62995632/102005632-af770d80-3d5d-11eb-8900-7ec07569a478.png">

이미지와 텍스트분류 등 여러 분야에서 활용 가능

크게 3가지 기능으로 나누면
1. 입력된 이미지, 벡터를 convolution을 통해 filter를 사용, convolution layer로 값들을 뽑아냄
2. 뽑아낸 데이터가 많으므로 크기를 줄이는 subsampling
3. 일반적인 fully connected network를 통해 값을 분류


#### CNN for CT images

#### Convolution layer and max pooling

<img width="1000" alt="스크린샷 2020-12-13 오후 4 12 16" src="https://user-images.githubusercontent.com/62995632/102005670-fc5ae400-3d5d-11eb-85ce-e858e177c8b6.png">

1. 주어진 이미지 벡터에 filter를 stride만큼 움직이면서 한개의 값을 뽑아냄
2. samplig을 통해 subsampling을 함


#### Simple convolution layer
> Stride: 1x1

<img width="1000" alt="스크린샷 2020-12-13 오후 4 13 36" src="https://user-images.githubusercontent.com/62995632/102005694-2b715580-3d5e-11eb-86b2-1a002d651fb1.png">

#### Toy image

```python
sess = tf.InteractiveSession()
image = np.array([[[[1],[2],[3]],
                    [[4],[5],[6]],
                    [[7],[8],[9]]]], dtype=np.float32)

print(image.shape)
plt.imshow(image.reshape(3,3), cmap='Greys')
```
(1, 3, 3, 1)

시각화 한 것

#### Simple convolution layer
> Image: 1,3,3,1 image, Filter: 2,2,1(color),1(filter개수), Stride: 1x1, Padding: VALID

<img width="1000" alt="스크린샷 2020-12-13 오후 4 21 12" src="https://user-images.githubusercontent.com/62995632/102005819-40021d80-3d5f-11eb-8d5b-3f0faedb6908.png">

```python
# print("imag:\n", image)
print("image.shape", image.shape)
weight = tf.constant([[[[1.]],[[1.]]],
                      [[[1.]],[[1.]]]])
print("weight.shape", weight.shape)
conv2d = tf.nn.conv2d(image, weight, strides=[1, 1, 1, 1], padding='VALID')
conv2d_img = conv2d.eval()
print("conv2d_img.shape", conv2d_img.shape)
conv2d_img = np.swapaxes(conv2d_img, 0, 3)
for i, one_img in enumerate(conv2d_img):
    print(one_img.reshape(2,2))
    plt.subplot(1,2,i+1), plt.imshow(one_img.reshpae(2,2), cmap='gray')
```

image.shape(1, 3, 3, 1)

weight.shape(2, 2, 1, 1)

conv2d_img.shape(1, 2, 2, 1)

[[ 12. 16.]

 [ 24. 28.]]
 
 
 #### Simple convolution layer
 > Image: 1,3,3,1 image, Filter: 2,2,1,1, Stride: 1x1, Padding: SAME
 
 <img width="1000" alt="스크린샷 2020-12-13 오후 4 37 21" src="https://user-images.githubusercontent.com/62995632/102006072-7e98d780-3d61-11eb-8245-4c4abd0304bb.png">
 
 Padding: SAME 이라는 것은 Stride 값이 어떻든 상관없이 처음 이미지의 크기과 convolution layer의 크기가 같음
 
 ```python
 # print("imag:\n", image)
print("image.shape", image.shape)

weight = tf.constant([[[[1.]],[[1.]]],
                      [[[1.]],[[1.]]]])
print("weight.shape", weight.shape)
conv2d = tf.nn.conv2d(image, weight, strides=[1, 1, 1, 1], padding='SAME')
conv2d_img = conv2d.eval()
print("conv2d_img.shape", conv2d_img.shape)
conv2d_img = np.swapaxes(conv2d_img, 0, 3)
for i, one_img in enumerate(conv2d_img):
    print(one_img.reshape(3,3))
    plt.subplot(1,2,i+1), plt.imshow(one_img.reshpae(3,3), cmap='gray')
 ```
 
 image.shape(1, 3, 3, 1)
 
 weight.shape(2, 2, 1, 1)
 
 conv2d_img.shape(1, 3, 3, 1)
 
 [[ 12. 16. 9.]
 
  [ 24. 28. 15.]
  
  [ 15. 17. 9.]]
  
#### 3 filters(2,2,1,3)

하나의 이미지에서 3개 filter를 쓰면 3개의 이미지가 나옴

<img width="1000" alt="스크린샷 2020-12-13 오후 4 44 18" src="https://user-images.githubusercontent.com/62995632/102006183-7f7e3900-3d62-11eb-8d66-fe8256f0335b.png">

## Max Pooling

데이터를 subsampling 하는 것

<img width="1000" alt="스크린샷 2020-12-13 오후 4 46 33" src="https://user-images.githubusercontent.com/62995632/102006228-c5d39800-3d62-11eb-9277-b60a1bec3db8.png">

padding='SAME'이므로 테두리에 0을 채울 것

#### MNIST image loading

<img width="1000" alt="스크린샷 2020-12-13 오후 4 47 40" src="https://user-images.githubusercontent.com/62995632/102006252-f1568280-3d62-11eb-8c88-a60874745ef9.png">

#### MNIST Convolution layer

<img width="1000" alt="스크린샷 2020-12-13 오후 4 51 22" src="https://user-images.githubusercontent.com/62995632/102006296-73df4200-3d63-11eb-807c-399ca09218e0.png">

reshape(-1, 28, 28, 1) -> -1은 n개의 이미지를 넣을 것이니 끝에서 크기를 확인, 28*28크기의 1색깔의 이미지

W1 -> (3, 3, 1, 5) -> 3x3 filter 에서 1개의 color, 5개 filter 사용

stride가 2x2이므로 출력은 14x14가 될 것

이미지 하나에서 convolution을 조금씩 다르게 해서 이미지 5개를 뽑아냈다.

#### MNIST Max pooling

<img width="1000" alt="스크린샷 2020-12-13 오후 4 53 05" src="https://user-images.githubusercontent.com/62995632/102006326-b6088380-3d63-11eb-8a71-bb2eed4872c5.png">

stride 때문에 14x14로 들어온 이미지가 다시 7x7로 될 것

subsampling을 하면서 해상도는 조금씩 떨어짐
