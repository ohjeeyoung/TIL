## Tensor Manipulation

#### Simple ID array and slicing

1차원 array 예로 김밥을 생각해볼 수 있다.

```python
t = np.array([0., 1., 2., 3., 4., 5., 6.])
pp.print(t)
print(t.ndim) # rank 1차원
print(t.shape) # shape
print(t[0], t[1], t[-1])
print(t[2:5], t[4:-1])
print(t[:2]. t[3:])
```

#### 2D Array

```python
t = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]])
pp.print(t)
print(t.ndim) # rank 2차원
print(t.shape) # shape (4, 3)
```

## Shape, Rank Axis

- Rank를 아는 방법: 앞의 [ 갯수가 rank

```python
t = tf.constant([1,2,3,4])
tf.shape(t).eval()
```

결과: array([4], dtype=int32)

```python
t = tf.constant([[1,2],[3,4]])
tf.shape(t).eval()
```

결과: array([2, 2], dtype=int32)

```python
t = tf.constant([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]]])
tf.shape(t).eval()
```

결과: array([1, 2, 3, 4], dtype=int32)

rank = 4 -> 4개의 축이 있다고 보면 됨

제일 바깥이 axis = 0, 안쪽으로 갈수록 커짐

#### Matmul VS multiply

```python
matrix1 = tf.constant([[1., 2.], [3., 4.]])
matrix2 = tf.constant([[1.], [2.]])
print("Matrix 1 shape", matrix1.shape)
print("Matrix 2 shape", matrix2.shape)
tf.matmul(matrix1, matrix2).eval()
```

결과
Matrix 1 shape (2, 2)

Matrix 2 shape (2, 1)
     
array([[ 5.],
      [ 11.]], dtype=float32)

```python
(matrix1 * matrix2).eval()
```

결과

array([[  1., 2.],

[  6., 8.]], dtype=float32)
            
일반곱셈과 행렬곱셈은 그 결과가 다르다

#### Broadcasting(WARNING)

shape이 같은 경우에는 그냥 연산, shape이 다른 경우에 연산을 가능하게 해주는 것이 broadcasting

```python
# Operations between the same shapes
matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2., 2.]])
(matrix1 + matrix2).eval()
```

결과: array([[5., 5.]], dtype=float32)

shape이 다른 경우 격이 맞지 않지만 이것을 맞춰주는 것도 broadcasting의 역할

<img width="1000" alt="스크린샷 2020-12-11 오후 8 53 08" src="https://user-images.githubusercontent.com/62995632/101900520-f5fc2900-3bf2-11eb-95bf-fd2c85495c50.png">

## Reduce mean

> 평균을 줄여서 구하는 것

```python
tf.reduce_mean([1, 2], axis = 0).eval()
```

type이 integer이기 때문에 1.5가 아닌 1이 나온다.

결과: 1

```python
x = [[1., 2.],
     [3., 4.]]
     
tf.reduce_mean(x).eval()
```

결과: 2.5

```python
tf.reduce_mean(x, axis=0).eval()
```

axis=0으로 볼 때(세로로 확인) 1, 3의 평균 2가 되고, 2, 4의 평균으로 3이 된 것

결과: array([2., 3.], dtype=float32)

```python
tf.reduce_mean(x, axis=1).eval()
```

axis=1로 볼 때(가로로 확인) 1, 2의 평균 1.5가 되고, 3, 4의 평균으로 3.5가 된 것

결과: array([1.5, 3.5], dtype=float32)

```python
tf.reduce_mean(x, axis=-1).eval()
```

결과: array([1.5, 3.5], dtype=float32)

## Reduce sum

> 합을 구하는 것

```python
x = [[1., 2.],
     [3., 4.]]
     
tf.reduce_sum(x).eval()
```

결과: 10.0

```python
tf.reduce_sum(x, axis=0).eval()
```

결과: array([4., 6.], dtype=float32)

```python
tf.reduce_sum(x, axis=-1).eval()
```

결과: array([3., 7.], dtype=float32)

```python
tf.reduce_mean(tf.reduce_sum(x, axis=-1)).eval()
```

결과: 5.0

## Argmax

> maximum 값의 위치를 구하는 것

```python
x = [[0, 1, 2],
     [2, 1, 0]]
tf.argmax(x, axis=0).eval()
```

axis=0이기 때문에 세로로 확인을 하고, 0, 2 중에서 2가 더 크고, 1, 1 중에서 1이 더 크고, 2, 0 중에서 2가
더 크기 때문에 인덱스 번호인 1, 0, 0을 반환

결과: array([1, 0, 0])

```python
tf.argmax(x, axis=1).eval()
```

결과: array([2, 0])

```python
tf.argmax(x, axis=-1).eval()
```

결과: array([2, 0])


## Reshape

```python
t = np.array([[[0, 1, 2],
               [3, 4, 5]],
               
              [[6, 7, 8],
               [9, 10, 11]]])
t.shape
```

결과: (2, 2, 3)

```python
tf.reshape(t, shape=[-1, 3]).eval()
```

결과

array([[0, 1, 2],

[3, 4, 5],

[6, 7, 8],
            
[9, 10, 11]])

```python
tf.reshape(t, shape=[-1, 1, 3]).eval()
```

결과

array([[[0, 1, 2]],

[[3, 4, 5]],
            
[[6, 7, 8]],
            
[[9, 10, 11]]])
            
## Reshape(squeeze, expand)

```python
tf.squeeze([[0], [1], [2]]).eval()
```

값들을 쫙 펴준다

결과: array([0, 1, 2], dtype=int32)

```python
tf.expand_dims([0, 1, 2], 1).eval()
```

dimension을 추가하고 싶을 때 expand를 써서 얼마나 expand하고 싶은지 입력

결과

array([[0],

[1],
            
[2]], dtype=int32)
            
            
## One hot

> 하나만 hot하게 1로 바꾸는 것

```python
tf.one_hot([[0], [1], [2], [0]], depth=3).eval()
```
위에 적인 값들의 인덱스 위치를 1로 하고 나머지는 0

처음에 0이기 때문에 0번 인덱스에 1로, 1일때는 1번 인덱스에 1로..

자동으로 rank를 expand함

결과

array([[[1., 0., 0.]],

[[0., 1., 0.]],
            
[[1., 0., 0.]]], dtype=float32)
            
```python
t = tf.one_hot([[0], [1], [2], [0]], depth=3)
tf.reshape(t, shape=[-1, 3]).eval()
```

결과

array([[[1., 0., 0.]],

[[0., 1., 0.]],
            
[[0., 0., 1.]],
            
[[1., 0., 0.]]], dtype=float32)
          
          
## Casting

> 원하는 형으로 바꾸는 것(ex. float32 -> int32)

```python
tf.cast([1.8, 2.2, 3.3, 4.9], tf.int32).eval()
```

결과: array([1, 2, 3, 4], dtype=int32)

```python
tf.cast([True, False, 1==1, 0==1], tf.int32).eval()
```

결과: array([1, 0, 1, 0], dtype=int32)

## Stack
> 쌓는 것

```python
x = [1, 4]
y = [2, 5]
z = [3, 6]

# Pack along first dim.
tf.stack([x, y, z]).eval()
```

결과

array([[1, 4],

[2, 5],
            
[3, 6]], dtype=int32)
            
```python
tf.stack([x, y, z], axis=1).eval()
```

결과

array([[1, 2, 3],

[4, 5, 6]], dtype=int32)
            
## Ones and Zeros like

> 주어진 형태의 tense가 있다면 모양이 같지만 0 또는 1로 이루어져있게 만들고 싶을 때 사용

```python
x = [[0, 1, 2],
     [2, 1, 0]]
tf.ones_like(x).eval()   # 같은 shape에 1로 채워짐
```

결과

array([[1, 1, 1],

[1, 1, 1]], dtype=int32)
            
```python
tf.zeros_like(x).eval()   # 같은 shape에 0으로 채워짐
```

결과

array([[0, 0, 0],

[0, 0, 0]], dtype=int32)
            
## Zip

```python
for x, y in zip([1, 2, 3], [4, 5, 6]):
     print(x, y)
```

결과
1 4

2 5
     
3 6
     
```python
for x, y, z in zip([1, 2, 3], [4, 5, 6], [7, 8, 9]):
     print(x, y, z)
```

결과

1 4 7

2 5 8
     
3 6 9
