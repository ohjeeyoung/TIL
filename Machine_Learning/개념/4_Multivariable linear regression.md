## Recap

Linear regression을 설계하기 위해 필요한 3가지
1. Hypothesis(가설): H(x) = Wx + b
2. Cost/Loss funcion
3. Gradient descent algorithm(경사하강 알고리즘)
---------------------------

### Predicting exam score
: regression using one input(x) vs regression using three inputs(x1, x2, x3)

### Hypothesis

- 변수 1개일 때 :
![CodeCogsEqn (5)](https://user-images.githubusercontent.com/62995632/93719366-151f8780-fbbd-11ea-9716-1a6b00b535b6.gif)

- 변수 3개일 때 :
![daum_equation_1606933600168](https://user-images.githubusercontent.com/62995632/100915441-b3ae5a00-3517-11eb-9c1c-c5df42629f42.png)

### Cost function

- 변수 3개일 때 :
![daum_equation_1606934008886](https://user-images.githubusercontent.com/62995632/100915896-57980580-3518-11eb-883f-834af3e673f8.png)

## Multi-variable

변수가 많아질수록 수식이 점점 길어짐

-> Matrix를 이용

---------------------------
## Matrix multiplication

### Hypothesis using matrix

![daum_equation_1606934234930](https://user-images.githubusercontent.com/62995632/100916253-db51f200-3518-11eb-9f44-a8d8eea3f520.png)

![daum_equation_1606934465492](https://user-images.githubusercontent.com/62995632/100916675-67fcb000-3519-11eb-93bc-b895d95e2deb.png)

-> H(X) = XW

instance가 많을 때도 각각을 계산하는 것이 아닌 한번에 H(X) = XW 형태로 행렬의 곱셈으로 계산 가능

X -> [5,3] = [instance, variable]
W -> [3,1]
H(X) -> [5,1] = [instance, Y]

- 행렬 곱셈의 장점: multi variable 일때도 instance가 많을 때도 n으로 쉽게 처리 가능

### WX vs XW
- Lecture(theory): H(x) = Wx + b
- Implementation(TensorFlow): H(X) = XW
