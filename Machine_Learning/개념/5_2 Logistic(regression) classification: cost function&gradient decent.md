#### Cost

![CodeCogsEqn](https://user-images.githubusercontent.com/62995632/93712521-5bf78800-fb91-11ea-8897-2533e31787c4.gif)

when 

![CodeCogsEqn (5)](https://user-images.githubusercontent.com/62995632/93719093-662e7c00-fbbb-11ea-9805-6a5063589840.gif)

<img width="300" alt="스크린샷 2020-12-08 오후 8 51 03" src="https://user-images.githubusercontent.com/62995632/101480489-1c258d00-3997-11eb-8d01-8b627601d00e.png">


#### Cost function

기존 Linear Regression과 Sigmoid function을 적용했을 때의 모습

- 기울기가 평평해 지는 지점에서 멈춤. 그러나 sigmoid에서는 기존 cost function을 적용하면 울퉁 불퉁하므로 시작점에 따라서 종료점(최소) 구간이 틀려질 수 있다. 즉 training을 멈추게 된다. 

<img width="1438" alt="스크린샷 2020-12-08 오후 9 50 17" src="https://user-images.githubusercontent.com/62995632/101486051-6b6fbb80-399f-11eb-8004-49c00300e069.png">

## New cost function for logistic

H(x)를 바꿨기 때문에 cost 함수도 바꿔야한다.

![daum_equation_1607431904071](https://user-images.githubusercontent.com/62995632/101486393-ecc74e00-399f-11eb-8585-71bcecab5f64.png)

![daum_equation_1607432045740](https://user-images.githubusercontent.com/62995632/101486399-ee911180-399f-11eb-9926-e567b9fc6a6e.png)

<img width="1439" alt="스크린샷 2020-12-08 오후 10 06 20" src="https://user-images.githubusercontent.com/62995632/101487569-b8ed2800-39a1-11eb-91a9-c4fe39f088b9.png">
- y:1 일때 예측이 틀릴경우 즉 H(x) = 0이면 cost 는 무한대가 된다. 

- y:0 일때 H(x) =0 이면 cost는 0이되고, H(x) = 1 이면 cost는 무한대가 된다. 

즉 잘못 예측되면 cost가 무한대로 커진다. 

## Cost function

![daum_equation_1607431904071](https://user-images.githubusercontent.com/62995632/101486393-ecc74e00-399f-11eb-8585-71bcecab5f64.png)

![daum_equation_1607432045740](https://user-images.githubusercontent.com/62995632/101486399-ee911180-399f-11eb-9926-e567b9fc6a6e.png)

![daum_equation_1607432983554](https://user-images.githubusercontent.com/62995632/101487863-1b462880-39a2-11eb-8b0e-02a0eb3f7bed.png)

## Minimize cost - Gradient decent algorithm

![daum_equation_1607441596611](https://user-images.githubusercontent.com/62995632/101504083-2905a900-39b6-11eb-84e8-8613e1bae60b.png)

![CodeCogsEqn (3)](https://user-images.githubusercontent.com/62995632/93718981-9de8f400-fbba-11ea-8c89-c2af5eaa2420.gif)

```python
# cost function
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis) + (1-Y)*tf.log(1-hypothesis)))

# Minimize
a = tf.Variable(0.1)  # Learning rate, alpha
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)
```
