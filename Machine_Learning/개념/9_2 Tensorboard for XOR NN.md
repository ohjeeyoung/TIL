## Tensorboard for XOR NN

#### TensorBoard: TF logging/debugging tool
- Visualize your TF graph
- Plot quantitative metrics
- Show additional data

#### Old fashion: print, print, print

그동안은 열심히 출력을 통해 값의 변화를 확인했음

## 5 steps of using TensorBoard

<img width="1000" alt="스크린샷 2020-12-12 오전 4 09 00" src="https://user-images.githubusercontent.com/62995632/101944270-c881a080-3c2f-11eb-97d6-c63fed10a5b4.png">

1. 어떤 것을 logging 할 것인지 정함
2. 한번에 쓰기 위해 merge
3. session에 들어가서 summary 어느 위치에 저장할지 file의 위치를 정함
4. sess.run에 summary를 넣어서 실행, file에 기록함
5. 터미널에서 tensorboard를 실행

1. Step1
#### Scalar tensors

값에 따라 정해짐

<img width="1000" alt="스크린샷 2020-12-12 오전 4 11 51" src="https://user-images.githubusercontent.com/62995632/101944511-2dd59180-3c30-11eb-8db5-0882e396574a.png">

#### Histogram(multi-dimensional tensors)

값이 벡터거나 그러면 histogram 사용

<img width="1000" alt="스크린샷 2020-12-12 오전 4 13 25" src="https://user-images.githubusercontent.com/62995632/101944661-6b3a1f00-3c30-11eb-9d52-5b11fb3f7619.png">

#### Add scope for better graph hierarchy

그래프를 보고 싶을 때

그래프를 한번에 펼쳐놓으면 보기 어렵기 때문에 name_scope를 이용해 계층별로 정리

layer를 보기 쉽게 나눠준다

<img width="1000" alt="스크린샷 2020-12-12 오전 4 14 50" src="https://user-images.githubusercontent.com/62995632/101944768-991f6380-3c30-11eb-9293-8d5f8c496dda.png">

2. Step2, 3
#### Merge summaries and create writer after creating session

```python
# Summary
summary = tf.summary.merge_all()

# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Create summary writer
writer = tf.summary.FileWriter(TB_SUMMARY_DIR)
writer.add_graph(sess.graph)  # Add graph in the tensorboard
```

3. Step4
#### Run merged summary and write (add summary)

```python
s, _ = sess.run([summary, optimizer], feed_dict=feed_dict)
writer.add_summary(s, global_step=global_step)
global_step += 1
```

4. Step5
#### Launch tensorboard(local)

```python
writer = tf.summary.FileWriter(".logs/xor_logs")
```

$ tensorboard -logdir=./logs/xor_logs

Starting TensorBoard b'41' on port 6006
(You can navigate to http://127.0.0.1:6006)

#### Launch tensorboard(remote server)

로컬에서 안돌리고 리모트에서 돌린다면 로컬번호를 조정해서 리모트에 있는 tensorboard를 바로 볼 수 있음

<img width="1000" alt="스크린샷 2020-12-12 오전 4 22 15" src="https://user-images.githubusercontent.com/62995632/101945501-aa1ca480-3c31-11eb-9e50-7f4c25faaa83.png">

#### Multiple runs

값을 비교해보고 싶을 때

ex. learning_rate=0.1 VS learning_rate=0.01

<img width="1000" alt="스크린샷 2020-12-12 오전 4 24 55" src="https://user-images.githubusercontent.com/62995632/101946180-0089e300-3c32-11eb-82b1-734db31cf4b8.png">

학습을 할 때 write 하는 부분에서 디렉토리를 다르게 준다.

같은 logs 안에 하위 디렉토리를 여러개 만들고 logs를 실행시키면 여러개의 그래프를 동시에 볼 수 있음

#### Exercise
- Wide and Deep NN for MNIST
- Add tensorboard
