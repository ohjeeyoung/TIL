# RNN with time series data(stock)

## Time series data

> 시간에 따라서 값이 변하는 데이터

ex. 주식

#### Many to one

<img width="1000" alt="스크린샷 2020-12-14 오전 4 05 24" src="https://user-images.githubusercontent.com/62995632/102021222-9b64f780-3dc1-11eb-86fd-fcc4c3589fa8.png">

7일 동안의 data를 가지고 있다면 8일째의 값이 궁금한 것

이전 것들이 다음 값에 영향을 미친다는 것이 time series의 기본 전제

<img width="1000" alt="스크린샷 2020-12-14 오전 4 07 45" src="https://user-images.githubusercontent.com/62995632/102021256-f0a10900-3dc1-11eb-9a5e-9596d628f4b8.png">

input_dim = 5(open, high, low, volume, close), seq_len = 7(7일동안의 데이터), output_dim = 1(8일째 알고싶은 값)

#### Reading data

<img width="1000" alt="스크린샷 2020-12-14 오전 4 09 14" src="https://user-images.githubusercontent.com/62995632/102021284-247c2e80-3dc2-11eb-94d6-916bcf4d62b7.png">

<img width="1000" alt="스크린샷 2020-12-14 오전 4 13 35" src="https://user-images.githubusercontent.com/62995632/102021383-c0a63580-3dc2-11eb-9b76-ea917d1cf45a.png">

```python
timesteps = seq_length = 7
data_dim = 5
output_dim = 1

# Open, High, Low, Close, Volume
xy = np.loadtxt('data-02-stock-daily.csv', delimiter=',')
xy = xy[::-1]   # reverse order (chronically ordered)
xy = MinMaxScaler(xy)
x = xy
y = xy[:, [-1]] # Close as label

dataX = []
dataY = []
for i in range(0, len(y) - seq_length):
    _x = x[i: i + seq_length]
    _y = y[i + seq_length]  # Next close price
    print(_x, "->", y)
    dataX.append(_x)
    dataY.append(_y)
```

#### Training and test datasets

