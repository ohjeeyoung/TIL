**멋쟁이사자처럼 9기 운영진 교육을 바탕으로 작성되었습니다.**

## Web & Web Service

"우리는 웹서비스로부터 원하는 것을 얻는다."

- 서버(serve+er): 요청하는 서비스에 대해 응답하는 입장
- 클라이언트(client): 서버에게 원하는 대상, 원하는 정보를 요청하는 입장

- request(요청): 클라이언트가 서버에게 원하는 것을 요청하는 것
- response(응답): 서버가 클라이언트에게 응답하는 것

웹서비스를 만든다 = client를 만든다

Request의 종류
1. 갖다 줘 = GET
2. 처리해 줘 = POST


예시:

나는 네이버를 켜고 -> GET(네이버 갖다줘)

웹툰을 본 다음 -> GET(서버야 웹툰 갖다줘)

재미있다고 댓글을 달았다. -> POST(내 댓글 좀 처리해(달아)줘)

#### P2P(Peer-To-Peer)

> 모두가 서버이자 모두가 클라이언트

#### www(World Wide Web)

> client-server 관계, P2P 관계 등 컴퓨터들마다 얽히고 섥혀 하나의 큰 거미줄처럼 정보의 망이 형성 된 것

#### 웹 브라우저

> 서버랑 통신을 할 때 오가는 HTML 등의 역할을 수행하는 것이 웹 브라우저


## 웹 서버를 만드는 방법

#### 서버 컴퓨터
> 신경 써야 할 부분만 확실히 신경 쓴 컴퓨터

1. 빠른 컴퓨팅 능력
2. 24시간 켜 있어야
3. 발열 냉각장치
4. 클라이언트 수 고려
5. 보안 중요

#### 서버가 되기 위한 두가지 방법
1. 내 컴퓨터 = 서버 컴퓨터화시키기(로컬환경세팅)(Apache, IIS 등)
2. 이 세상 어딘가의 서버컴퓨터 빌리기(웹 호스팅 업체 대여)(AWS Cloud9, Github)

1) 로컬 환경 세팅
- 설치 다소 까다로움
- 추가적인 지식 요구
- 한번 익히면 자유로운 개발 가능

2) 웹 호스팅업체 이용
- 설치와 조작이 단순
- 과금발생
- 개발에 있어 제약 O
- 클라이언트 수를 고려하지 않아도 됨 -> 상황에 맞춰 대여하면 됨


## HTML(1)

HTML(Hyper Text Markup Language)

Hyper Text = Link
(Link에 따라 자유롭게 페이지를 이동할 수 있음 -> 정보를 받아들이는 수순을 다르게 만들어줌)

- 이해가 쉬움
- 정형화된 문법
- 쓰이는 문법만 맨날 쓰임

1. 글
2. 태그
3. 속성


## HTML(2)

- 대원칙! HTML로 꾸미려 들지 말자
- HTML은 애초에 '꾸미는 언어'가 아님 -> 꾸미는 언어는 CSS!


1. 문서의 일부를 설명해 주기 위한 HTML
2. 문서의 전체를 설명해 주기 위한 HTML

<HTML 코드>

1. "이거 HTML(로 작성된)문서야~"를 알려주는 태그

```html
<!DOCTYPE html>
<html>
```
  
2. 직접 화면에 등장하진 않지만 이 문서를 설명하는 태그(예. 이 문서를 한 마디로 설명하는 문서의 'Title', 인코딩 방식(utf-8) 등등...)
```html
<head>
```
3. 직접적으로 화면에 등장하는, 문서에서 보이는 태그(예. h1, p, li, ...)
```html
<body>
<h1>
<p>
<img>
```

form: 사용자로부터 입력값을 받아들이는 태그 -> <form action="전송받을 대상"> 전송받을 대상에게 내용을 전달해준다.

ol(ordered list) 태그 안에 li가 4개 생긴다
```html
ol>li*4
```
위의 코드를 치고 tab

## HTML(3)

#### 코드 

<img width="670" alt="스크린샷 2021-01-19 오후 11 19 07" src="https://user-images.githubusercontent.com/62995632/105046775-d9ce9000-5aac-11eb-9599-a30648d4ee62.png">
<img width="670" alt="스크린샷 2021-01-19 오후 11 19 29" src="https://user-images.githubusercontent.com/62995632/105046779-da672680-5aac-11eb-9037-fd842fbc9661.png">

#### 결과

<img width="1024" alt="나를 소개해요!" src="https://user-images.githubusercontent.com/62995632/105047077-3d58bd80-5aad-11eb-8fef-5c836a0dbad5.png">

## Bootstrap
> CSS/JavaScript 기반 웹 프레임워크

+ 웹 프레임워크: '웹을 만드는 재료들의 모음' 정도로 생각

- 공짜!
- 반응형 웹 지원(자동화면조정)
- 브라우저 호환성

- 딱 보면 티 남
- 성능이 다소 떨어짐

header에 부트스트랩 CDN 넣어줄 것

jquery CDN도 따로 넣어줄 것 -> https://code.jquery.com/

#### 콘테이너

> 여백을 만들어 주는 것

#### 실습코드

```html
<!DOCTYPE html>
<html lang="ko">
<head>
    <script
    src="https://code.jquery.com/jquery-3.5.1.js"
    integrity="sha256-QWo7LDvxbWT2tbbQ97B53yJnYU3WhH/C8ycbRAkjPDc="
    crossorigin="anonymous"></script>
    <!-- 합쳐지고 최소화된 최신 CSS -->
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.2/css/bootstrap.min.css">

    <!-- 부가적인 테마 -->
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.2/css/bootstrap-theme.min.css">

    <!-- 합쳐지고 최소화된 최신 자바스크립트 -->
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.2/js/bootstrap.min.js"></script>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>나를 소개해요!</title>
</head>
<body>
    <div class="container">
        <br>
        <ul class="nav nav-tabs">
            <li role="presentation" class="active"><a href="1.html">유년기</a></li>
            <li role="presentation"><a href="2.html">질풍노도의 시기</a></li>
            <li role="presentation"><a href="3.html">방황기</a></li>
            <li role="presentation"><a href="4.html">지금</a></li>
          </ul>
    <h1>오지영을 소개합니다</h1>
    <h2>오지영의 소개해요</h2><br>

    <form action="전송받을 대상">
        아이디: <input type="text" name="id">
        비밀번호: <input type="password" name="pw">
        <div class="form-group">
          <label for="exampleInputEmail1">이메일 주소</label>
          <input type="email" class="form-control" id="exampleInputEmail1" placeholder="이메일을 입력하세요">
        </div>
        <div class="form-group">
          <label for="exampleInputPassword1">암호</label>
          <input type="password" class="form-control" id="exampleInputPassword1" placeholder="암호">
        </div>
        <div class="form-group">
          <label for="exampleInputFile">파일 업로드</label>
          <input type="file" id="exampleInputFile">
          <p class="help-block">여기에 블록레벨 도움말 예제</p>
        </div>
        <div class="checkbox">
          <label>
            <input type="checkbox"> 입력을 기억합니다
          </label>
        </div>
        <input type="submit" class="btn btn-primary"></button>
      </form>
    <br>
    <img src="IMG.jpg" width=300>
    <br>

    <form action="전송받을 대상">
        <h2>나의 일기장</h2>
        제목: <input type="text" name="diarytitle"><br>
        <select>
            <option value="goodday">좋은날</option>
            <option value="sadday">슬픈날</option>
            <option value="soso">그저그런날</option>
        </select> <br>
        내용: <br>
        <textarea cols="30" rows="20"></textarea>
        <br>
        <input type="submit">
    </form>
    <div class="progress">
        <div class="progress-bar progress-bar-success progress-bar-striped" role="progressbar" aria-valuenow="40" aria-valuemin="0" aria-valuemax="100" style="width: 40%">
          <span class="sr-only">40% Complete (success)</span>
        </div>
      </div>
      <div class="progress">
        <div class="progress-bar progress-bar-info progress-bar-striped" role="progressbar" aria-valuenow="20" aria-valuemin="0" aria-valuemax="100" style="width: 20%">
          <span class="sr-only">20% Complete</span>
        </div>
      </div>
      <div class="progress">
        <div class="progress-bar progress-bar-warning progress-bar-striped" role="progressbar" aria-valuenow="60" aria-valuemin="0" aria-valuemax="100" style="width: 60%">
          <span class="sr-only">60% Complete (warning)</span>
        </div>
      </div>
      <div class="progress">
        <div class="progress-bar progress-bar-danger progress-bar-striped" role="progressbar" aria-valuenow="80" aria-valuemin="0" aria-valuemax="100" style="width: 80%">
          <span class="sr-only">80% Complete (danger)</span>
        </div>
      </div>
</div>
</body>
</html>
```

#### 결과

<img width="1024" alt="나를 소개해요!" src="https://user-images.githubusercontent.com/62995632/105630989-58517600-5e8f-11eb-9362-48ab705507d7.png">
