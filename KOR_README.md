# 문장 분류
여러 문장 분류 모델들을 Tensorflow2 로 구현

### 모델
---
1. Convolutional Neural Networks for Sentence Classification by Yoon Kim - https://arxiv.org/pdf/1408.5882v2.pdf
    - Original model use word embedding, going to be implemented
    - Currently, Using this model as character base.
2. (Working On) Character-level Convolutional Networks for Text Classification - https://arxiv.org/pdf/1509.01626v3.pdf
    - Working On
 

### 실행 방법
---
<pre><code>python train.py config-file</code></pre>

예시.

<pre><code>python train.py nsmc_default.yaml</code></pre>


### 데이터셋
---
1. 네이버 영화 댓글 감정 분석 v1.0 - https://github.com/e9t/nsmc/
    - nsmc_default.yaml
    - nsmc_custom.yaml
  
2. 이후에 추가될 데이터셋:
    - IMDb 

 

### (구현중) Config 파일을 작성하여 다른 데이터셋과 하이퍼 파라미터를 사용할 수 있습니다.
---
현재 작성되어 있는 yaml 파일을 수정하여 사용 가능
1. train_file_path 를 원하는 데이터 파일로 설정
2. train data type (txt, xlsx, ...) 을 설정
3. model type 설정
4. 하이퍼 파라미터 튜닝
