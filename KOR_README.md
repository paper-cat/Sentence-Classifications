# 문장 분류
여러 문장 분류 모델들을 Tensorflow2 로 구현

###모델

---
1. Convolutional Neural Networks for Sentence Classification by Yoon Kim - https://arxiv.org/pdf/1408.5882v2.pdf

2. (Working On) Character-level Convolutional Networks for Text Classification - https://arxiv.org/pdf/1509.01626v3.pdf

계속 추가중입니다.

---
###학습 실행 방법

템플릿:
<pre><code>python train.py config-file</code></pre>

<pre><code>python train.py nsmc_default.yaml</code></pre>

### 예측(테스트) 실행
---
1. 파일 전체 예측 (accuracy)
<pre><code>python predict.py config-file</code></pre>

2. 한 문장 예측
<pre><code>python predict.py nsmc_default.yaml 테스트할 문장 한개를 이후에 작성</code></pre>
<pre><code>python predict.py nsmc_default.yaml 이 영화 재밌다!</code></pre>


### 데이터셋
---
1. 네이버 영화 댓글 감정 분석 v1.0 - https://github.com/e9t/nsmc/
    - nsmc_default.yaml
    - nsmc_custom.yaml
  
2. 이후에 추가될 데이터셋:
    - IMDb 

 

### (구현중) Config 파일을 작성하여 다른 데이터셋과 하이퍼 파라미터를 사용할 수 있습니다.
---
1. mode
- char : 글자 단위로 vectorize 합니다
- token : 한국어 형태소 분석기를 사용합니다 (konlpy 의 otk 사용)
2. model
- cnn-basic : 1번 모델
- char-cnn-basic : 2번 모델
3. dataset: (데이터 타입)
- nsmc : 네이버 영화 감정 분석 데이터 입니다.

---
현재 작성되어 있는 yaml 파일을 수정하여 사용 가능
1. train_file_path, 를 원하는 데이터 파일로 설정
2. mode, model, dataset 설정
3. 하이퍼 파라미터 튜닝
