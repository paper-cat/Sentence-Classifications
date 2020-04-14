# Sentence-Classifications

[-한국어버전-](https://github.com/paper-cat/Sentence-Classifications/blob/master/KOR_README.md)

Implements some sentence(text) classification models by Tensorflow 2

Using tensorflow2.1


### Models
---
1. Convolutional Neural Networks for Sentence Classification by Yoon Kim - https://arxiv.org/pdf/1408.5882v2.pdf
    
2. Character-level Convolutional Networks for Text Classification - https://arxiv.org/pdf/1509.01626v3.pdf
 
... And More In Future

### How To Run Train
---
<pre><code>python train.py config-file</code></pre>

example.

<pre><code>python train.py nsmc_default.yaml</code></pre>


### How To Run Test (predict)
---
1. Run Bundle test for get accuracy
<pre><code>python predict.py config-file</code></pre>

2. Run A Sentence Prediction
<pre><code>python predict.py config-file TYPE A SENTENCE AFTER CONFIG FILE</code></pre>
<pre><code>python predict.py config-file 이 영화 꿀잼이네!</code></pre>



### DataSet
---
1. Naver sentiment movie corpus v1.0 - https://github.com/e9t/nsmc/
    - nsmc_default.yaml
    - nsmc_custom.yaml
  
2. Future Data Set:
    - IMDb 

 

### (Working On) Edit Config file to use your own dataset, hyper parameters
---
1. mode
- char : use character base vectorize
- token : (korean Only for now ) use Korean Morphological Analyzer

2. model
- cnn-basic : Model List #1 by Yoon kim
- char-cnn-basic : Model List #2 by Xiang Zhang...

3. dataset:
- nsmc : Naver Sentiment Movie Corpus by Lucy Park
---

You Can Use Your own datasets by edit config file as YAML
1. Edit train_file_path to train file
2. Choose train data type (txt, xlsx, ...)
3. Choose model type
4. Tune Hyper parameters
