# Sentence-Classifications
![HitCount](http://hits.dwyl.com/{readme_en}.svg)
[-한국어버전-](https://github.com/paper-cat/Sentence-Classifications/blob/master/KOR_README.md)

Implements some sentence(text) classification models by Tensorflow 2

Using tensorflow2.1


## Models

---
1. Convolutional Neural Networks for Sentence Classification by Yoon Kim - https://arxiv.org/pdf/1408.5882v2.pdf
    
2. Character-level Convolutional Networks for Text Classification - https://arxiv.org/pdf/1509.01626v3.pdf
 
... And More In Future

## How To Run Train

---
template:
<pre><code>python train.py config-file</code></pre>

example:

<pre><code>python train.py imdb_default.yaml</code></pre>


## How To Run Test (predict)

---
1. Run Bundle test for get accuracy
<pre><code>python predict.py config-file</code></pre>

2. Run A Sentence Prediction
<pre><code>python predict.py config-file TYPE A SENTENCE AFTER CONFIG FILE</code></pre>
example:
<pre><code>python predict.py imdb_token I love this movie!</code></pre>



## DataSet
---
1. Naver sentiment movie corpus v1.0 - https://github.com/e9t/nsmc/
    - nsmc_default.yaml
    - nsmc_custom.yaml
    - nsmc_tokens.yaml
  
2. IMDB dataset
    - imdb_default.yaml
    - imdb_token.yaml
    

## Edit or Add Config file to use your own dataset, hyper parameters
---
1. mode
- char : use character base vectorize
- token : use tokenizer Analyzer

2. model
- cnn-basic : Model List #1 by Yoon kim
- char-cnn-basic : Model List #2 by Xiang Zhang...

3. dataset:
- nsmc : Naver Sentiment Movie Corpus by Lucy Park
- imdb : Famous IMDB dataset

4. parameters in default parameter.
---

(working on...) You Can Use Your own datasets by edit config file as YAML
1. Edit train_file_path to train file
2. Choose model, mode, dataset type
3. Tune Hyper parameters
