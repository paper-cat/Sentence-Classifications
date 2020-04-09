# Sentence-Classifications

[-한국어버전-](https://github.com/paper-cat/Sentence-Classifications/blob/master/KOR_README.md)

Implements some sentence(text) classification models by Tensorflow 2


### Models
---
1. Convolutional Neural Networks for Sentence Classification by Yoon Kim - https://arxiv.org/pdf/1408.5882v2.pdf
    - Original model use word embedding, going to be implemented
    - Currently, Using this model as character base.
2. (Working On) Character-level Convolutional Networks for Text Classification - https://arxiv.org/pdf/1509.01626v3.pdf
    - Working On
 

### How To Run
---
<pre><code>python train.py config-file</code></pre>

example.

<pre><code>python train.py nsmc_default.yaml</code></pre>


### DataSet
---
1. Naver sentiment movie corpus v1.0 - https://github.com/e9t/nsmc/
    - nsmc_default.yaml
    - nsmc_custom.yaml
  
2. Future Data Set:
    - IMDb 

 

### (Not yet implemented) Edit Config file to use your own dataset, hyper parameters
---
You Can Use Your own datasets by edit config file as YAML
1. Edit train_file_path to train file
2. Choose train data type (txt, xlsx, ...)
3. Choose model type
4. Tune Hyper parameters
