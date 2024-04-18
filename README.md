# Readme

本项目是针对于Kaggle中的烂番茄情感评论所设计的深度学习模型与框架，同时运用各种模型增强的方式和手段对于模型效果进行更好地提升。该文件是为了您更好地运行本项目，具体运行方法请见下。

### 下载库文件

下载本项目用到的python库，如有不全，请自行下载。

```python
pip install -r requirements.txt
```

### 获取数据

本部分是为了获取[GloVe: Global Vectors for Word Representation (stanford.edu)](https://nlp.stanford.edu/projects/glove/)中的 **pre-trained word vectors**，通过加载glove为文本向量化做准备。

```
python fetch_data.py
```

### 数据准备

本部分是为了处理数据，为了以后的训练模型和测试模型做准备。

```python
cd sentiment-analysis-on-movie-reviews/scripts
python preprocessing.py
```

### 训练模型

为了更加方便的进行训练模型的步骤，加载好的数据集已经放入data中，可以不进行之前步骤，直接进行模型的训练。

本部分是进行模型的训练，模型训练的参数如下：

- checkpoint：加载预训练模型的相对路径
- pretrain：预训练模型是否为MLM任务后得到的预训练模型
- model：选择加载的模型
- use_pgd：是否进行对抗训练

```
cd training
```

1. 基于CNN的文本分类模型TextCNN

   ```
   python train.py --model testcnn
   ```

2. 基于RNN的BiLSTM模型

   ```
   python train.py --model bilstm
   ```

3. BiGRU和注意力机制结合的深度模型

   ```
   python train.py --model normal
   ```

4. 基于Bert的大规模预训练模型Sibert

   ```
   python process_SiBert.py
   ```

5. 进行进一步预训练（MLM任务）

   ```
   python pretrain.py
   ```

### 测试模型

本部分是对于模型进行进一步的测试以获得submission.csv。

```
cd ../testing
```

1. 基于CNN的文本分类模型TextCNN

   ```
   python test.py --model testcnn
   ```

2. 基于RNN的BiLSTM模型

   ```
   python test.py --model bilstm
   ```

3. BiGRU和注意力机制结合的深度模型

   ```
   python test.py --model normal
   ```

### 说明

由于文件大小的限制，没有保存训练好的模型，只保存了模型的具体结果，请见submission。
