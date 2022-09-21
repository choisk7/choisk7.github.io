---
layout: single
title:  "트랜스포머를 이용한 한국어 챗봇(Transformer Chatbot Tutorial)"
categories: DL
tag: [transformer, chatbot]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: false
---
**[공지사항]** ["출처: https://wikidocs.net/89786"](https://wikidocs.net/89786)
{: .notice--danger}


# 트랜스포머를 이용한 한국어 챗봇(Transformer Chatbot Tutorial)

([여기 설명을 먼저 읽고오세요](https://choisk7.github.io/dl/%ED%8A%B8%EB%9E%9C%EC%8A%A4%ED%8F%AC%EB%A8%B8(Transformer)/))

## 1. 데이터 로드하기


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import urllib.request
import time
import tensorflow_datasets as tfds
import tensorflow as tf
```

이번 챕터에서 사용할 챗봇 데이터를 로드하여 상위 5개의 샘플을 출력해봅시다.


```python
urllib.request.urlretrieve("https://raw.githubusercontent.com/deepseasw/seq2seq_chatbot/master/dataset/chatbot/ChatbotData.csv", filename="ChatBotData.csv")
train_data = pd.read_csv('ChatBotData.csv')
train_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Q</th>
      <th>A</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12시 땡!</td>
      <td>하루가 또 가네요.</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1지망 학교 떨어졌어</td>
      <td>위로해 드립니다.</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3박4일 놀러가고 싶다</td>
      <td>여행은 언제나 좋죠.</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3박4일 정도 놀러가고 싶다</td>
      <td>여행은 언제나 좋죠.</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PPL 심하네</td>
      <td>눈살이 찌푸려지죠.</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



이 데이터는 질문(Q)과 대답(A)의 쌍으로 이루어진 데이터입니다.


```python
print('챗봇 샘플의 개수 :', len(train_data))
```

    챗봇 샘플의 개수 : 11823
    

총 샘플의 개수는 11,823개입니다. 불필요한 Null 값이 있는지 확인해봅시다.


```python
print(train_data.isnull().sum())
```

    Q        0
    A        0
    label    0
    dtype: int64
    

Null 값은 별도로 존재하지 않습니다. 이번 챕터에서는 토큰화를 위해 형태소 분석기를 사용하지 않고, 다른 방법인 학습 기반의 토크나이저를 사용할 것입니다. 그래서 원 데이터에서 ?, ., !와 같은 구두점을 미리 처리해두어야 하는데, 구두점들을 단순히 제거할 수도 있겠지만, 여기서는 구두점 앞에 공백. 즉, 띄어쓰기를 추가하여 다른 문자들과 구분하겠습니다.

가령, '12시 땡!' 이라는 문장이 있다면 '12시 땡 !'으로 땡과 !사이에 공백을 추가합니다. 이는 정규 표현식을 사용하여 가능합니다. 이 전처리는 질문 데이터와 답변 데이터 모두에 적용해줍니다.


```python
questions = []
for sentence in train_data["Q"]:
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = sentence.strip()
    questions.append(sentence)
    
answers = []
for sentence in train_data["A"]:
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = sentence.strip()
    answers.append(sentence)
```

질문과 대답에 대해서 상위 5개만 출력하여 구두점들이 띄어쓰기를 통해 분리되었는지 확인해봅시다.


```python
print(questions[:5])
print(answers[:5])
```

    ['12시 땡 !', '1지망 학교 떨어졌어', '3박4일 놀러가고 싶다', '3박4일 정도 놀러가고 싶다', 'PPL 심하네']
    ['하루가 또 가네요 .', '위로해 드립니다 .', '여행은 언제나 좋죠 .', '여행은 언제나 좋죠 .', '눈살이 찌푸려지죠 .']
    

## 2. 단어 집합 생성
앞서 14챕터 서브워드 토크나이저 챕터에서 배웠던 서브워드텍스트인코더를 사용해봅시다. 자주 사용되는 서브워드 단위로 토큰을 분리하는 토크나이저로 학습 데이터로부터 학습하여 서브워드로 구성된 단어 집합을 생성합니다.


```python
tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    questions + answers, target_vocab_size=2**13
)
```

단어 집합이 생성되었습니다. 그런데 seq2seq 챕터에서 배웠던 것처럼 인코더-디코더 모델 계열에는 디코더의 입력으로 사용할 시작을 의미하는 시작 토큰 SOS와 종료 토큰 EOS 또한 존재합니다. 해당 토큰들도 단어 집합에 포함시킬 필요가 있으므로 이 두 토큰에 정수를 부여해줍니다.


```python
START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]

VOCAB_SIZE = tokenizer.vocab_size + 2
```

시작 토큰과 종료 토큰을 추가해주었으나 단어 집합의 크기도 +2를 해줍니다.
시작 토큰의 번호와 종료 토큰의 번호, 그리고 단어 집합의 크기를 출력해봅시다.


```python
print('시작 토큰 번호 :',START_TOKEN)
print('종료 토큰 번호 :',END_TOKEN)
print('단어 집합의 크기 :',VOCAB_SIZE)
```

    시작 토큰 번호 : [8178]
    종료 토큰 번호 : [8179]
    단어 집합의 크기 : 8180
    

패딩에 사용될 0번 토큰부터 마지막 토큰인 8,179번 토큰까지의 개수를 카운트하면 단어 집합의 크기는 8,180개입니다.

## 3. 정수 인코딩과 패딩
단어 집합을 생성한 후에는 서브워드텍스트인코더의 토크나이저로 정수 인코딩을 진행할 수 있습니다. 이는 토크나이저의 .encode()를 사용하여 가능합니다. 우선 임의로 선택한 20번 질문 샘플. 즉, questions[20]을 가지고 정수 인코딩을 진행해봅시다.


```python
print('임의의 질문 샘플을 정수 인코딩 : {}'.format(tokenizer.encode(questions[20])))
```

    임의의 질문 샘플을 정수 인코딩 : [5766, 611, 3509, 141, 685, 3747, 849]
    

임의의 질문 문장이 정수 시퀀스로 변환되었습니다. 반대로 정수 인코딩 된 결과는 다시 decode()를 사용하여 기존의 텍스트 시퀀스로 복원할 수 있습니다. 20번 질문 샘플을 가지고 정수 인코딩하고, 다시 이를 디코딩하는 과정은 다음과 같습니다.


```python
sample_string = questions[20]

tokenized_string = tokenizer.encode(sample_string)
print ('정수 인코딩 후의 문장 {}'.format(tokenized_string))

original_string = tokenizer.decode(tokenized_string)
print ('기존 문장: {}'.format(original_string))
```

    정수 인코딩 후의 문장 [5766, 611, 3509, 141, 685, 3747, 849]
    기존 문장: 가스비 비싼데 감기 걸리겠어
    

정수 인코딩 된 문장을 .decode()을 하면 자동으로 서브워드들까지 다시 붙여서 기존 단어로 복원해줍니다. 가령, 정수 인코딩 문장을 보면 정수가 7개인데 기존 문장의 띄어쓰기 단위인 어절은 4개밖에 존재하지 않습니다. 이는 '가스비'나 '비싼데'라는 한 어절이 정수 인코딩 후에는 두 개 이상의 정수일 수 있다는 겁니다. 각 정수가 어떤 서브워드로 맵핑되는지 출력해봅시다.


```python
for ts in tokenized_string:
    print ('{} ----> {}'.format(ts, tokenizer.decode([ts])))
```

    5766 ----> 가스
    611 ----> 비 
    3509 ----> 비싼
    141 ----> 데 
    685 ----> 감기 
    3747 ----> 걸리
    849 ----> 겠어
    

샘플 1개를 가지고 정수 인코딩과 디코딩을 수행해보았습니다. 이번에는 전체 데이터에 대해서 정수 인코딩과 패딩을 진행합니다. 이를 위한 함수로 tokenize_and_filter()를 만듭니다. 여기서는 임의로 패딩의 길이는 40으로 정했습니다.


```python
MAX_LENGTH = 40

def tokenize_and_filter(inputs, outputs):
    tokenized_inputs, tokenized_outputs = [], []
    
    for (sentence1, sentence2) in zip(inputs, outputs):
        sentence1 = START_TOKEN + tokenizer.encode(sentence1) + END_TOKEN
        sentence2 = START_TOKEN + tokenizer.encode(sentence2) + END_TOKEN
        
        tokenized_inputs.append(sentence1)
        tokenized_outputs.append(sentence2)
        
    tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_inputs, maxlen=MAX_LENGTH, padding="post"
    )
    tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_outputs, maxlen=MAX_LENGTH, padding="post"
    )
    
    return tokenized_inputs, tokenized_outputs
```


```python
questions, answers = tokenize_and_filter(questions, answers)
```

정수 인코딩과 패딩이 진행된 후의 데이터의 크기를 확인해봅시다.


```python
print('질문 데이터의 크기(shape) :', questions.shape)
print('답변 데이터의 크기(shape) :', answers.shape)
```

    질문 데이터의 크기(shape) : (11823, 40)
    답변 데이터의 크기(shape) : (11823, 40)
    

질문과 답변 데이터의 모든 문장이 모두 길이 40으로 변환되었습니다. 임의로 0번 샘플을 출력해봅시다.


```python
print(questions[0])
print(answers[0])
```

    [8178 7915 4207 3060   41 8179    0    0    0    0    0    0    0    0
        0    0    0    0    0    0    0    0    0    0    0    0    0    0
        0    0    0    0    0    0    0    0    0    0    0    0]
    [8178 3844   74 7894    1 8179    0    0    0    0    0    0    0    0
        0    0    0    0    0    0    0    0    0    0    0    0    0    0
        0    0    0    0    0    0    0    0    0    0    0    0]
    

길이 40을 맞추기 위해 뒤에 0이 패딩된 것을 확인할 수 있습니다.

## 4. 인코더와 디코더의 입력, 그리고 레이블 만들기.
tf.data.Dataset을 사용하여 데이터를 배치 단위로 불러올 수 있습니다.


```python
BATCH_SIZE = 64
BUFFER_SIZE = 20000

# 디코더의 실제값 시퀀스에서는 시작 토큰을 제거해야 한다.
dataset = tf.data.Dataset.from_tensor_slices((
    {
        "inputs": questions,
        "dec_inputs": answers[:, :-1]
    },
    {
        "outputs": answers[:, 1:]
    }
))

dataset = dataset.cache() # 데이터셋을 캐시, 즉 메모리 또는 파일에 보관합니다. 따라서 두번째 이터레이션부터는 캐시된 데이터를 사용합니다.
dataset = dataset.shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE) # 데이터 추출하는데 걸리는  시간을 줄여줌
```


```python
# 임의의 샘플에 대해서 [:, :-1]과 [:, 1:]이 어떤 의미를 가지는지 테스트해본다.
print(answers[0]) # 기존 샘플
print(answers[:1][:, :-1]) # 마지막 패딩 토큰 제거하면서 길이가 39가 된다.
print(answers[:1][:, 1:]) # 맨 처음 토큰이 제거된다. 다시 말해 시작 토큰이 제거된다. 길이는 역시 39가 된다.
```

    [8178 3844   74 7894    1 8179    0    0    0    0    0    0    0    0
        0    0    0    0    0    0    0    0    0    0    0    0    0    0
        0    0    0    0    0    0    0    0    0    0    0    0]
    [[8178 3844   74 7894    1 8179    0    0    0    0    0    0    0    0
         0    0    0    0    0    0    0    0    0    0    0    0    0    0
         0    0    0    0    0    0    0    0    0    0    0]]
    [[3844   74 7894    1 8179    0    0    0    0    0    0    0    0    0
         0    0    0    0    0    0    0    0    0    0    0    0    0    0
         0    0    0    0    0    0    0    0    0    0    0]]
    

## 5. 트랜스포머 만들기
이제 트랜스포머를 만들어봅시다. 하이퍼파라미터를 조정하여 실제 논문의 트랜스포머보다는 작은 모델을 만듭니다.
여기서 선택한 주요 하이퍼파라미터의 값은 다음과 같습니다.

$d_{model} = 256$ \
$\text{num_layers} = 2$ \
$\text{num_heads} = 8$ \
$d_{ff} = 512$ 


```python
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model):
        super().__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)
        
    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles
    
    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis], # (position, 1)
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :], # (1, d_model)
            d_model=d_model)
        # angle_rads == (pos, d_model)
            
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])
        
        angle_rads = np.zeros(angle_rads.shape)
        angle_rads[:, 0::2] = sines
        angle_rads[:, 1::2] = cosines
        pos_encoding = tf.constant(angle_rads)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        print(pos_encoding.shape) # (1, position, 128)
        return tf.cast(pos_encoding, tf.float32)
    
    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :] # (1, None, 128)
```


```python
def scaled_dot_product_attention(query, key, value, mask):
    # query 크기 : (batch_size, num_heads, query의 문장 길이, d_model/num_heads)
    # key 크기 : (batch_size, num_heads, key의 문장 길이, d_model/num_heads)
    # value 크기 : (batch_size, num_heads, value의 문장 길이, d_model/num_heads)
    # padding_mask : (batch_size, 1, 1, key의 문장 길이)

    # Q와 K의 곱. 어텐션 스코어 행렬.
    matmul_qk = tf.matmul(query, key, transpose_b=True)

    # 스케일링
    # dk의 루트값으로 나눠준다.
    depth = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(depth)

    # 마스킹. 어텐션 스코어 행렬의 마스킹 할 위치에 매우 작은 음수값을 넣는다.
    # 매우 작은 값이므로 소프트맥스 함수를 지나면 행렬의 해당 위치의 값은 0이 된다.
    if mask is not None:
        logits += (mask * -1e9)

    # 소프트맥스 함수는 마지막 차원인 key의 문장 길이 방향으로 수행된다.
    # attention weight : (batch_size, num_heads, query의 문장 길이, key의 문장 길이)
    attention_weights = tf.nn.softmax(logits, axis=-1)

    # output : (batch_size, num_heads, query의 문장 길이, d_model/num_heads)
    output = tf.matmul(attention_weights, value)

    return output, attention_weights
```


```python
class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads, name="multi_head_attention"):
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        # d_model을 num_heads로 나눈 값.
        # 논문 기준 : 64
        self.depth = d_model // self.num_heads

        # WQ, WK, WV에 해당하는 밀집층 정의
        self.query_dense = tf.keras.layers.Dense(units=d_model)
        self.key_dense = tf.keras.layers.Dense(units=d_model)
        self.value_dense = tf.keras.layers.Dense(units=d_model)

        # WO에 해당하는 밀집층 정의
        self.dense = tf.keras.layers.Dense(units=d_model)

    # num_heads 개수만큼 q, k, v를 split하는 함수
    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(
            inputs, shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def call(self, inputs):
        query, key, value, mask = inputs['query'], inputs['key'], inputs[
            'value'], inputs['mask']
        batch_size = tf.shape(query)[0]

        # 1. WQ, WK, WV에 해당하는 밀집층 지나기
        # q : (batch_size, query의 문장 길이, d_model)
        # k : (batch_size, key의 문장 길이, d_model)
        # v : (batch_size, value의 문장 길이, d_model)
        # 참고) 인코더(k, v)-디코더(q) 어텐션에서는 query 길이와 key, value의 길이는 다를 수 있다.
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # 2. 헤드 나누기
        # q : (batch_size, num_heads, query의 문장 길이, d_model/num_heads)
        # k : (batch_size, num_heads, key의 문장 길이, d_model/num_heads)
        # v : (batch_size, num_heads, value의 문장 길이, d_model/num_heads)
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        # 3. 스케일드 닷 프로덕트 어텐션. 앞서 구현한 함수 사용.
        # (batch_size, num_heads, query의 문장 길이, d_model/num_heads)
        scaled_attention, _ = scaled_dot_product_attention(query, key, value, mask)
        # (batch_size, query의 문장 길이, num_heads, d_model/num_heads)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # 4. 헤드 연결(concatenate)하기
        # (batch_size, query의 문장 길이, d_model)
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))

        # 5. WO에 해당하는 밀집층 지나기
        # (batch_size, query의 문장 길이, d_model)
        outputs = self.dense(concat_attention)

        return outputs
```


```python
def create_padding_mask(x):
    mask = tf.cast(tf.math.equal(x, 0), tf.float32)
    # (batch_size, 1, 1, key의 문장 길이)
    return mask[:, tf.newaxis, tf.newaxis, :]
```


```python
def encoder_layer(dff, d_model, num_heads, dropout, name="encoder_layer"):
    inputs = tf.keras.Input(shape=(None, d_model), name="inputs")

    # 인코더는 패딩 마스크 사용
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

    # 멀티-헤드 어텐션 (첫번째 서브층 / 셀프 어텐션)
    attention = MultiHeadAttention(
      d_model, num_heads, name="attention")({
          'query': inputs, 'key': inputs, 'value': inputs, # Q = K = V
          'mask': padding_mask # 패딩 마스크 사용
      })

    # 드롭아웃 + 잔차 연결과 층 정규화
    attention = tf.keras.layers.Dropout(rate=dropout)(attention)
    attention = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(inputs + attention)

    # 포지션 와이즈 피드 포워드 신경망 (두번째 서브층)
    outputs = tf.keras.layers.Dense(units=dff, activation='relu')(attention)
    outputs = tf.keras.layers.Dense(units=d_model)(outputs)

    # 드롭아웃 + 잔차 연결과 층 정규화
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(attention + outputs)

    return tf.keras.Model(
      inputs=[inputs, padding_mask], outputs=outputs, name=name)
```


```python
def encoder(vocab_size, num_layers, dff, d_model, num_heads, dropout, name="encoder"):
    inputs = tf.keras.Input(shape=(None,), name="inputs")

    # 인코더는 패딩 마스크 사용
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

    # 포지셔널 인코딩 + 드롭아웃
    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)
    outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

    # 인코더를 num_layers개 쌓기
    for i in range(num_layers):
        outputs = encoder_layer(dff=dff, d_model=d_model, num_heads=num_heads,
            dropout=dropout, name="encoder_layer_{}".format(i),
        )([outputs, padding_mask])

    return tf.keras.Model(
      inputs=[inputs, padding_mask], outputs=outputs, name=name)
```


```python
# 디코더의 첫번째 서브층(sublayer)에서 미래 토큰을 Mask하는 함수
def create_look_ahead_mask(x):
    seq_len = tf.shape(x)[1]
# tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0) 
# ==
# tf.Tensor(
# [[1. 0. 0. 0. 0.]
#  [1. 1. 0. 0. 0.]
#  [1. 1. 1. 0. 0.]
#  [1. 1. 1. 1. 0.]
#  [1. 1. 1. 1. 1.]], shape=(5, 5), dtype=float32)
# (5, 5)
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    padding_mask = create_padding_mask(x)
    return tf.maximum(look_ahead_mask, padding_mask)
```


```python
def decoder_layer(dff, d_model, num_heads, dropout, name="decoder_layer"):
    inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
    enc_outputs = tf.keras.Input(shape=(None, d_model), name="encoder_outputs")

    # 룩어헤드 마스크(첫번째 서브층)
    look_ahead_mask = tf.keras.Input(
      shape=(1, None, None), name="look_ahead_mask")

    # 패딩 마스크(두번째 서브층)
    padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

    # 멀티-헤드 어텐션 (첫번째 서브층 / 마스크드 셀프 어텐션)
    attention1 = MultiHeadAttention(
      d_model, num_heads, name="attention_1")(inputs={
          'query': inputs, 'key': inputs, 'value': inputs, # Q = K = V
          'mask': look_ahead_mask # 룩어헤드 마스크
      })

    # 잔차 연결과 층 정규화
    attention1 = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(attention1 + inputs)

    # 멀티-헤드 어텐션 (두번째 서브층 / 디코더-인코더 어텐션)
    attention2 = MultiHeadAttention(
      d_model, num_heads, name="attention_2")(inputs={
          'query': attention1, 'key': enc_outputs, 'value': enc_outputs, # Q != K = V
          'mask': padding_mask # 패딩 마스크
      })

    # 드롭아웃 + 잔차 연결과 층 정규화
    attention2 = tf.keras.layers.Dropout(rate=dropout)(attention2)
    attention2 = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(attention2 + attention1)

    # 포지션 와이즈 피드 포워드 신경망 (세번째 서브층)
    outputs = tf.keras.layers.Dense(units=dff, activation='relu')(attention2)
    outputs = tf.keras.layers.Dense(units=d_model)(outputs)

    # 드롭아웃 + 잔차 연결과 층 정규화
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(outputs + attention2)

    return tf.keras.Model(
      inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
      outputs=outputs,
      name=name)
```


```python
def decoder(vocab_size, num_layers, dff,
            d_model, num_heads, dropout,
            name='decoder'):
    inputs = tf.keras.Input(shape=(None,), name='inputs')
    enc_outputs = tf.keras.Input(shape=(None, d_model), name='encoder_outputs')

    # 디코더는 룩어헤드 마스크(첫번째 서브층)와 패딩 마스크(두번째 서브층) 둘 다 사용.
    look_ahead_mask = tf.keras.Input(
      shape=(1, None, None), name='look_ahead_mask')
    padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

    # 포지셔널 인코딩 + 드롭아웃
    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)
    outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

    # 디코더를 num_layers개 쌓기
    for i in range(num_layers):
        outputs = decoder_layer(dff=dff, d_model=d_model, num_heads=num_heads,
            dropout=dropout, name='decoder_layer_{}'.format(i),
        )(inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask])

    return tf.keras.Model(
      inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
      outputs=outputs,
      name=name)
```


```python
def transformer(vocab_size, num_layers, dff,
                d_model, num_heads, dropout,
                name="transformer"):

    # 인코더의 입력
    inputs = tf.keras.Input(shape=(None,), name="inputs")

    # 디코더의 입력
    dec_inputs = tf.keras.Input(shape=(None,), name="dec_inputs")

    # 인코더의 패딩 마스크
    enc_padding_mask = tf.keras.layers.Lambda(
      create_padding_mask, output_shape=(1, 1, None),
      name='enc_padding_mask')(inputs)

    # 디코더의 룩어헤드 마스크(첫번째 서브층)
    look_ahead_mask = tf.keras.layers.Lambda(
      create_look_ahead_mask, output_shape=(1, None, None),
      name='look_ahead_mask')(dec_inputs)

    # 디코더의 패딩 마스크(두번째 서브층)
    dec_padding_mask = tf.keras.layers.Lambda(
      create_padding_mask, output_shape=(1, 1, None),
      name='dec_padding_mask')(inputs)

    # 인코더의 출력은 enc_outputs. 디코더로 전달된다.
    enc_outputs = encoder(vocab_size=vocab_size, num_layers=num_layers, dff=dff,
      d_model=d_model, num_heads=num_heads, dropout=dropout,
    )(inputs=[inputs, enc_padding_mask]) # 인코더의 입력은 입력 문장과 패딩 마스크

    # 디코더의 출력은 dec_outputs. 출력층으로 전달된다.
    dec_outputs = decoder(vocab_size=vocab_size, num_layers=num_layers, dff=dff,
      d_model=d_model, num_heads=num_heads, dropout=dropout,
    )(inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])

    # 다음 단어 예측을 위한 출력층
    outputs = tf.keras.layers.Dense(units=vocab_size, name="outputs")(dec_outputs)

    return tf.keras.Model(inputs=[inputs, dec_inputs], outputs=outputs, name=name)
```


```python
# hyperparameter 또는 cross-validation과 같이 여러 모델을 연속적으로 생성할 때 사용
# 메모리 확보, 속도 저하 방지 목적
tf.keras.backend.clear_session()

# Hyper-parameters
D_MODEL = 256
NUM_LAYERS = 2
NUM_HEADS = 8
DFF = 512
DROPOUT = 0.1

model = transformer(
    vocab_size=VOCAB_SIZE,
    num_layers=NUM_LAYERS,
    dff=DFF,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    dropout=DROPOUT)
```

    (1, 8180, 256)
    (1, 8180, 256)
    

학습률과 옵티마이저를 정의하고 모델을 컴파일합니다.


```python
def loss_function(y_true, y_pred):
    y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))
    
    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="none")(y_true, y_pred) 
    # 클래스 분류 문제에서 softmax 함수를 거치면 from_logits = False(default값), 그렇지 않으면 from_logits = True
    # reduction="none" == https://www.tensorflow.org/api_docs/python/tf/keras/losses/SparseCategoricalCrossentropy
    # (차원을 줄이지 않음)
    
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    loss = tf.multiply(loss, mask)
    
    return tf.reduce_mean(loss)
```


```python
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps
        
    def __call__(self, step):
        arg1 = tf.math.rsqrt(step) # 제곱근의 역수
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
```


```python
learning_rate = CustomSchedule(D_MODEL)

optimizer = tf.keras.optimizers.Adam(
    learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

def accuracy(y_true, y_pred):
    # 레이블의 크기는 (batch_size, MAX_LENGTH - 1)
    y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))
    return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)

model.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy])
```


```python
model.summary()
```

    Model: "transformer"
    __________________________________________________________________________________________________
     Layer (type)                   Output Shape         Param #     Connected to                     
    ==================================================================================================
     inputs (InputLayer)            [(None, None)]       0           []                               
                                                                                                      
     dec_inputs (InputLayer)        [(None, None)]       0           []                               
                                                                                                      
     enc_padding_mask (Lambda)      (None, 1, 1, None)   0           ['inputs[0][0]']                 
                                                                                                      
     encoder (Functional)           (None, None, 256)    3148288     ['inputs[0][0]',                 
                                                                      'enc_padding_mask[0][0]']       
                                                                                                      
     look_ahead_mask (Lambda)       (None, 1, None, Non  0           ['dec_inputs[0][0]']             
                                    e)                                                                
                                                                                                      
     dec_padding_mask (Lambda)      (None, 1, 1, None)   0           ['inputs[0][0]']                 
                                                                                                      
     decoder (Functional)           (None, None, 256)    3675648     ['dec_inputs[0][0]',             
                                                                      'encoder[0][0]',                
                                                                      'look_ahead_mask[0][0]',        
                                                                      'dec_padding_mask[0][0]']       
                                                                                                      
     outputs (Dense)                (None, None, 8180)   2102260     ['decoder[0][0]']                
                                                                                                      
    ==================================================================================================
    Total params: 8,926,196
    Trainable params: 8,926,196
    Non-trainable params: 0
    __________________________________________________________________________________________________
    

총 50회 모델을 학습합니다.


```python
EPOCHS = 50
model.fit(dataset, epochs=EPOCHS)
```

    Epoch 1/50
    185/185 [==============================] - 20s 75ms/step - loss: 1.4490 - accuracy: 0.0269
    Epoch 2/50
    185/185 [==============================] - 14s 75ms/step - loss: 1.1781 - accuracy: 0.0495
    Epoch 3/50
    185/185 [==============================] - 14s 76ms/step - loss: 1.0077 - accuracy: 0.0506
    Epoch 4/50
    185/185 [==============================] - 14s 76ms/step - loss: 0.9306 - accuracy: 0.0542
    Epoch 5/50
    185/185 [==============================] - 14s 76ms/step - loss: 0.8719 - accuracy: 0.0576
    Epoch 6/50
    185/185 [==============================] - 14s 76ms/step - loss: 0.8118 - accuracy: 0.0617
    Epoch 7/50
    185/185 [==============================] - 14s 76ms/step - loss: 0.7450 - accuracy: 0.0677
    Epoch 8/50
    185/185 [==============================] - 14s 76ms/step - loss: 0.6715 - accuracy: 0.0753
    Epoch 9/50
    185/185 [==============================] - 14s 76ms/step - loss: 0.5925 - accuracy: 0.0842
    Epoch 10/50
    185/185 [==============================] - 14s 76ms/step - loss: 0.5093 - accuracy: 0.0937
    Epoch 11/50
    185/185 [==============================] - 14s 76ms/step - loss: 0.4256 - accuracy: 0.1039
    Epoch 12/50
    185/185 [==============================] - 14s 76ms/step - loss: 0.3451 - accuracy: 0.1152
    Epoch 13/50
    185/185 [==============================] - 14s 76ms/step - loss: 0.2698 - accuracy: 0.1262
    Epoch 14/50
    185/185 [==============================] - 14s 76ms/step - loss: 0.2040 - accuracy: 0.1363
    Epoch 15/50
    185/185 [==============================] - 14s 76ms/step - loss: 0.1500 - accuracy: 0.1458
    Epoch 16/50
    185/185 [==============================] - 14s 76ms/step - loss: 0.1079 - accuracy: 0.1534
    Epoch 17/50
    185/185 [==============================] - 14s 76ms/step - loss: 0.0786 - accuracy: 0.1590
    Epoch 18/50
    185/185 [==============================] - 14s 76ms/step - loss: 0.0599 - accuracy: 0.1623
    Epoch 19/50
    185/185 [==============================] - 14s 77ms/step - loss: 0.0500 - accuracy: 0.1638
    Epoch 20/50
    185/185 [==============================] - 14s 77ms/step - loss: 0.0440 - accuracy: 0.1648
    Epoch 21/50
    185/185 [==============================] - 14s 78ms/step - loss: 0.0419 - accuracy: 0.1651
    Epoch 22/50
    185/185 [==============================] - 17s 90ms/step - loss: 0.0406 - accuracy: 0.1652
    Epoch 23/50
    185/185 [==============================] - 18s 98ms/step - loss: 0.0359 - accuracy: 0.1663
    Epoch 24/50
    185/185 [==============================] - 18s 98ms/step - loss: 0.0310 - accuracy: 0.1674
    Epoch 25/50
    185/185 [==============================] - 18s 97ms/step - loss: 0.0277 - accuracy: 0.1682
    Epoch 26/50
    185/185 [==============================] - 18s 96ms/step - loss: 0.0248 - accuracy: 0.1688
    Epoch 27/50
    185/185 [==============================] - 18s 96ms/step - loss: 0.0223 - accuracy: 0.1696
    Epoch 28/50
    185/185 [==============================] - 18s 98ms/step - loss: 0.0197 - accuracy: 0.1702
    Epoch 29/50
    185/185 [==============================] - 18s 97ms/step - loss: 0.0179 - accuracy: 0.1706
    Epoch 30/50
    185/185 [==============================] - 18s 96ms/step - loss: 0.0168 - accuracy: 0.1708
    Epoch 31/50
    185/185 [==============================] - 18s 95ms/step - loss: 0.0152 - accuracy: 0.1714
    Epoch 32/50
    185/185 [==============================] - 17s 94ms/step - loss: 0.0144 - accuracy: 0.1715
    Epoch 33/50
    185/185 [==============================] - 16s 87ms/step - loss: 0.0131 - accuracy: 0.1719
    Epoch 34/50
    185/185 [==============================] - 14s 76ms/step - loss: 0.0126 - accuracy: 0.1720
    Epoch 35/50
    185/185 [==============================] - 14s 77ms/step - loss: 0.0116 - accuracy: 0.1723
    Epoch 36/50
    185/185 [==============================] - 14s 77ms/step - loss: 0.0107 - accuracy: 0.1725
    Epoch 37/50
    185/185 [==============================] - 14s 77ms/step - loss: 0.0105 - accuracy: 0.1725
    Epoch 38/50
    185/185 [==============================] - 14s 76ms/step - loss: 0.0094 - accuracy: 0.1728
    Epoch 39/50
    185/185 [==============================] - 14s 77ms/step - loss: 0.0092 - accuracy: 0.1729
    Epoch 40/50
    185/185 [==============================] - 14s 75ms/step - loss: 0.0088 - accuracy: 0.1730
    Epoch 41/50
    185/185 [==============================] - 14s 75ms/step - loss: 0.0079 - accuracy: 0.1732
    Epoch 42/50
    185/185 [==============================] - 14s 75ms/step - loss: 0.0078 - accuracy: 0.1733
    Epoch 43/50
    185/185 [==============================] - 14s 75ms/step - loss: 0.0076 - accuracy: 0.1733
    Epoch 44/50
    185/185 [==============================] - 14s 75ms/step - loss: 0.0073 - accuracy: 0.1733
    Epoch 45/50
    185/185 [==============================] - 14s 75ms/step - loss: 0.0066 - accuracy: 0.1735
    Epoch 46/50
    185/185 [==============================] - 14s 75ms/step - loss: 0.0066 - accuracy: 0.1734
    Epoch 47/50
    185/185 [==============================] - 14s 75ms/step - loss: 0.0064 - accuracy: 0.1735
    Epoch 48/50
    185/185 [==============================] - 14s 75ms/step - loss: 0.0063 - accuracy: 0.1736
    Epoch 49/50
    185/185 [==============================] - 14s 75ms/step - loss: 0.0056 - accuracy: 0.1737
    Epoch 50/50
    185/185 [==============================] - 14s 75ms/step - loss: 0.0060 - accuracy: 0.1737
    




    <keras.callbacks.History at 0x20bf0ce5850>



## 6. 챗봇 평가하기.


```python
def evaluate(sentence):
    sentence = preprocess_sentence(sentence)
    sentence = tf.expand_dims(
        START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0
    )
    output = tf.expand_dims(START_TOKEN, 0)
    
    # 디코더의 예측 시작
    for i in range(MAX_LENGTH):
        predictions = model(inputs=[sentence, output], training=False)
        
        # 현재(마지막) 시점의 예측 단어를 받아온다.
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        
        # 만약 마지막 시점의 예측 단어가 종료 토큰이라면 예측을 중단
        if tf.equal(predicted_id, END_TOKEN[0]):
            break
        
        # 마지막 시점의 예측 단어를 출력에 연결한다.
        # 이는 for문을 통해서 디코더의 입력으로 사용될 예정이다.
        output = tf.concat([output, predicted_id], axis=-1)
    return tf.squeeze(output, axis=0)
```


```python
def predict(sentence):
    prediction = evaluate(sentence)
    
    predicted_sentence = tokenizer.decode(
        [i for i in prediction if i < tokenizer.vocab_size]
    )
    
    print('Input: {}'.format(sentence))
    print('Output: {}'.format(predicted_sentence))

    return predicted_sentence
```


```python
def preprocess_sentence(sentence):
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = sentence.strip()
    return sentence
```

학습된 트랜스포머에 임의로 생각나는 말들을 적어보았습니다.


```python
output = predict("영화 볼래?")
```

    Input: 영화 볼래?
    Output: 최신 영화가 좋을 것 같아요 .
    


```python
output = predict("고민이 있어")
```

    Input: 고민이 있어
    Output: 저는 생각을 종이에 끄젹여여 보는게 도움이 될 수도 있어요 .
    


```python
output = predict("너무 화가나")
```

    Input: 너무 화가나
    Output: 그럴수록 당신이 힘들 거예요 .
    


```python
output = predict("카페갈래?")
```

    Input: 카페갈래?
    Output: 저는 서로 마음을 이어주는 위로봇입니다 .
    


```python
output = predict("게임하고싶당")
```

    Input: 게임하고싶당
    Output: 몸을 피곤하게 하거나 다른 생각을 해보세요 .
    


```python
output = predict("게임하자")
```

    Input: 게임하자
    Output: 게임하세요 !

