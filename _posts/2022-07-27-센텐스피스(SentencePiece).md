---
layout: single
title:  "[tensorflow] 센텐스피스(SentencePiece)"
categories: DL
tag: [SentencePiece, tensorflow]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: false
---

**[공지사항]** ["출처: https://wikidocs.net/86657"](https://wikidocs.net/86657)
{: .notice--danger}

# 02) 센텐스피스(SentencePiece)
앞서 서브워드 토큰화를 위한 BPE(Byte Pair Encoding) 알고리즘과 그 외 BPE의 변형 알고리즘에 대해서 간단히 언급했습니다. 물론, 알고리즘을 아는 것도 중요하지만 어쩌면 그보다 더 중요한 것은 실무에서 바로 적용 가능하느냐의 문제일 수 있습니다. 이 경우 BPE를 포함하여 기타 서브워드 토크나이징 알고리즘들을 내장한 센텐스피스(SentencePiece)가 일반적으로 최선의 선택일 수 있습니다.

## 1. 센텐스피스(Sentencepiece)
논문 : [https://arxiv.org/pdf/1808.06226.pdf](https://arxiv.org/pdf/1808.06226.pdf) \
센텐스피스 깃허브 : [https://github.com/google/sentencepiece](https://github.com/google/sentencepiece)

내부 단어 분리를 위한 유용한 패키지로 구글의 센텐스피스(Sentencepiece)가 있습니다. 구글은 BPE 알고리즘과 Unigram Language Model Tokenizer를 구현한 센텐스피스를 깃허브에 공개하였습니다.

내부 단어 분리 알고리즘을 사용하기 위해서, 데이터에 단어 토큰화를 먼저 진행한 상태여야 한다면 이 단어 분리 알고리즘을 모든 언어에 사용하는 것은 쉽지 않습니다. 영어와 달리 한국어와 같은 언어는 단어 토큰화부터가 쉽지 않기 때문입니다. 그런데, 이런 사전 토큰화 작업(pretokenization)없이 전처리를 하지 않은 데이터(raw data)에 바로 단어 분리 토크나이저를 사용할 수 있다면, 이 토크나이저는 그 어떤 언어에도 적용할 수 있는 토크나이저가 될 것입니다. 센텐스피스는 이 이점을 살려서 구현되었습니다. 센텐스피스는 사전 토큰화 작업없이 단어 분리 토큰화를 수행하므로 언어에 종속되지 않습니다.


```python
!pip install sentencepiece
```

    Collecting sentencepiece
      Downloading sentencepiece-0.1.96-cp38-cp38-win_amd64.whl (1.1 MB)
    Installing collected packages: sentencepiece
    Successfully installed sentencepiece-0.1.96
    

## 2. IMDB 리뷰 토큰화하기


```python
import sentencepiece as spm
import pandas as pd
import urllib.request
import csv
```

IMDB 리뷰 데이터를 다운로드하고 이를 데이터프레임에 저장합니다.


```python
urllib.request.urlretrieve("https://raw.githubusercontent.com/LawrenceDuan/IMDb-Review-Analysis/master/IMDb_Reviews.csv", 
                           filename="IMDb_Reviews.csv")
```




    ('IMDb_Reviews.csv', <http.client.HTTPMessage at 0x24c94c73460>)




```python
train_df = pd.read_csv('IMDb_Reviews.csv')
train_df['review']
```




    0        My family and I normally do not watch local mo...
    1        Believe it or not, this was at one time the wo...
    2        After some internet surfing, I found the "Home...
    3        One of the most unheralded great works of anim...
    4        It was the Sixties, and anyone with long hair ...
                                   ...                        
    49995    the people who came up with this are SICK AND ...
    49996    The script is so so laughable... this in turn,...
    49997    "So there's this bride, you see, and she gets ...
    49998    Your mind will not be satisfied by this nobud...
    49999    The chaser's war on everything is a weekly sho...
    Name: review, Length: 50000, dtype: object




```python
print('리뷰 개수 :',len(train_df))
```

    리뷰 개수 : 50000
    

총 5만개의 샘플이 존재합니다. 센텐스피스의 입력으로 사용하기 위해서 데이터프레임을 txt 파일로 저장합니다


```python
with open("imdb_review.txt", "w", encoding="utf8") as f:
    f.write("\n".join(train_df["review"]))
```

센텐스피스로 단어 집합과 각 단어에 고유한 정수를 부여해보겠습니다.


```python
spm.SentencePieceTrainer.Train('--input=imdb_review.txt --model_prefix=imdb --vocab_size=5000 --model_type=bpe --max_sentence_length=9999')
```

각 인자가 의미하는 바는 다음과 같습니다.

- input : 학습시킬 파일
- model_prefix : 만들어질 모델 이름
- vocab_size : 단어 집합의 크기
- model_type : 사용할 모델 (unigram(default), bpe, char, word)
- max_sentence_length: 문장의 최대 길이
- pad_id, pad_piece: pad token id, 값
- unk_id, unk_piece: unknown token id, 값
- bos_id, bos_piece: begin of sentence token id, 값
- eos_id, eos_piece: end of sequence token id, 값
- user_defined_symbols: 사용자 정의 토큰

vocab 생성이 완료되면 imdb.model, imdb.vocab 파일 두개가 생성 됩니다. vocab 파일에서 학습된 서브워드들을 확인할 수 있습니다. 단어 집합의 크기를 확인하기 위해 vocab 파일을 데이터프레임에 저장해봅시다.


```python
vocab_list = pd.read_csv("imdb.vocab", sep="\t", header=None, quoting=csv.QUOTE_NONE)
vocab_list.sample(10)
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
      <th>0</th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>600</th>
      <td>▁old</td>
      <td>-597</td>
    </tr>
    <tr>
      <th>2277</th>
      <td>▁mil</td>
      <td>-2274</td>
    </tr>
    <tr>
      <th>3526</th>
      <td>unting</td>
      <td>-3523</td>
    </tr>
    <tr>
      <th>2804</th>
      <td>▁stat</td>
      <td>-2801</td>
    </tr>
    <tr>
      <th>4691</th>
      <td>▁Max</td>
      <td>-4688</td>
    </tr>
    <tr>
      <th>3892</th>
      <td>▁lover</td>
      <td>-3889</td>
    </tr>
    <tr>
      <th>1500</th>
      <td>▁Some</td>
      <td>-1497</td>
    </tr>
    <tr>
      <th>2602</th>
      <td>acy</td>
      <td>-2599</td>
    </tr>
    <tr>
      <th>3396</th>
      <td>▁journey</td>
      <td>-3393</td>
    </tr>
    <tr>
      <th>2735</th>
      <td>▁catch</td>
      <td>-2732</td>
    </tr>
  </tbody>
</table>
</div>



위에서 vocab_size의 인자를 통해 단어 집합의 크기를 5,000개로 제한하였으므로 단어 집합의 크기는 5,000개입니다.


```python
len(vocab_list)
```




    5000



이제 model 파일을 로드하여 단어 시퀀스를 정수 시퀀스로 바꾸는 인코딩 작업이나 반대로 변환하는 디코딩 작업을 할 수 있습니다.


```python
sp = spm.SentencePieceProcessor()
vocab_file = "imdb.model"
sp.load(vocab_file)
```




    True



아래의 두 가지 도구를 테스트해보겠습니다.

- encode_as_pieces : 문장을 입력하면 서브 워드 시퀀스로 변환합니다.
- encode_as_ids : 문장을 입력하면 정수 시퀀스로 변환합니다.


```python
lines = [
  "I didn't at all think of it this way.",
  "I have waited a long time for someone to film"
]

for line in lines:
    print(line)
    print(sp.encode_as_pieces(line))
    print(sp.encode_as_ids(line))
    print()
```

    I didn't at all think of it this way.
    ['▁I', '▁didn', "'", 't', '▁at', '▁all', '▁think', '▁of', '▁it', '▁this', '▁way', '.']
    [41, 623, 4950, 4926, 138, 169, 378, 30, 58, 73, 413, 4945]
    
    I have waited a long time for someone to film
    ['▁I', '▁have', '▁wa', 'ited', '▁a', '▁long', '▁time', '▁for', '▁someone', '▁to', '▁film']
    [41, 141, 1364, 1120, 4, 666, 285, 92, 1078, 33, 91]
    
    

- GetPieceSize() : 단어 집합의 크기를 확인합니다.


```python
sp.GetPieceSize()
```




    5000



- idToPiece : 정수로부터 맵핑되는 서브 워드로 변환합니다.


```python
sp.IdToPiece(430)
```




    '▁character'



- PieceToId : 서브워드로부터 맵핑되는 정수로 변환합니다.


```python
sp.PieceToId("▁character")
```




    430



- DecodeIds : 정수 시퀀스로부터 문장으로 변환합니다.


```python
sp.DecodeIds([41, 141, 1364, 1120, 4, 666, 285, 92, 1078, 33, 91])
```




    'I have waited a long time for someone to film'



- DecodePieces : 서브워드 시퀀스로부터 문장으로 변환합니다.


```python
sp.DecodePieces(['▁I', '▁have', '▁wa', 'ited', '▁a', '▁long', '▁time', '▁for', '▁someone', '▁to', '▁film'])
```




    'I have waited a long time for someone to film'



- encode : 문장으로부터 인자값에 따라서 정수 시퀀스 또는 서브워드 시퀀스로 변환 가능합니다.


```python
print(sp.encode('I have waited a long time for someone to film', out_type=str))
print(sp.encode('I have waited a long time for someone to film', out_type=int))
```

    ['▁I', '▁have', '▁wa', 'ited', '▁a', '▁long', '▁time', '▁for', '▁someone', '▁to', '▁film']
    [41, 141, 1364, 1120, 4, 666, 285, 92, 1078, 33, 91]
    

## 3. 네이버 영화 리뷰 토큰화하기
네이버 영화 리뷰 데이터에 대해서 위의 IMDB 리뷰 데이터와 동일한 과정을 진행해보겠습니다.


```python
import pandas as pd
import sentencepiece as spm
import urllib.request
import csv
```

네이버 영화 리뷰 데이터를 다운로드하여 데이터프레임에 저장합니다.


```python
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings.txt", 
                           filename="ratings.txt")
```




    ('ratings.txt', <http.client.HTTPMessage at 0x24c950d9d90>)




```python
naver_df = pd.read_table('ratings.txt')
naver_df[:5]
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
      <th>id</th>
      <th>document</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8112052</td>
      <td>어릴때보고 지금다시봐도 재밌어요ㅋㅋ</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8132799</td>
      <td>디자인을 배우는 학생으로, 외국디자이너와 그들이 일군 전통을 통해 발전해가는 문화산...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4655635</td>
      <td>폴리스스토리 시리즈는 1부터 뉴까지 버릴께 하나도 없음.. 최고.</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9251303</td>
      <td>와.. 연기가 진짜 개쩔구나.. 지루할거라고 생각했는데 몰입해서 봤다.. 그래 이런...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10067386</td>
      <td>안개 자욱한 밤하늘에 떠 있는 초승달 같은 영화.</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



총 20만개의 샘플이 존재합니다.


```python
print('리뷰 개수 :',len(naver_df))
```

    리뷰 개수 : 200000
    

네이버 영화 리뷰 데이터의 경우 Null 값이 존재하므로 이를 제거한 후에 수행합니다.


```python
print(naver_df.isnull().values.any())
```

    True
    


```python
naver_df = naver_df.dropna(how='any')
print(naver_df.isnull().values.any())
```

    False
    


```python
print('리뷰 개수 :',len(naver_df))
```

    리뷰 개수 : 199992
    

최종적으로 199,992개의 샘플을 naver_review.txt 파일에 저장한 후에 센텐스피스를 통해 단어 집합을 생성합니다.


```python
with open('naver_review.txt', 'w', encoding='utf8') as f:
    f.write('\n'.join(naver_df['document']))
```


```python
spm.SentencePieceTrainer.Train('--input=naver_review.txt --model_prefix=naver --vocab_size=5000 --model_type=bpe --max_sentence_length=9999')
```

vocab 생성이 완료되면 naver.model, naver.vocab 파일 두개가 생성 됩니다. .vocab 에서 학습된 subwords를 확인할 수 있습니다.


```python
vocab_list = pd.read_csv("naver.vocab", sep="\t", header=None, quoting=csv.QUOTE_NONE)
vocab_list[:10]
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
      <th>0</th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>&lt;unk&gt;</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>&lt;s&gt;</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>&lt;/s&gt;</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>..</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>영화</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>▁영화</td>
      <td>-2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>▁이</td>
      <td>-3</td>
    </tr>
    <tr>
      <th>7</th>
      <td>▁아</td>
      <td>-4</td>
    </tr>
    <tr>
      <th>8</th>
      <td>...</td>
      <td>-5</td>
    </tr>
    <tr>
      <th>9</th>
      <td>▁그</td>
      <td>-6</td>
    </tr>
  </tbody>
</table>
</div>




```python
vocab_list.sample(10)
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
      <th>0</th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>414</th>
      <td>▁말이</td>
      <td>-411</td>
    </tr>
    <tr>
      <th>3749</th>
      <td>_</td>
      <td>-3746</td>
    </tr>
    <tr>
      <th>77</th>
      <td>▁인</td>
      <td>-74</td>
    </tr>
    <tr>
      <th>4708</th>
      <td>펏</td>
      <td>-4705</td>
    </tr>
    <tr>
      <th>1021</th>
      <td>▁진정</td>
      <td>-1018</td>
    </tr>
    <tr>
      <th>2194</th>
      <td>감을</td>
      <td>-2191</td>
    </tr>
    <tr>
      <th>3338</th>
      <td>주</td>
      <td>-3335</td>
    </tr>
    <tr>
      <th>1145</th>
      <td>▁말도</td>
      <td>-1142</td>
    </tr>
    <tr>
      <th>1953</th>
      <td>▁깜</td>
      <td>-1950</td>
    </tr>
    <tr>
      <th>3724</th>
      <td>승</td>
      <td>-3721</td>
    </tr>
  </tbody>
</table>
</div>



Vocabulary 에는 unknown, 문장의 시작, 문장의 끝을 의미하는 special token이 0, 1, 2에 사용되었습니다.


```python
len(vocab_list)
```




    5000



설정한대로 5000개의 서브워드가 단어 집합에 존재합니다.




```python
spm.SentencePieceProcessor()
vocab_file = "naver.model"
sp.load(vocab_file)
```




    True




```python
lines = [
  "뭐 이딴 것도 영화냐.",
  "진짜 최고의 영화입니다 ㅋㅋ",
]

for line in lines:
    print(line)
    print(sp.encode_as_pieces(line))
    print(sp.encode_as_ids(line))
    print()
```

    뭐 이딴 것도 영화냐.
    ['▁뭐', '▁이딴', '▁것도', '▁영화냐', '.']
    [132, 966, 1296, 2590, 3276]
    
    진짜 최고의 영화입니다 ㅋㅋ
    ['▁진짜', '▁최고의', '▁영화입니다', '▁ᄏᄏ']
    [54, 200, 821, 85]
    
    


```python
sp.GetPieceSize()
```




    5000




```python
sp.IdToPiece(4)
```




    '영화'




```python
sp.PieceToId("영화")
```




    4




```python
sp.DecodeIds([54, 200, 821, 85])
```




    '진짜 최고의 영화입니다 ᄏᄏ'




```python
sp.DecodePieces(['▁진짜', '▁최고의', '▁영화입니다', '▁ᄏᄏ'])
```




    '진짜 최고의 영화입니다 ᄏᄏ'




```python
print(sp.encode('진짜 최고의 영화입니다 ㅋㅋ', out_type=str))
print(sp.encode('진짜 최고의 영화입니다 ㅋㅋ', out_type=int))
```

    ['▁진짜', '▁최고의', '▁영화입니다', '▁ᄏᄏ']
    [54, 200, 821, 85]
    

