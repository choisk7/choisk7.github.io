---
layout: single
title:  "Scikit-learn으로 ML 모델링부터 ML Pipeline까지"
categories: ML
tag: [ML pipeline, scikit-learn]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: false
---
**[공지사항]** ["출처: https://towardsdatascience.com/from-ml-model-to-ml-pipeline-9f95c32c6512"](https://towardsdatascience.com/from-ml-model-to-ml-pipeline-9f95c32c6512)
{: .notice--danger}


# From ML Model to ML Pipeline With Scikit-learn in Python
Machine learning model을 구축하는 것은 올바른 알고리즘을 선택하고 hyperparameter를 tuning하는 것만이 아닙니다. model experimentation이 시작되기 전에 data wrangling, feature engineering을 하는 데 상당한 시간이 소요됩니다. 이러한 preprocessing은 workflow를 대응하기 힘드고, 추적하기 어렵게 만들 수 있습니다. ML model에서 ML pipeline에 초점을 맞추고 preprocessing을 모델 구축의 필수 부분으로 보는 것은 workflow를 보다 체계적으로 유지하는 데 도움이 될 수 있습니다. 이 아티클에서는 먼저 모델의 데이터를 preprocessing하는 잘못된 방법을 살펴본 다음, ML pipeline을 구축하는 두 가지 올바른 접근 방식을 배웁니다.

<p align="center"><img src="/assets/images/221018/1.png"></p>

ML pipeline은 context에 따라 여러 가지 정의가 있습니다. 이 아티클에서 ML Pipeline은 preprocessing steps와 모델의 collection으로 정의됩니다. 즉, raw data가 ML pipeline에 전달되면 데이터를 올바른 형식으로 preprocessing하고 모델을 사용하여 데이터를 채점한 후, 예측 점수가 튀어나옵니다.

# 0. Setup


```python
from seaborn import load_dataset
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn import set_config
set_config(display="diagram")

columns = ['alive', 'class', 'embarked', 'who', 'alone', 'adult_male']
df = load_dataset("titanic").drop(columns=columns)
df["deck"] = df["deck"].astype("object")
print(df.shape)
df.head()
```

    (891, 9)
    




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
      <th>survived</th>
      <th>pclass</th>
      <th>sex</th>
      <th>age</th>
      <th>sibsp</th>
      <th>parch</th>
      <th>fare</th>
      <th>deck</th>
      <th>embark_town</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>Southampton</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>C</td>
      <td>Cherbourg</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>Southampton</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>C</td>
      <td>Southampton</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>Southampton</td>
    </tr>
  </tbody>
</table>
</div>



나중에 참고할 수 있도록 자주 사용하는 variables를 정의합니다.


```python
SEED = 42
TARGET = "survived"
FEATURES = df.columns.drop(TARGET)

NUMERICAL = df[FEATURES].select_dtypes("number").columns
print(f"Numerical features: {', '.join(NUMERICAL)}")

CATEGORICAL = pd.Index(np.setdiff1d(FEATURES, NUMERICAL))
print(f"Categorical features: {', '.join(CATEGORICAL)}")
```

    Numerical features: pclass, age, sibsp, parch, fare
    Categorical features: deck, embark_town, sex
    

이제 첫번째 접근을 살펴봅시다

#  1. Wrong approach
preprocessing을 할 때, 아래처럼 pandas methods를 사용하는 것은 흔치 않습니다


```python
df_num_imputed = df[NUMERICAL].fillna(df[NUMERICAL].mean())
df_num_scaled = df_num_imputed.subtract(df_num_imputed.min(), axis=1)\
                              .divide(df_num_imputed.max()-df_num_imputed.min(), axis=1)

df_cat_imputed = df[CATEGORICAL].fillna("missing")
df_cat_encoded = pd.get_dummies(df_cat_imputed, drop_first=True)

df_preprocessed = df_num_scaled.join(df_cat_encoded)
df_preprocessed.head()
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
      <th>pclass</th>
      <th>age</th>
      <th>sibsp</th>
      <th>parch</th>
      <th>fare</th>
      <th>deck_B</th>
      <th>deck_C</th>
      <th>deck_D</th>
      <th>deck_E</th>
      <th>deck_F</th>
      <th>deck_G</th>
      <th>deck_missing</th>
      <th>embark_town_Queenstown</th>
      <th>embark_town_Southampton</th>
      <th>embark_town_missing</th>
      <th>sex_male</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>0.271174</td>
      <td>0.125</td>
      <td>0.0</td>
      <td>0.014151</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.472229</td>
      <td>0.125</td>
      <td>0.0</td>
      <td>0.139136</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>0.321438</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.015469</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.434531</td>
      <td>0.125</td>
      <td>0.0</td>
      <td>0.103644</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>0.434531</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.015713</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



여기서는 missing values와 0에서 1 사이의 scaled numerical variables, one-hot encoding을 한 categoricla variables로 imputing했습니다. preprocessing 후, 데이터를 분할하고 모델을 훈련시킵니다:


```python
X_train, X_test, y_train, y_test = train_test_split(df_preprocessed, df[TARGET],
                                                    test_size=.2, random_state=SEED,
                                                    stratify=df[TARGET])

model = LogisticRegression()
model.fit(X_train, y_train)
```




<style>#sk-d6f3bab7-da92-4376-aa1e-29fbe041ba81 {color: black;background-color: white;}#sk-d6f3bab7-da92-4376-aa1e-29fbe041ba81 pre{padding: 0;}#sk-d6f3bab7-da92-4376-aa1e-29fbe041ba81 div.sk-toggleable {background-color: white;}#sk-d6f3bab7-da92-4376-aa1e-29fbe041ba81 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-d6f3bab7-da92-4376-aa1e-29fbe041ba81 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-d6f3bab7-da92-4376-aa1e-29fbe041ba81 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-d6f3bab7-da92-4376-aa1e-29fbe041ba81 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-d6f3bab7-da92-4376-aa1e-29fbe041ba81 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-d6f3bab7-da92-4376-aa1e-29fbe041ba81 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-d6f3bab7-da92-4376-aa1e-29fbe041ba81 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-d6f3bab7-da92-4376-aa1e-29fbe041ba81 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-d6f3bab7-da92-4376-aa1e-29fbe041ba81 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-d6f3bab7-da92-4376-aa1e-29fbe041ba81 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-d6f3bab7-da92-4376-aa1e-29fbe041ba81 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-d6f3bab7-da92-4376-aa1e-29fbe041ba81 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-d6f3bab7-da92-4376-aa1e-29fbe041ba81 div.sk-estimator:hover {background-color: #d4ebff;}#sk-d6f3bab7-da92-4376-aa1e-29fbe041ba81 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-d6f3bab7-da92-4376-aa1e-29fbe041ba81 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-d6f3bab7-da92-4376-aa1e-29fbe041ba81 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 2em;bottom: 0;left: 50%;}#sk-d6f3bab7-da92-4376-aa1e-29fbe041ba81 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;}#sk-d6f3bab7-da92-4376-aa1e-29fbe041ba81 div.sk-item {z-index: 1;}#sk-d6f3bab7-da92-4376-aa1e-29fbe041ba81 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;}#sk-d6f3bab7-da92-4376-aa1e-29fbe041ba81 div.sk-parallel::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 2em;bottom: 0;left: 50%;}#sk-d6f3bab7-da92-4376-aa1e-29fbe041ba81 div.sk-parallel-item {display: flex;flex-direction: column;position: relative;background-color: white;}#sk-d6f3bab7-da92-4376-aa1e-29fbe041ba81 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-d6f3bab7-da92-4376-aa1e-29fbe041ba81 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-d6f3bab7-da92-4376-aa1e-29fbe041ba81 div.sk-parallel-item:only-child::after {width: 0;}#sk-d6f3bab7-da92-4376-aa1e-29fbe041ba81 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;position: relative;}#sk-d6f3bab7-da92-4376-aa1e-29fbe041ba81 div.sk-label label {font-family: monospace;font-weight: bold;background-color: white;display: inline-block;line-height: 1.2em;}#sk-d6f3bab7-da92-4376-aa1e-29fbe041ba81 div.sk-label-container {position: relative;z-index: 2;text-align: center;}#sk-d6f3bab7-da92-4376-aa1e-29fbe041ba81 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-d6f3bab7-da92-4376-aa1e-29fbe041ba81 div.sk-text-repr-fallback {display: none;}</style><div id="sk-d6f3bab7-da92-4376-aa1e-29fbe041ba81" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LogisticRegression()</pre><b>Please rerun this cell to show the HTML repr or trust the notebook.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="de0dba2a-c4f3-4f4c-b9fc-1a8ba051dea3" type="checkbox" checked><label for="de0dba2a-c4f3-4f4c-b9fc-1a8ba051dea3" class="sk-toggleable__label sk-toggleable__label-arrow">LogisticRegression</label><div class="sk-toggleable__content"><pre>LogisticRegression()</pre></div></div></div></div></div>



이제 이 접근 방식이 왜 잘못되었는지 살펴봅시다:

- **Imputation:** 전체 데이터 세트가 아니라 training 데이터의 mean 값으로 numerical variables를 imputing했어야 합니다.

- **Scaling:** Min과 Max도 training 데이터에서 계산되어야 합니다.

- **Encoding:** Categories는 training 데이터에서 추론해야 합니다. 또한, 데이터가 preprocessing 전에 분할되더라도 ```pd.get_dummies(X_train)``` 및 ```pd.get_dummies(X_test)```를 사용한 one-hot encoding은 서로 다른 training 및 test 데이터를 만들 수 있습니다(칼럼이 category에 따라 다를 수 있음). 따라서, ```pd.get_dummies()```는 모델에 대한 데이터를 준비할 때 one-hot-encoding에 사용하지 않아야 합니다.

> test 데이터는 전처리 전에 따로 보관해야 합니다. preprocessing에 사용되는 mean, min 및 max와 같은 모든 sttistics는 training 데이터에서 만들어져야 합니다. 그렇지 않으면, data leakage problem이 발생합니다.

이제 모델을 평가해 보겠습니다. 여기서는 ROC-AUC를 사용하여 모델을 평가합니다. ROC-AUC를 계산하는 함수는 후속 approach를 평가하는 데 유용하기 때문에 만들어줍니다


```python
def calculate_roc_auc(model_pipe, X, y):
    """Calculate roc auc score. 
    
    Parameters:
    ===========
    model_pipe: sklearn model or pipeline
    X: features
    y: true target
    """
    y_proba = model_pipe.predict_proba(X)[:,1]
    return roc_auc_score(y, y_proba)
  
print(f"Train ROC-AUC: {calculate_roc_auc(model, X_train, y_train):.4f}")
print(f"Test ROC-AUC: {calculate_roc_auc(model, X_test, y_test):.4f}")
```

    Train ROC-AUC: 0.8669
    Test ROC-AUC: 0.8329
    

# 2. Correct approach but …
데이터를 먼저 분할하고, Scikit-learn의 transformers를 사용하여 데이터를 preprocessing해 data leakage를 방지합니다.


```python
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=TARGET), df[TARGET],
                                                    test_size=.2, random_state=SEED,
                                                    stratify=df[TARGET])
num_imputer = SimpleImputer(strategy="mean")
train_num_imputed = num_imputer.fit_transform(X_train[NUMERICAL])

scaler = MinMaxScaler()
train_num_scaled = scaler.fit_transform(train_num_imputed)

cat_imputer = SimpleImputer(strategy="constant", fill_value="missing")
train_cat_imputed = cat_imputer.fit_transform(X_train[CATEGORICAL])

encoder = OneHotEncoder(drop="first", handle_unknown="ignore", sparse=False)
train_cat_encoded = encoder.fit_transform(train_cat_imputed)

train_preprocessed = np.concatenate((train_num_scaled, train_cat_encoded), axis=1)

columns = np.append(NUMERICAL, encoder.get_feature_names_out(CATEGORICAL))
pd.DataFrame(train_preprocessed, columns=columns, index=X_train.index).head()
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
      <th>pclass</th>
      <th>age</th>
      <th>sibsp</th>
      <th>parch</th>
      <th>fare</th>
      <th>deck_B</th>
      <th>deck_C</th>
      <th>deck_D</th>
      <th>deck_E</th>
      <th>deck_F</th>
      <th>deck_G</th>
      <th>deck_missing</th>
      <th>embark_town_Queenstown</th>
      <th>embark_town_Southampton</th>
      <th>embark_town_missing</th>
      <th>sex_male</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>692</th>
      <td>1.0</td>
      <td>0.369285</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.110272</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>481</th>
      <td>0.5</td>
      <td>0.369285</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>527</th>
      <td>0.0</td>
      <td>0.369285</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.432884</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>855</th>
      <td>1.0</td>
      <td>0.220910</td>
      <td>0.000</td>
      <td>0.166667</td>
      <td>0.018250</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>801</th>
      <td>0.5</td>
      <td>0.384267</td>
      <td>0.125</td>
      <td>0.166667</td>
      <td>0.051237</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



이제 모델을 학습시켜 봅시다


```python
model = LogisticRegression()
model.fit(train_preprocessed, y_train)
```




<style>#sk-dd05379a-508a-47c6-8872-c8bd85a1458c {color: black;background-color: white;}#sk-dd05379a-508a-47c6-8872-c8bd85a1458c pre{padding: 0;}#sk-dd05379a-508a-47c6-8872-c8bd85a1458c div.sk-toggleable {background-color: white;}#sk-dd05379a-508a-47c6-8872-c8bd85a1458c label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-dd05379a-508a-47c6-8872-c8bd85a1458c label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-dd05379a-508a-47c6-8872-c8bd85a1458c label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-dd05379a-508a-47c6-8872-c8bd85a1458c div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-dd05379a-508a-47c6-8872-c8bd85a1458c div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-dd05379a-508a-47c6-8872-c8bd85a1458c div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-dd05379a-508a-47c6-8872-c8bd85a1458c input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-dd05379a-508a-47c6-8872-c8bd85a1458c input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-dd05379a-508a-47c6-8872-c8bd85a1458c div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-dd05379a-508a-47c6-8872-c8bd85a1458c div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-dd05379a-508a-47c6-8872-c8bd85a1458c input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-dd05379a-508a-47c6-8872-c8bd85a1458c div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-dd05379a-508a-47c6-8872-c8bd85a1458c div.sk-estimator:hover {background-color: #d4ebff;}#sk-dd05379a-508a-47c6-8872-c8bd85a1458c div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-dd05379a-508a-47c6-8872-c8bd85a1458c div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-dd05379a-508a-47c6-8872-c8bd85a1458c div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 2em;bottom: 0;left: 50%;}#sk-dd05379a-508a-47c6-8872-c8bd85a1458c div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;}#sk-dd05379a-508a-47c6-8872-c8bd85a1458c div.sk-item {z-index: 1;}#sk-dd05379a-508a-47c6-8872-c8bd85a1458c div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;}#sk-dd05379a-508a-47c6-8872-c8bd85a1458c div.sk-parallel::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 2em;bottom: 0;left: 50%;}#sk-dd05379a-508a-47c6-8872-c8bd85a1458c div.sk-parallel-item {display: flex;flex-direction: column;position: relative;background-color: white;}#sk-dd05379a-508a-47c6-8872-c8bd85a1458c div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-dd05379a-508a-47c6-8872-c8bd85a1458c div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-dd05379a-508a-47c6-8872-c8bd85a1458c div.sk-parallel-item:only-child::after {width: 0;}#sk-dd05379a-508a-47c6-8872-c8bd85a1458c div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;position: relative;}#sk-dd05379a-508a-47c6-8872-c8bd85a1458c div.sk-label label {font-family: monospace;font-weight: bold;background-color: white;display: inline-block;line-height: 1.2em;}#sk-dd05379a-508a-47c6-8872-c8bd85a1458c div.sk-label-container {position: relative;z-index: 2;text-align: center;}#sk-dd05379a-508a-47c6-8872-c8bd85a1458c div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-dd05379a-508a-47c6-8872-c8bd85a1458c div.sk-text-repr-fallback {display: none;}</style><div id="sk-dd05379a-508a-47c6-8872-c8bd85a1458c" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LogisticRegression()</pre><b>Please rerun this cell to show the HTML repr or trust the notebook.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="51c162d2-2aab-4076-b9f3-7b0d9a65180d" type="checkbox" checked><label for="51c162d2-2aab-4076-b9f3-7b0d9a65180d" class="sk-toggleable__label sk-toggleable__label-arrow">LogisticRegression</label><div class="sk-toggleable__content"><pre>LogisticRegression()</pre></div></div></div></div></div>



evaluate 전에 위와 똑같은 방식으로 test 데이터를 전처리합니다


```python
test_num_imputed = num_imputer.transform(X_test[NUMERICAL])
test_num_scaled = scaler.transform(test_num_imputed)
test_cat_imputed = cat_imputer.transform(X_test[CATEGORICAL])
test_cat_encoded = encoder.transform(test_cat_imputed)
test_preprocessed = np.concatenate((test_num_scaled, test_cat_encoded), axis=1)

print(f"Train ROC-AUC: {calculate_roc_auc(model, train_preprocessed, y_train):.4f}")
print(f"Test ROC-AUC: {calculate_roc_auc(model, test_preprocessed, y_test):.4f}")
```

    Train ROC-AUC: 0.8670
    Test ROC-AUC: 0.8332
    

이번에는 approach는 정확했습니다. 하지만, 좋은 코드를 작성하는 것은 정확한 것에 그치지 않습니다. 위 코드는 각 preprocessing 단계에 대해 training 데이터 세트와 test 데이터 세트 모두에 대한 중간 출력을 저장했지만, preprocessing 단계의 수가 증가하면 이러한 과정은 곧 지겨워져 test 데이터 preprocessing 단계를 놓치는 것과 같은 오류가 발생하기 쉽습니다. 이 코드는 보다 체계적이고 능률적이며 읽기 쉽게 만들 수 있으며, 이는 다음 단계에서 보여줍니다.

# 3. Elegant approach \#1
Scikit-learn의 ```Pipeline``` 및 ```ColumnTransformer```를 사용하여 이전 코드를 간소화해 보겠습니다.


```python
numerical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", MinMaxScaler())
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore", sparse=False))
])

preprocessors = ColumnTransformer(transformers=[
    ("num", numerical_pipe, NUMERICAL),
    ("cat", categorical_pipe, CATEGORICAL)
])

pipe = Pipeline([
    ("proprocessors", preprocessors),
    ("model", LogisticRegression())
])

pipe.fit(X_train, y_train)
```




<style>#sk-890083aa-5440-437c-9054-72ef203db157 {color: black;background-color: white;}#sk-890083aa-5440-437c-9054-72ef203db157 pre{padding: 0;}#sk-890083aa-5440-437c-9054-72ef203db157 div.sk-toggleable {background-color: white;}#sk-890083aa-5440-437c-9054-72ef203db157 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-890083aa-5440-437c-9054-72ef203db157 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-890083aa-5440-437c-9054-72ef203db157 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-890083aa-5440-437c-9054-72ef203db157 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-890083aa-5440-437c-9054-72ef203db157 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-890083aa-5440-437c-9054-72ef203db157 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-890083aa-5440-437c-9054-72ef203db157 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-890083aa-5440-437c-9054-72ef203db157 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-890083aa-5440-437c-9054-72ef203db157 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-890083aa-5440-437c-9054-72ef203db157 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-890083aa-5440-437c-9054-72ef203db157 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-890083aa-5440-437c-9054-72ef203db157 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-890083aa-5440-437c-9054-72ef203db157 div.sk-estimator:hover {background-color: #d4ebff;}#sk-890083aa-5440-437c-9054-72ef203db157 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-890083aa-5440-437c-9054-72ef203db157 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-890083aa-5440-437c-9054-72ef203db157 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 2em;bottom: 0;left: 50%;}#sk-890083aa-5440-437c-9054-72ef203db157 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;}#sk-890083aa-5440-437c-9054-72ef203db157 div.sk-item {z-index: 1;}#sk-890083aa-5440-437c-9054-72ef203db157 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;}#sk-890083aa-5440-437c-9054-72ef203db157 div.sk-parallel::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 2em;bottom: 0;left: 50%;}#sk-890083aa-5440-437c-9054-72ef203db157 div.sk-parallel-item {display: flex;flex-direction: column;position: relative;background-color: white;}#sk-890083aa-5440-437c-9054-72ef203db157 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-890083aa-5440-437c-9054-72ef203db157 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-890083aa-5440-437c-9054-72ef203db157 div.sk-parallel-item:only-child::after {width: 0;}#sk-890083aa-5440-437c-9054-72ef203db157 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;position: relative;}#sk-890083aa-5440-437c-9054-72ef203db157 div.sk-label label {font-family: monospace;font-weight: bold;background-color: white;display: inline-block;line-height: 1.2em;}#sk-890083aa-5440-437c-9054-72ef203db157 div.sk-label-container {position: relative;z-index: 2;text-align: center;}#sk-890083aa-5440-437c-9054-72ef203db157 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-890083aa-5440-437c-9054-72ef203db157 div.sk-text-repr-fallback {display: none;}</style><div id="sk-890083aa-5440-437c-9054-72ef203db157" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Pipeline(steps=[(&#x27;proprocessors&#x27;,
                 ColumnTransformer(transformers=[(&#x27;num&#x27;,
                                                  Pipeline(steps=[(&#x27;imputer&#x27;,
                                                                   SimpleImputer()),
                                                                  (&#x27;scaler&#x27;,
                                                                   MinMaxScaler())]),
                                                  Index([&#x27;pclass&#x27;, &#x27;age&#x27;, &#x27;sibsp&#x27;, &#x27;parch&#x27;, &#x27;fare&#x27;], dtype=&#x27;object&#x27;)),
                                                 (&#x27;cat&#x27;,
                                                  Pipeline(steps=[(&#x27;imputer&#x27;,
                                                                   SimpleImputer(fill_value=&#x27;missing&#x27;,
                                                                                 strategy=&#x27;constant&#x27;)),
                                                                  (&#x27;encoder&#x27;,
                                                                   OneHotEncoder(drop=&#x27;first&#x27;,
                                                                                 handle_unknown=&#x27;ignore&#x27;,
                                                                                 sparse=False))]),
                                                  Index([&#x27;deck&#x27;, &#x27;embark_town&#x27;, &#x27;sex&#x27;], dtype=&#x27;object&#x27;))])),
                (&#x27;model&#x27;, LogisticRegression())])</pre><b>Please rerun this cell to show the HTML repr or trust the notebook.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="b428d1c1-71b4-4258-8306-1a2bc5effea1" type="checkbox" ><label for="b428d1c1-71b4-4258-8306-1a2bc5effea1" class="sk-toggleable__label sk-toggleable__label-arrow">Pipeline</label><div class="sk-toggleable__content"><pre>Pipeline(steps=[(&#x27;proprocessors&#x27;,
                 ColumnTransformer(transformers=[(&#x27;num&#x27;,
                                                  Pipeline(steps=[(&#x27;imputer&#x27;,
                                                                   SimpleImputer()),
                                                                  (&#x27;scaler&#x27;,
                                                                   MinMaxScaler())]),
                                                  Index([&#x27;pclass&#x27;, &#x27;age&#x27;, &#x27;sibsp&#x27;, &#x27;parch&#x27;, &#x27;fare&#x27;], dtype=&#x27;object&#x27;)),
                                                 (&#x27;cat&#x27;,
                                                  Pipeline(steps=[(&#x27;imputer&#x27;,
                                                                   SimpleImputer(fill_value=&#x27;missing&#x27;,
                                                                                 strategy=&#x27;constant&#x27;)),
                                                                  (&#x27;encoder&#x27;,
                                                                   OneHotEncoder(drop=&#x27;first&#x27;,
                                                                                 handle_unknown=&#x27;ignore&#x27;,
                                                                                 sparse=False))]),
                                                  Index([&#x27;deck&#x27;, &#x27;embark_town&#x27;, &#x27;sex&#x27;], dtype=&#x27;object&#x27;))])),
                (&#x27;model&#x27;, LogisticRegression())])</pre></div></div></div><div class="sk-serial"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="abf22c17-bfd2-498c-83e5-3d1122c5447b" type="checkbox" ><label for="abf22c17-bfd2-498c-83e5-3d1122c5447b" class="sk-toggleable__label sk-toggleable__label-arrow">proprocessors: ColumnTransformer</label><div class="sk-toggleable__content"><pre>ColumnTransformer(transformers=[(&#x27;num&#x27;,
                                 Pipeline(steps=[(&#x27;imputer&#x27;, SimpleImputer()),
                                                 (&#x27;scaler&#x27;, MinMaxScaler())]),
                                 Index([&#x27;pclass&#x27;, &#x27;age&#x27;, &#x27;sibsp&#x27;, &#x27;parch&#x27;, &#x27;fare&#x27;], dtype=&#x27;object&#x27;)),
                                (&#x27;cat&#x27;,
                                 Pipeline(steps=[(&#x27;imputer&#x27;,
                                                  SimpleImputer(fill_value=&#x27;missing&#x27;,
                                                                strategy=&#x27;constant&#x27;)),
                                                 (&#x27;encoder&#x27;,
                                                  OneHotEncoder(drop=&#x27;first&#x27;,
                                                                handle_unknown=&#x27;ignore&#x27;,
                                                                sparse=False))]),
                                 Index([&#x27;deck&#x27;, &#x27;embark_town&#x27;, &#x27;sex&#x27;], dtype=&#x27;object&#x27;))])</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="a382fd62-c7b4-4f02-a3cd-98706efb2d4f" type="checkbox" ><label for="a382fd62-c7b4-4f02-a3cd-98706efb2d4f" class="sk-toggleable__label sk-toggleable__label-arrow">num</label><div class="sk-toggleable__content"><pre>Index([&#x27;pclass&#x27;, &#x27;age&#x27;, &#x27;sibsp&#x27;, &#x27;parch&#x27;, &#x27;fare&#x27;], dtype=&#x27;object&#x27;)</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="ee184967-a0e6-43b4-a6b9-85587c201f73" type="checkbox" ><label for="ee184967-a0e6-43b4-a6b9-85587c201f73" class="sk-toggleable__label sk-toggleable__label-arrow">SimpleImputer</label><div class="sk-toggleable__content"><pre>SimpleImputer()</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="ae4592ea-1478-4f64-ab66-921ba501d319" type="checkbox" ><label for="ae4592ea-1478-4f64-ab66-921ba501d319" class="sk-toggleable__label sk-toggleable__label-arrow">MinMaxScaler</label><div class="sk-toggleable__content"><pre>MinMaxScaler()</pre></div></div></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="7c748821-23f4-4c42-9348-2b992ebf2144" type="checkbox" ><label for="7c748821-23f4-4c42-9348-2b992ebf2144" class="sk-toggleable__label sk-toggleable__label-arrow">cat</label><div class="sk-toggleable__content"><pre>Index([&#x27;deck&#x27;, &#x27;embark_town&#x27;, &#x27;sex&#x27;], dtype=&#x27;object&#x27;)</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="a7b5ec2d-3a25-4545-a2be-c3a0d8ebd425" type="checkbox" ><label for="a7b5ec2d-3a25-4545-a2be-c3a0d8ebd425" class="sk-toggleable__label sk-toggleable__label-arrow">SimpleImputer</label><div class="sk-toggleable__content"><pre>SimpleImputer(fill_value=&#x27;missing&#x27;, strategy=&#x27;constant&#x27;)</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="44bc2614-2c38-4b9e-84cc-b74e19a2e42a" type="checkbox" ><label for="44bc2614-2c38-4b9e-84cc-b74e19a2e42a" class="sk-toggleable__label sk-toggleable__label-arrow">OneHotEncoder</label><div class="sk-toggleable__content"><pre>OneHotEncoder(drop=&#x27;first&#x27;, handle_unknown=&#x27;ignore&#x27;, sparse=False)</pre></div></div></div></div></div></div></div></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="0b30e3e7-deb1-4b69-9027-1280c5e3c037" type="checkbox" ><label for="0b30e3e7-deb1-4b69-9027-1280c5e3c037" class="sk-toggleable__label sk-toggleable__label-arrow">LogisticRegression</label><div class="sk-toggleable__content"><pre>LogisticRegression()</pre></div></div></div></div></div></div></div>



pipeline은:

- numerical과 categorical로 input 데이터를 나누고

- 동시에 두 그룹을 preprocessing 합니다

- 그 후, 두개의 전처리된 그룹을 합치고

- 전처리된 데이터를 모델에 전달합니다


raw 데이터가 훈련된 pipeline에 전달되면 preprocessing하고 예측합니다. 즉, 더 이상 training 및 test 데이터 세트에 대한 중간 결과를 저장할 필요가 없습니다. 보이지 않는 데이터에 점수를 매기는 것은 ```pipe.predict()```으로 매우 간단합니다. 이제 모델의 성능을 평가해 보겠습니다.


```python
print(f"Train ROC-AUC: {calculate_roc_auc(pipe, X_train, y_train):.4f}")
print(f"Test ROC-AUC: {calculate_roc_auc(pipe, X_test, y_test):.4f}")
```

    Train ROC-AUC: 0.8670
    Test ROC-AUC: 0.8332
    

transformation이 정확히 동일하지만, 더 깔끔한 방식으로 코드가 작성되었기 때문에 이전 approach의 성능과 일치하는 것을 확인하는 것이 좋습니다. 여기서의 예에서는 이 아티클에 표시된 네 가지 approach 중 가장 좋은 접근 방식입니다.

```OneHotEncoder```나 ```SimpleImputer```와 같은 Scikit-learn의 즉시 사용 가능한 transformer는 빠르고 효율적입니다. 하지만, 이러한 미리 만들어진 transformer가 항상 우리가 원하는 preprocessing 요구 사항을 만족하는 것은 아닙니다. 이러한 경우, 다음 approach에 익숙해지면 맞춤형 preprocessing 방법을 더 잘 control할 수 있습니다.

# 4. Elegant approach \#2
이 approach에서는 Scikit-learn을 사용하여 custom transformer를 만듭니다. 우리가 친숙했던 preprocessing 단계가 custom transformer로 어떻게 바뀌는지 보는 것이 이해하는 데 도움이 되기를 바랍니다. custom transformer의 사용 예시에 관심이 있다면 이 [GitHub repository](https://github.com/zluvsand/ml_pipeline)를 확인하세요.


```python
# Transformer
# fit(), transform() 메소드가 있으며, 데이터를 원하는 포맷으로 바꿔주는데 도움을 줍니다.
# OneHotEncoder, MinMaxScaler가 예

# Estimator
# fit(), predict() 메소드가 있으며, 여기서는 모델과 같은 의미로 사용됩니다.
class Imputer(BaseEstimator, TransformerMixin):
    def __init__(self, features, method="constant", value="missing"):
        self.features = features
        self.method = method
        self.value = value
        
    def fit(self, X, y=None):
        if self.method == "mean":
            self.value = X[self.features].mean()
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        X_transformed[self.features] = X[self.features].fillna(self.value)
        return X_transformed
    
    
class Scaler(BaseEstimator, TransformerMixin):
    def __init__(self, features):
        self.features = features
    
    def fit(self, X, y=None):
        self.min = X[self.features].min()
        self.range = X[self.features].max() - self.min
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        X_transformed[self.features] = (X[self.features] - self.min) / self.range
        return X_transformed
    
    
class Encoder(BaseEstimator, TransformerMixin):
    def __init__(self, features, drop="first"):
        self.features = features
        self.drop = drop
        
    def fit(self, X, y=None):
        self.encoder = OneHotEncoder(sparse=False, drop=self.drop)
        self.encoder.fit(X[self.features])
        return self
    
    def transform(self, X):
        X_transformed = pd.concat([X.drop(columns=self.features).reset_index(drop=True),
                                  pd.DataFrame(self.encoder.transform(X[self.features]))],
                                  axis=1)
        return X_transformed
    
    
pipe = Pipeline([
    ("num_imputer", Imputer(NUMERICAL, method="mean")),
    ("scaler", Scaler(NUMERICAL)),
    ("cat_imputer", Imputer(CATEGORICAL)),
    ("encoder", Encoder(CATEGORICAL)),
    ("model", LogisticRegression())
])

pipe.fit(X_train, y_train)
```




<style>#sk-cc2c73f9-ddbc-493f-a863-60b26fc6d3de {color: black;background-color: white;}#sk-cc2c73f9-ddbc-493f-a863-60b26fc6d3de pre{padding: 0;}#sk-cc2c73f9-ddbc-493f-a863-60b26fc6d3de div.sk-toggleable {background-color: white;}#sk-cc2c73f9-ddbc-493f-a863-60b26fc6d3de label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-cc2c73f9-ddbc-493f-a863-60b26fc6d3de label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-cc2c73f9-ddbc-493f-a863-60b26fc6d3de label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-cc2c73f9-ddbc-493f-a863-60b26fc6d3de div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-cc2c73f9-ddbc-493f-a863-60b26fc6d3de div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-cc2c73f9-ddbc-493f-a863-60b26fc6d3de div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-cc2c73f9-ddbc-493f-a863-60b26fc6d3de input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-cc2c73f9-ddbc-493f-a863-60b26fc6d3de input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-cc2c73f9-ddbc-493f-a863-60b26fc6d3de div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-cc2c73f9-ddbc-493f-a863-60b26fc6d3de div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-cc2c73f9-ddbc-493f-a863-60b26fc6d3de input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-cc2c73f9-ddbc-493f-a863-60b26fc6d3de div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-cc2c73f9-ddbc-493f-a863-60b26fc6d3de div.sk-estimator:hover {background-color: #d4ebff;}#sk-cc2c73f9-ddbc-493f-a863-60b26fc6d3de div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-cc2c73f9-ddbc-493f-a863-60b26fc6d3de div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-cc2c73f9-ddbc-493f-a863-60b26fc6d3de div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 2em;bottom: 0;left: 50%;}#sk-cc2c73f9-ddbc-493f-a863-60b26fc6d3de div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;}#sk-cc2c73f9-ddbc-493f-a863-60b26fc6d3de div.sk-item {z-index: 1;}#sk-cc2c73f9-ddbc-493f-a863-60b26fc6d3de div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;}#sk-cc2c73f9-ddbc-493f-a863-60b26fc6d3de div.sk-parallel::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 2em;bottom: 0;left: 50%;}#sk-cc2c73f9-ddbc-493f-a863-60b26fc6d3de div.sk-parallel-item {display: flex;flex-direction: column;position: relative;background-color: white;}#sk-cc2c73f9-ddbc-493f-a863-60b26fc6d3de div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-cc2c73f9-ddbc-493f-a863-60b26fc6d3de div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-cc2c73f9-ddbc-493f-a863-60b26fc6d3de div.sk-parallel-item:only-child::after {width: 0;}#sk-cc2c73f9-ddbc-493f-a863-60b26fc6d3de div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;position: relative;}#sk-cc2c73f9-ddbc-493f-a863-60b26fc6d3de div.sk-label label {font-family: monospace;font-weight: bold;background-color: white;display: inline-block;line-height: 1.2em;}#sk-cc2c73f9-ddbc-493f-a863-60b26fc6d3de div.sk-label-container {position: relative;z-index: 2;text-align: center;}#sk-cc2c73f9-ddbc-493f-a863-60b26fc6d3de div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-cc2c73f9-ddbc-493f-a863-60b26fc6d3de div.sk-text-repr-fallback {display: none;}</style><div id="sk-cc2c73f9-ddbc-493f-a863-60b26fc6d3de" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Pipeline(steps=[(&#x27;num_imputer&#x27;,
                 Imputer(features=Index([&#x27;pclass&#x27;, &#x27;age&#x27;, &#x27;sibsp&#x27;, &#x27;parch&#x27;, &#x27;fare&#x27;], dtype=&#x27;object&#x27;),
                         method=&#x27;mean&#x27;,
                         value=pclass     2.308989
age       29.807687
sibsp      0.492978
parch      0.390449
fare      31.819826
dtype: float64)),
                (&#x27;scaler&#x27;,
                 Scaler(features=Index([&#x27;pclass&#x27;, &#x27;age&#x27;, &#x27;sibsp&#x27;, &#x27;parch&#x27;, &#x27;fare&#x27;], dtype=&#x27;object&#x27;))),
                (&#x27;cat_imputer&#x27;,
                 Imputer(features=Index([&#x27;deck&#x27;, &#x27;embark_town&#x27;, &#x27;sex&#x27;], dtype=&#x27;object&#x27;))),
                (&#x27;encoder&#x27;,
                 Encoder(features=Index([&#x27;deck&#x27;, &#x27;embark_town&#x27;, &#x27;sex&#x27;], dtype=&#x27;object&#x27;))),
                (&#x27;model&#x27;, LogisticRegression())])</pre><b>Please rerun this cell to show the HTML repr or trust the notebook.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="f0582704-4187-4606-9399-962dc4c3b7fb" type="checkbox" ><label for="f0582704-4187-4606-9399-962dc4c3b7fb" class="sk-toggleable__label sk-toggleable__label-arrow">Pipeline</label><div class="sk-toggleable__content"><pre>Pipeline(steps=[(&#x27;num_imputer&#x27;,
                 Imputer(features=Index([&#x27;pclass&#x27;, &#x27;age&#x27;, &#x27;sibsp&#x27;, &#x27;parch&#x27;, &#x27;fare&#x27;], dtype=&#x27;object&#x27;),
                         method=&#x27;mean&#x27;,
                         value=pclass     2.308989
age       29.807687
sibsp      0.492978
parch      0.390449
fare      31.819826
dtype: float64)),
                (&#x27;scaler&#x27;,
                 Scaler(features=Index([&#x27;pclass&#x27;, &#x27;age&#x27;, &#x27;sibsp&#x27;, &#x27;parch&#x27;, &#x27;fare&#x27;], dtype=&#x27;object&#x27;))),
                (&#x27;cat_imputer&#x27;,
                 Imputer(features=Index([&#x27;deck&#x27;, &#x27;embark_town&#x27;, &#x27;sex&#x27;], dtype=&#x27;object&#x27;))),
                (&#x27;encoder&#x27;,
                 Encoder(features=Index([&#x27;deck&#x27;, &#x27;embark_town&#x27;, &#x27;sex&#x27;], dtype=&#x27;object&#x27;))),
                (&#x27;model&#x27;, LogisticRegression())])</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="eed9f0b1-99ff-4ff3-9143-9d4f8197351f" type="checkbox" ><label for="eed9f0b1-99ff-4ff3-9143-9d4f8197351f" class="sk-toggleable__label sk-toggleable__label-arrow">Imputer</label><div class="sk-toggleable__content"><pre>Imputer(features=Index([&#x27;pclass&#x27;, &#x27;age&#x27;, &#x27;sibsp&#x27;, &#x27;parch&#x27;, &#x27;fare&#x27;], dtype=&#x27;object&#x27;),
        method=&#x27;mean&#x27;,
        value=pclass     2.308989
age       29.807687
sibsp      0.492978
parch      0.390449
fare      31.819826
dtype: float64)</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="6a5d0cba-5486-48d7-a15f-829354aaa752" type="checkbox" ><label for="6a5d0cba-5486-48d7-a15f-829354aaa752" class="sk-toggleable__label sk-toggleable__label-arrow">Scaler</label><div class="sk-toggleable__content"><pre>Scaler(features=Index([&#x27;pclass&#x27;, &#x27;age&#x27;, &#x27;sibsp&#x27;, &#x27;parch&#x27;, &#x27;fare&#x27;], dtype=&#x27;object&#x27;))</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="a2d1f14c-a1b6-4511-8e43-64aa805a21b6" type="checkbox" ><label for="a2d1f14c-a1b6-4511-8e43-64aa805a21b6" class="sk-toggleable__label sk-toggleable__label-arrow">Imputer</label><div class="sk-toggleable__content"><pre>Imputer(features=Index([&#x27;deck&#x27;, &#x27;embark_town&#x27;, &#x27;sex&#x27;], dtype=&#x27;object&#x27;))</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="480ded4d-cbac-4f55-950e-9f0b34d62760" type="checkbox" ><label for="480ded4d-cbac-4f55-950e-9f0b34d62760" class="sk-toggleable__label sk-toggleable__label-arrow">Encoder</label><div class="sk-toggleable__content"><pre>Encoder(features=Index([&#x27;deck&#x27;, &#x27;embark_town&#x27;, &#x27;sex&#x27;], dtype=&#x27;object&#x27;))</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="0eb72c8b-961f-4c3a-b783-4966c380a748" type="checkbox" ><label for="0eb72c8b-961f-4c3a-b783-4966c380a748" class="sk-toggleable__label sk-toggleable__label-arrow">LogisticRegression</label><div class="sk-toggleable__content"><pre>LogisticRegression()</pre></div></div></div></div></div></div></div>



이전과 달리 각 단계는 output을 input으로 차례대로 전달합니다. 이제 모델을 평가할 시간입니다:


```python
print(f"Train ROC-AUC: {calculate_roc_auc(pipe, X_train, y_train):.4f}")
print(f"Test ROC-AUC: {calculate_roc_auc(pipe, X_test, y_test):.4f}")
```

    Train ROC-AUC: 0.8670
    Test ROC-AUC: 0.8332
    

이번 포스팅은 여기까지였습니다. 후자의 2가지 접근 방식을 사용할 때, 한 가지 장점은 모델에서만이 아니라 전체 pipeline에서 hyperparameter를 수행할 수 있다는 것입니다. ML pipeline 사용을 시작하는 실용적인 방법을 배웠기를 바랍니다.
