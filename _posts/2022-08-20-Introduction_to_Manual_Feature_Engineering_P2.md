---
layout: single
title:  "[Home Credit Default Risk] Introduction: Manual Feature Engineering (part two)"
categories: Kaggle
tag: [Home Credit Default Risk]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: false
---
**[공지사항]** ["출처: https://www.kaggle.com/code/jundthird/kor-manual-feature-engineering-pt2"](https://www.kaggle.com/code/jundthird/kor-manual-feature-engineering-pt2)
{: .notice--danger}


# Introduction: Manual Feature Engineering (part two)

이 노트북에서는 [Introduction to Manual Feature Engineering](https://www.kaggle.com/code/jundthird/introduction-manual-feature-engineering)에서 한단계 더 확장해 나갈 것입니다. 여기서는 ```previous_application```, ```POS_CASH_balance```, ```installments_payments``` 및 ```credit_card_balance``` 데이터 파일의 정보를 통합하기 위해 해당 노트북에서 만든 aggregation 및 value 계산 함수를 사용할 것입니다. 이미 이전 노트북에서 ```Bureau``` 및 ```Bureau_balance```의 정보를 사용했으며 ```application``` 데이터만 사용하는 것에 비해 competition 점수를 향상시킬 수 있었습니다. 여기에 포함된 features로 모델을 실행하면 성능이 향상되지만 feature의 수가 폭발적으로 늘어나는 문제에 봉착하게 됩니다. feature selection 노트북을 작성 중이지만, 이 노트북을 위해 우리 모델에 대한 많은 데이터 세트를 계속 구축해 나갈 것입니다.

<br/>

4개의 추가 데이터에 대한 정의는:

- previous_application(called ```previous```): application 데이터에 대출이 있는 고객의 Home Credit 대출에 대한 previous applications. application 데이터의 각 현재 대출에는 여러 개의 previous 대출이 있을 수 있습니다. 각 previous application에는 하나의 행이 있으며 ```SK_ID_PREV``` feature로 식별됩니다.

<br/>

- POS_CASH_BALANCE(called ```cash```): 고객이 Home Credit을 통해 보유한 previous 판매 시점 또는 현금 대출에 대한 월별 데이터입니다. 각 행은 previous 판매 시점 또는 현금 대출의 한 달치이며 하나의 previous 대출에는 여러 행이 있을 수 있습니다.

<br/>

- credit_card_balance(called ```credit```): 고객이 Home Credit을 사용하여 가지고 있던 previous 신용 카드에 대한 월별 데이터입니다. 각 행은 신용 카드 잔액의 한 달치이며 단일 신용 카드에는 여러 행이 있을 수 있습니다.

<br/>

- installments_payment(called ```installments```): Home Credit에서 previous 대출에 대한 지불 내역. 모든  payment에 대해 하나의 행이 있고 모든 missed payment에 대해 하나의 행이 있습니다.

# Functions

이전 노트북에서는 두가지 함수를 만들었습니다:

- ```agg_numeric```: numeric 변수에 대한 ```mean```, ```count```, ```max```, ```min```를 계산합니다.
- ```agg_categorical```: 카테고리형 변수에서 각 카테고리에 대한 counts와 normalized counts를 계산합니다.

이 두 함수는 함께 데이터 프레임의 숫자 및 카테고리형 데이터에 대한 정보를 추출할 수 있습니다. 여기서의 일반적인 접근 방식은 클라이언트 ID ```SK_ID_CURR```로 그룹화하여 이 두 함수를 모두 데이터 프레임에 적용하는 것입니다. ```POS_CASH_balance```, ```credit_card_balance``` 및 ```installment_payments```의 경우 먼저 previous 대출의 고유 ID인 ```SK_ID_PREV```로 그룹화할 수 있습니다. 그런 다음 결과로 나온 데이터 프레임을 ```SK_ID_CURR```로 그룹화하여 모든 previous 대출에 대한 각 클라이언트의 통계치를 계산합니다.


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
plt.style.use("fivethirtyeight")
import gc
```

## Function to Aggregate Numeric Data
```group_var``` 별로 데이터를 그룹화하고 ```mean```, ```max```, ```min``` 및 ```sum```을 계산합니다.


```python
def agg_numeric(df, parent_var, df_name):
    
    # id variables 제거
    for col in df:
        if col != parent_var and "SK_ID" in col:
            df = df.drop(columns=col)
            
    parent_ids = df[parent_var].copy()
    numeric_df = df.select_dtypes("number").copy()
    numeric_df[parent_var] = parent_ids
    
    agg = numeric_df.groupby(parent_var).agg(['count', 'mean', 'max', 'min', 'sum'])
    
    columns = []
    
    for var in agg.columns.levels[0]:
        if var != parent_var:
            for stat in agg.columns.levels[1]:
                columns.append("%s_%s_%s" % (df_name, var, stat))
                
    agg.columns = columns
    
    # 불필요한 중복된 변수 제거
    _, idx = np.unique(agg, axis=1, return_index=True # 각각 처음으로 등장한 인덱스 번호
                       )
    agg = agg.iloc[:, idx]
    
    return agg
```

## Function to Calculate Categorical Counts
이 함수는 각 클라이언트에 대한 카테고리형 변수에서 각 카테고리의 등장 횟수(개수)을 계산합니다. 또한 카테고리형 변수의 모든 카테고리에 대한 총 개수로 나눈 카테고리의 개수인 normed count를 계산합니다.


```python
def agg_categorical(df, parent_var, df_name):
    categorical = pd.get_dummies(df.select_dtypes("category"))
    categorical[parent_var] = df[parent_var]
    categorical = categorical.groupby(parent_var).agg(["sum", "count", "mean"])
    
    column_names = []
    
    for var in categorical.columns.levels[0]:
        for stat in ["sum", "count", "mean"]:
            column_names.append("%s_%s_%s" % (df_name, var, stat))
            
    categorical.columns = column_names
    
    _, idx = np.unique(categorical, axis=1, return_index=True)
    categorical = categorical.iloc[:, idx]
    
    return categorical
```

## Function for KDE Plots of Variable
```TARGET``` 값에 따라 다르게 색깔을 입힌 변수의 분포를 그리는 함수를 만들었습니다(1은 대출 상환 x, 0은 대출 상환). 이 함수를 사용하여 생성한 새 변수를 시각적으로 확인할 수 있습니다. 이는 생성된 변수가 유용할지 여부에 대한 근사치로 사용할 수 있는 target과 변수의 상관 계수를 계산합니다.


```python
def kde_target(var_name, df):
    corr = df["TARGET"].corr(df[var_name])
    avg_repaid = df.loc[df["TARGET"] == 0, var_name].median()
    avg_not_repaid = df.loc[df["TARGET"] == 1, var_name].median()
    plt.figure(figsize=(12, 6))
    
    sns.kdeplot(df.loc[df["TARGET"] == 0, var_name], label="TARGET == 0")
    sns.kdeplot(df.loc[df["TARGET"] == 1, var_name], label="TARGET == 1")
    
    plt.xlabel(var_name); plt.ylabel('Density'); plt.title('%s Distribution' % var_name)
    plt.legend(loc="best")
    
    print('The correlation between %s and the TARGET is %0.4f' % (var_name, corr))
    print('Median value for loan that was not repaid = %0.4f' % avg_not_repaid)
    print('Median value for loan that was repaid = %0.4f' % avg_repaid)
```

# Function to Convert Data Types
변수에 보다 효율적인 type을 사용하여 메모리 사용량을 줄이는 데 도움이 됩니다. 예를 들어, ```category```는 보통 ```object```보다 더 나은 type입니다(unique 값의 개수가 데이터 프레임의 행 개수에 가까운 값이 아닌 경우)


```python
import sys

def return_size(df):
    """Return size of dataframe in gigabytes"""
    return round(sys.getsizeof(df) / 1e9, 2)

def convert_types(df, print_info=False):
    original_memory = df.memory_usage().sum()
    
    for c in df:
        if ("SK_ID" in c):
            df[c] = df[c].fillna(0).astype(np.int32)
        
        elif (df[c].dtype == "object") and (df[c].nunique() < df.shape[0]):
            df[c] = df[c].astype("category")
            
        elif set(df[c].unique()) == set([1, 0]):
            df[c] = df[c].astype(bool)
        
        elif df[c].dtype == float:
            df[c] = df[c].astype(np.float32)
            
        # Int64 to int32
        elif df[c].dtype == int:
            df[c] = df[c].astype(np.int32)
            
    new_memory = df.memory_usage().sum()
    
    if print_info:
        print(f'Original Memory Usage: {round(original_memory / 1e9, 2)} gb.')
        print(f'New Memory Usage: {round(new_memory / 1e9, 2)} gb.')
    
    return df
```

한 번에 하나의 데이터 프레임을 처리합시다. 첫 번째는 ```previous_applications```입니다. 여기에는 고객이 Home Credit에서 가지고 있었던 모든 previous 대출에 대한 하나의 행이 있습니다. 고객은 여러 개의 previous 대출을 보유할 수 있으므로 각 고객에 대한 통계를 만들어야 합니다.

**previous_application**


```python
previous = pd.read_csv('./home_credit/previous_application.csv')
previous = convert_types(previous, print_info=True)
previous.head()
```

    Original Memory Usage: 0.49 gb.
    New Memory Usage: 0.18 gb.
    




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
      <th>SK_ID_PREV</th>
      <th>SK_ID_CURR</th>
      <th>NAME_CONTRACT_TYPE</th>
      <th>AMT_ANNUITY</th>
      <th>AMT_APPLICATION</th>
      <th>AMT_CREDIT</th>
      <th>AMT_DOWN_PAYMENT</th>
      <th>AMT_GOODS_PRICE</th>
      <th>WEEKDAY_APPR_PROCESS_START</th>
      <th>HOUR_APPR_PROCESS_START</th>
      <th>...</th>
      <th>NAME_SELLER_INDUSTRY</th>
      <th>CNT_PAYMENT</th>
      <th>NAME_YIELD_GROUP</th>
      <th>PRODUCT_COMBINATION</th>
      <th>DAYS_FIRST_DRAWING</th>
      <th>DAYS_FIRST_DUE</th>
      <th>DAYS_LAST_DUE_1ST_VERSION</th>
      <th>DAYS_LAST_DUE</th>
      <th>DAYS_TERMINATION</th>
      <th>NFLAG_INSURED_ON_APPROVAL</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2030495</td>
      <td>271877</td>
      <td>Consumer loans</td>
      <td>1730.430054</td>
      <td>17145.0</td>
      <td>17145.0</td>
      <td>0.0</td>
      <td>17145.0</td>
      <td>SATURDAY</td>
      <td>15</td>
      <td>...</td>
      <td>Connectivity</td>
      <td>12.0</td>
      <td>middle</td>
      <td>POS mobile with interest</td>
      <td>365243.0</td>
      <td>-42.0</td>
      <td>300.0</td>
      <td>-42.0</td>
      <td>-37.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2802425</td>
      <td>108129</td>
      <td>Cash loans</td>
      <td>25188.615234</td>
      <td>607500.0</td>
      <td>679671.0</td>
      <td>NaN</td>
      <td>607500.0</td>
      <td>THURSDAY</td>
      <td>11</td>
      <td>...</td>
      <td>XNA</td>
      <td>36.0</td>
      <td>low_action</td>
      <td>Cash X-Sell: low</td>
      <td>365243.0</td>
      <td>-134.0</td>
      <td>916.0</td>
      <td>365243.0</td>
      <td>365243.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2523466</td>
      <td>122040</td>
      <td>Cash loans</td>
      <td>15060.735352</td>
      <td>112500.0</td>
      <td>136444.5</td>
      <td>NaN</td>
      <td>112500.0</td>
      <td>TUESDAY</td>
      <td>11</td>
      <td>...</td>
      <td>XNA</td>
      <td>12.0</td>
      <td>high</td>
      <td>Cash X-Sell: high</td>
      <td>365243.0</td>
      <td>-271.0</td>
      <td>59.0</td>
      <td>365243.0</td>
      <td>365243.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2819243</td>
      <td>176158</td>
      <td>Cash loans</td>
      <td>47041.335938</td>
      <td>450000.0</td>
      <td>470790.0</td>
      <td>NaN</td>
      <td>450000.0</td>
      <td>MONDAY</td>
      <td>7</td>
      <td>...</td>
      <td>XNA</td>
      <td>12.0</td>
      <td>middle</td>
      <td>Cash X-Sell: middle</td>
      <td>365243.0</td>
      <td>-482.0</td>
      <td>-152.0</td>
      <td>-182.0</td>
      <td>-177.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1784265</td>
      <td>202054</td>
      <td>Cash loans</td>
      <td>31924.394531</td>
      <td>337500.0</td>
      <td>404055.0</td>
      <td>NaN</td>
      <td>337500.0</td>
      <td>THURSDAY</td>
      <td>9</td>
      <td>...</td>
      <td>XNA</td>
      <td>24.0</td>
      <td>high</td>
      <td>Cash Street: high</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 37 columns</p>
</div>




```python
previous_agg = agg_numeric(previous, 'SK_ID_CURR', 'previous')
print('Previous aggregation shape: ', previous_agg.shape)
previous_agg.head()
```

    Previous aggregation shape:  (338857, 80)
    




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
      <th>previous_DAYS_DECISION_sum</th>
      <th>previous_DAYS_DECISION_min</th>
      <th>previous_DAYS_DECISION_mean</th>
      <th>previous_DAYS_DECISION_max</th>
      <th>previous_DAYS_FIRST_DUE_sum</th>
      <th>previous_DAYS_FIRST_DUE_min</th>
      <th>previous_DAYS_FIRST_DUE_mean</th>
      <th>previous_DAYS_FIRST_DUE_max</th>
      <th>previous_DAYS_LAST_DUE_sum</th>
      <th>previous_DAYS_LAST_DUE_min</th>
      <th>...</th>
      <th>previous_DAYS_FIRST_DRAWING_min</th>
      <th>previous_DAYS_FIRST_DRAWING_mean</th>
      <th>previous_DAYS_FIRST_DRAWING_max</th>
      <th>previous_DAYS_FIRST_DRAWING_sum</th>
      <th>previous_RATE_INTEREST_PRIMARY_min</th>
      <th>previous_RATE_INTEREST_PRIMARY_mean</th>
      <th>previous_RATE_INTEREST_PRIMARY_max</th>
      <th>previous_RATE_INTEREST_PRIVILEGED_min</th>
      <th>previous_RATE_INTEREST_PRIVILEGED_mean</th>
      <th>previous_RATE_INTEREST_PRIVILEGED_max</th>
    </tr>
    <tr>
      <th>SK_ID_CURR</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>100001</th>
      <td>-1740</td>
      <td>-1740</td>
      <td>-1740.0</td>
      <td>-1740</td>
      <td>-1709.0</td>
      <td>-1709.0</td>
      <td>-1709.000000</td>
      <td>-1709.0</td>
      <td>-1619.0</td>
      <td>-1619.0</td>
      <td>...</td>
      <td>365243.0</td>
      <td>365243.0</td>
      <td>365243.0</td>
      <td>365243.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>100002</th>
      <td>-606</td>
      <td>-606</td>
      <td>-606.0</td>
      <td>-606</td>
      <td>-565.0</td>
      <td>-565.0</td>
      <td>-565.000000</td>
      <td>-565.0</td>
      <td>-25.0</td>
      <td>-25.0</td>
      <td>...</td>
      <td>365243.0</td>
      <td>365243.0</td>
      <td>365243.0</td>
      <td>365243.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>100003</th>
      <td>-3915</td>
      <td>-2341</td>
      <td>-1305.0</td>
      <td>-746</td>
      <td>-3823.0</td>
      <td>-2310.0</td>
      <td>-1274.333374</td>
      <td>-716.0</td>
      <td>-3163.0</td>
      <td>-1980.0</td>
      <td>...</td>
      <td>365243.0</td>
      <td>365243.0</td>
      <td>365243.0</td>
      <td>1095729.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>100004</th>
      <td>-815</td>
      <td>-815</td>
      <td>-815.0</td>
      <td>-815</td>
      <td>-784.0</td>
      <td>-784.0</td>
      <td>-784.000000</td>
      <td>-784.0</td>
      <td>-724.0</td>
      <td>-724.0</td>
      <td>...</td>
      <td>365243.0</td>
      <td>365243.0</td>
      <td>365243.0</td>
      <td>365243.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>100005</th>
      <td>-1072</td>
      <td>-757</td>
      <td>-536.0</td>
      <td>-315</td>
      <td>-706.0</td>
      <td>-706.0</td>
      <td>-706.000000</td>
      <td>-706.0</td>
      <td>-466.0</td>
      <td>-466.0</td>
      <td>...</td>
      <td>365243.0</td>
      <td>365243.0</td>
      <td>365243.0</td>
      <td>365243.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 80 columns</p>
</div>




```python
previous_counts = agg_categorical(previous, 'SK_ID_CURR', 'previous')
print('Previous counts shape: ', previous_counts.shape)
previous_counts.head()
```

    Previous counts shape:  (338857, 285)
    




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
      <th>previous_NAME_GOODS_CATEGORY_Animals_mean</th>
      <th>previous_NAME_GOODS_CATEGORY_Animals_sum</th>
      <th>previous_NAME_GOODS_CATEGORY_House Construction_mean</th>
      <th>previous_NAME_GOODS_CATEGORY_House Construction_sum</th>
      <th>previous_NAME_CASH_LOAN_PURPOSE_Refusal to name the goal_mean</th>
      <th>previous_NAME_CASH_LOAN_PURPOSE_Refusal to name the goal_sum</th>
      <th>previous_NAME_CASH_LOAN_PURPOSE_Money for a third person_mean</th>
      <th>previous_NAME_CASH_LOAN_PURPOSE_Money for a third person_sum</th>
      <th>previous_NAME_CASH_LOAN_PURPOSE_Hobby_mean</th>
      <th>previous_NAME_CASH_LOAN_PURPOSE_Hobby_sum</th>
      <th>...</th>
      <th>previous_CODE_REJECT_REASON_XAP_mean</th>
      <th>previous_FLAG_LAST_APPL_PER_CONTRACT_Y_mean</th>
      <th>previous_NAME_PORTFOLIO_POS_sum</th>
      <th>previous_NAME_CONTRACT_TYPE_Consumer loans_sum</th>
      <th>previous_NAME_CASH_LOAN_PURPOSE_XAP_sum</th>
      <th>previous_NAME_PRODUCT_TYPE_XNA_sum</th>
      <th>previous_NAME_CONTRACT_STATUS_Approved_sum</th>
      <th>previous_CODE_REJECT_REASON_XAP_sum</th>
      <th>previous_FLAG_LAST_APPL_PER_CONTRACT_Y_sum</th>
      <th>previous_NAME_CONTRACT_TYPE_Cash loans_count</th>
    </tr>
    <tr>
      <th>SK_ID_CURR</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>100001</th>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>100002</th>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>100003</th>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>100004</th>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>100005</th>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 285 columns</p>
</div>



```merge```를 사용하여 계산된 데이터 프레임을 기본 training 데이터 프레임에 합칩니다. 그런 다음 kernel memory를 너무 많이 사용하지 않도록 계산된 데이터 프레임을 삭제해야 합니다.


```python
train = pd.read_csv('./home_credit/application_train.csv')
train = convert_types(train)
test = pd.read_csv('./home_credit/application_test.csv')
test = convert_types(test)

train = train.merge(previous_counts, on="SK_ID_CURR", how="left")
train = train.merge(previous_agg, on="SK_ID_CURR", how="left")

test = test.merge(previous_counts, on="SK_ID_CURR", how="left")
test = test.merge(previous_agg, on="SK_ID_CURR", how="left")

# 메모리를 위해 변수 제거
gc.enable()
del previous, previous_agg, previous_counts
gc.collect()
```




    0



너무 많은 feature을 계산하는 것에 주의해야 합니다. 관련 없는 feature이 너무 많거나 null 값이 너무 많은 feature이 있는 모델을 과하게 만들고 싶지 않습니다. 이전 노트북에서는 null 값이 75% 이상인 모든 feature을 제거했습니다. 일관성을 위해 동일한 logic을 적용합니다.

## Function to Calculate Missing Values


```python
def missing_values_table(df, print_info=False):
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    
    mis_val_table_ren_columns = mis_val_table.rename(columns={0: "Missing Values",
                                                              1: "% of Total Values"})
    
    mis_val_table_ren_columns = mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:, 1]].\
                                sort_values("% of Total Values", asencding=False).round(1)
    
    if print_info:
            print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
                "There are " + str(mis_val_table_ren_columns.shape[0]) +
                  " columns that have missing values.")
        
    return mis_val_table_ren_columns
```


```python
def remove_missing_columns(train, test, threshold=90):
    train_miss = pd.DataFrame(train.isnull().sum())
    train_miss["percent"] = 100 * train_miss[0] / len(train)
    
    test_miss = pd.DataFrame(test.isnull().sum())
    test_miss["percent"] = 100 * test_miss[0] / len(test)
    
    missing_train_columns = list(train_miss.index[train_miss["percent"] > threshold])
    missing_test_columns = list(test_miss.index[test_miss["percent"] > threshold])
    
    missing_columns = list(set(missing_train_columns + missing_test_columns))
    
    print('There are %d columns with greater than %d%% missing values.' % (len(missing_columns), threshold))
    
    train = train.drop(columns=missing_columns)
    test = test.drop(columns=missing_columns)
    
    return train, test

# def remove_missing_columns(train, test, threshold = 90):
#     # Calculate missing stats for train and test (remember to calculate a percent!)
#     train_miss = pd.DataFrame(train.isnull().sum())
#     train_miss['percent'] = 100 * train_miss[0] / len(train)
    
#     test_miss = pd.DataFrame(test.isnull().sum())
#     test_miss['percent'] = 100 * test_miss[0] / len(test)
    
#     # list of missing columns for train and test
#     missing_train_columns = list(train_miss.index[train_miss['percent'] > threshold])
#     missing_test_columns = list(test_miss.index[test_miss['percent'] > threshold])
    
#     # Combine the two lists together
#     missing_columns = list(set(missing_train_columns + missing_test_columns))
    
#     # Print information
#     print('There are %d columns with greater than %d%% missing values.' % (len(missing_columns), threshold))
    
#     # Drop the missing columns and return
#     train = train.drop(columns = missing_columns)
#     test = test.drop(columns = missing_columns)
    
#     return train, test
```


```python
train, test = remove_missing_columns(train, test)
```

    There are 6 columns with greater than 90% missing values.
    

# Applying to More Data

## Function to Aggregate Stats at the Client Level


```python
def aggregate_client(df, group_vars, df_names):
    """
    group_vars = ['SK_ID_PREV', 'SK_ID_CURR']
    df_names = ['cash', 'client']
    """
    
    df_agg = agg_numeric(df, parent_var=group_vars[0], df_name=df_names[0])
    
    if any(df.dtypes == "category"):
        df_counts = agg_categorical(df, parent_var=group_vars[0], df_name=df_names[0])
        
        # numeric, categorical 합치기
        df_by_loan = df_counts.merge(df_agg, on=group_vars[0], how="outer")
        
        gc.enable()
        del df_agg, df_counts
        gc.collect()
        
        # client id로 합치기
        df_by_loan = df_by_loan.merge(df[[group_vars[0], group_vars[1]]],
                                      on=group_vars[0], how="left")
        
        df_by_loan = df_by_loan.drop(columns=[group_vars[0]])
        
        df_by_client = agg_numeric(df_by_loan, parent_var=group_vars[1],
                                   df_name=df_names[1])
        
    else:
        df_by_loan = df_agg.merge(df[[group_vars[0], group_vars[1]]],
                                  on=group_vars[0], how="left")
        
        gc.enable()
        del df_agg
        gc.collect()
        
        df_by_loan = df_by_loan.drop(columns=[group_vars[0]])
        
        df_by_client = agg_numeric(df_by_loan, parent_var=group_vars[1],
                                   df_name=df_names[1])
        
    gc.enable()
    del df, df_by_loan
    gc.collect()
    
    return df_by_client
```

## Monthly Cash Data


```python
cash = pd.read_csv('./home_credit/POS_CASH_balance.csv')
cash = convert_types(cash, print_info=True)
cash.head()
```

    Original Memory Usage: 0.64 gb.
    New Memory Usage: 0.41 gb.
    




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
      <th>SK_ID_PREV</th>
      <th>SK_ID_CURR</th>
      <th>MONTHS_BALANCE</th>
      <th>CNT_INSTALMENT</th>
      <th>CNT_INSTALMENT_FUTURE</th>
      <th>NAME_CONTRACT_STATUS</th>
      <th>SK_DPD</th>
      <th>SK_DPD_DEF</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1803195</td>
      <td>182943</td>
      <td>-31</td>
      <td>48.0</td>
      <td>45.0</td>
      <td>Active</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1715348</td>
      <td>367990</td>
      <td>-33</td>
      <td>36.0</td>
      <td>35.0</td>
      <td>Active</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1784872</td>
      <td>397406</td>
      <td>-32</td>
      <td>12.0</td>
      <td>9.0</td>
      <td>Active</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1903291</td>
      <td>269225</td>
      <td>-35</td>
      <td>48.0</td>
      <td>42.0</td>
      <td>Active</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2341044</td>
      <td>334279</td>
      <td>-35</td>
      <td>36.0</td>
      <td>35.0</td>
      <td>Active</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
cash_by_client = aggregate_client(cash, group_vars=['SK_ID_PREV', 'SK_ID_CURR'], df_names=['cash', 'client'])
cash_by_client.head()
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
      <th>client_cash_MONTHS_BALANCE_sum_sum</th>
      <th>client_cash_MONTHS_BALANCE_min_sum</th>
      <th>client_cash_MONTHS_BALANCE_mean_sum</th>
      <th>client_cash_MONTHS_BALANCE_max_sum</th>
      <th>client_cash_MONTHS_BALANCE_sum_min</th>
      <th>client_cash_MONTHS_BALANCE_sum_mean</th>
      <th>client_cash_MONTHS_BALANCE_sum_max</th>
      <th>client_cash_MONTHS_BALANCE_min_min</th>
      <th>client_cash_MONTHS_BALANCE_mean_min</th>
      <th>client_cash_MONTHS_BALANCE_max_min</th>
      <th>...</th>
      <th>client_cash_CNT_INSTALMENT_FUTURE_max_sum</th>
      <th>client_cash_NAME_CONTRACT_STATUS_Active_sum_sum</th>
      <th>client_cash_CNT_INSTALMENT_min_sum</th>
      <th>client_cash_CNT_INSTALMENT_mean_sum</th>
      <th>client_cash_CNT_INSTALMENT_max_sum</th>
      <th>client_cash_CNT_INSTALMENT_count_sum</th>
      <th>client_cash_CNT_INSTALMENT_FUTURE_count_sum</th>
      <th>client_cash_NAME_CONTRACT_STATUS_Active_count_sum</th>
      <th>client_cash_CNT_INSTALMENT_FUTURE_sum_sum</th>
      <th>client_cash_CNT_INSTALMENT_sum_sum</th>
    </tr>
    <tr>
      <th>SK_ID_CURR</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>100001</th>
      <td>-2887</td>
      <td>-669</td>
      <td>-653.0</td>
      <td>-637</td>
      <td>-378</td>
      <td>-320.777778</td>
      <td>-275</td>
      <td>-96</td>
      <td>-94.5</td>
      <td>-93</td>
      <td>...</td>
      <td>28.0</td>
      <td>32.0</td>
      <td>36.0</td>
      <td>36.000000</td>
      <td>36.0</td>
      <td>41</td>
      <td>41</td>
      <td>41</td>
      <td>62.0</td>
      <td>164.0</td>
    </tr>
    <tr>
      <th>100002</th>
      <td>-3610</td>
      <td>-361</td>
      <td>-190.0</td>
      <td>-19</td>
      <td>-190</td>
      <td>-190.000000</td>
      <td>-190</td>
      <td>-19</td>
      <td>-10.0</td>
      <td>-1</td>
      <td>...</td>
      <td>456.0</td>
      <td>361.0</td>
      <td>456.0</td>
      <td>456.000000</td>
      <td>456.0</td>
      <td>361</td>
      <td>361</td>
      <td>361</td>
      <td>5415.0</td>
      <td>8664.0</td>
    </tr>
    <tr>
      <th>100003</th>
      <td>-13240</td>
      <td>-1348</td>
      <td>-1226.0</td>
      <td>-1104</td>
      <td>-858</td>
      <td>-472.857143</td>
      <td>-172</td>
      <td>-77</td>
      <td>-71.5</td>
      <td>-66</td>
      <td>...</td>
      <td>288.0</td>
      <td>256.0</td>
      <td>248.0</td>
      <td>283.000000</td>
      <td>288.0</td>
      <td>272</td>
      <td>272</td>
      <td>272</td>
      <td>1608.0</td>
      <td>2840.0</td>
    </tr>
    <tr>
      <th>100004</th>
      <td>-408</td>
      <td>-108</td>
      <td>-102.0</td>
      <td>-96</td>
      <td>-102</td>
      <td>-102.000000</td>
      <td>-102</td>
      <td>-27</td>
      <td>-25.5</td>
      <td>-24</td>
      <td>...</td>
      <td>16.0</td>
      <td>12.0</td>
      <td>12.0</td>
      <td>15.000000</td>
      <td>16.0</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>36.0</td>
      <td>60.0</td>
    </tr>
    <tr>
      <th>100005</th>
      <td>-2420</td>
      <td>-275</td>
      <td>-220.0</td>
      <td>-165</td>
      <td>-220</td>
      <td>-220.000000</td>
      <td>-220</td>
      <td>-25</td>
      <td>-20.0</td>
      <td>-15</td>
      <td>...</td>
      <td>132.0</td>
      <td>99.0</td>
      <td>99.0</td>
      <td>128.699997</td>
      <td>132.0</td>
      <td>110</td>
      <td>110</td>
      <td>121</td>
      <td>792.0</td>
      <td>1287.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 162 columns</p>
</div>




```python
print('Cash by Client Shape: ', cash_by_client.shape)
train = train.merge(cash_by_client, on="SK_ID_CURR", how="left")
test = test.merge(cash_by_client, on="SK_ID_CURR", how="left")

gc.enable()
del cash, cash_by_client
gc.collect()
```

    Cash by Client Shape:  (337252, 162)
    




    0




```python
train, test = remove_missing_columns(train, test)
```

    There are 0 columns with greater than 90% missing values.
    

## Monthly Credit Data


```python
credit = pd.read_csv("./home_credit/credit_card_balance.csv")
credit = convert_types(credit, print_info=True)
credit.head()
```

    Original Memory Usage: 0.71 gb.
    New Memory Usage: 0.42 gb.
    




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
      <th>SK_ID_PREV</th>
      <th>SK_ID_CURR</th>
      <th>MONTHS_BALANCE</th>
      <th>AMT_BALANCE</th>
      <th>AMT_CREDIT_LIMIT_ACTUAL</th>
      <th>AMT_DRAWINGS_ATM_CURRENT</th>
      <th>AMT_DRAWINGS_CURRENT</th>
      <th>AMT_DRAWINGS_OTHER_CURRENT</th>
      <th>AMT_DRAWINGS_POS_CURRENT</th>
      <th>AMT_INST_MIN_REGULARITY</th>
      <th>...</th>
      <th>AMT_RECIVABLE</th>
      <th>AMT_TOTAL_RECEIVABLE</th>
      <th>CNT_DRAWINGS_ATM_CURRENT</th>
      <th>CNT_DRAWINGS_CURRENT</th>
      <th>CNT_DRAWINGS_OTHER_CURRENT</th>
      <th>CNT_DRAWINGS_POS_CURRENT</th>
      <th>CNT_INSTALMENT_MATURE_CUM</th>
      <th>NAME_CONTRACT_STATUS</th>
      <th>SK_DPD</th>
      <th>SK_DPD_DEF</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2562384</td>
      <td>378907</td>
      <td>-6</td>
      <td>56.970001</td>
      <td>135000</td>
      <td>0.0</td>
      <td>877.5</td>
      <td>0.0</td>
      <td>877.5</td>
      <td>1700.324951</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>35.0</td>
      <td>Active</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2582071</td>
      <td>363914</td>
      <td>-1</td>
      <td>63975.554688</td>
      <td>45000</td>
      <td>2250.0</td>
      <td>2250.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2250.000000</td>
      <td>...</td>
      <td>64875.554688</td>
      <td>64875.554688</td>
      <td>1.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>69.0</td>
      <td>Active</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1740877</td>
      <td>371185</td>
      <td>-7</td>
      <td>31815.224609</td>
      <td>450000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2250.000000</td>
      <td>...</td>
      <td>31460.085938</td>
      <td>31460.085938</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>30.0</td>
      <td>Active</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1389973</td>
      <td>337855</td>
      <td>-4</td>
      <td>236572.109375</td>
      <td>225000</td>
      <td>2250.0</td>
      <td>2250.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>11795.759766</td>
      <td>...</td>
      <td>233048.968750</td>
      <td>233048.968750</td>
      <td>1.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10.0</td>
      <td>Active</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1891521</td>
      <td>126868</td>
      <td>-1</td>
      <td>453919.468750</td>
      <td>450000</td>
      <td>0.0</td>
      <td>11547.0</td>
      <td>0.0</td>
      <td>11547.0</td>
      <td>22924.890625</td>
      <td>...</td>
      <td>453919.468750</td>
      <td>453919.468750</td>
      <td>0.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>101.0</td>
      <td>Active</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>




```python
credit_by_client = aggregate_client(credit, group_vars=['SK_ID_PREV', 'SK_ID_CURR'], df_names=['credit', 'client'])
credit_by_client.head()
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
      <th>client_credit_MONTHS_BALANCE_sum_sum</th>
      <th>client_credit_MONTHS_BALANCE_min_sum</th>
      <th>client_credit_MONTHS_BALANCE_mean_sum</th>
      <th>client_credit_MONTHS_BALANCE_sum_min</th>
      <th>client_credit_MONTHS_BALANCE_sum_mean</th>
      <th>client_credit_MONTHS_BALANCE_sum_max</th>
      <th>client_credit_MONTHS_BALANCE_max_sum</th>
      <th>client_credit_MONTHS_BALANCE_min_min</th>
      <th>client_credit_MONTHS_BALANCE_min_mean</th>
      <th>client_credit_MONTHS_BALANCE_min_max</th>
      <th>...</th>
      <th>client_credit_AMT_DRAWINGS_ATM_CURRENT_mean_max</th>
      <th>client_credit_AMT_PAYMENT_CURRENT_mean_min</th>
      <th>client_credit_AMT_PAYMENT_CURRENT_mean_mean</th>
      <th>client_credit_AMT_PAYMENT_CURRENT_mean_max</th>
      <th>client_credit_AMT_PAYMENT_CURRENT_max_min</th>
      <th>client_credit_AMT_PAYMENT_CURRENT_max_mean</th>
      <th>client_credit_AMT_PAYMENT_CURRENT_max_max</th>
      <th>client_credit_AMT_DRAWINGS_ATM_CURRENT_max_min</th>
      <th>client_credit_AMT_DRAWINGS_ATM_CURRENT_max_mean</th>
      <th>client_credit_AMT_DRAWINGS_ATM_CURRENT_max_max</th>
    </tr>
    <tr>
      <th>SK_ID_CURR</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>100006</th>
      <td>-126</td>
      <td>-36</td>
      <td>-21.0</td>
      <td>-21</td>
      <td>-21.0</td>
      <td>-21</td>
      <td>-6</td>
      <td>-6</td>
      <td>-6.0</td>
      <td>-6</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>100011</th>
      <td>-210826</td>
      <td>-5550</td>
      <td>-2849.0</td>
      <td>-2849</td>
      <td>-2849.0</td>
      <td>-2849</td>
      <td>-148</td>
      <td>-75</td>
      <td>-75.0</td>
      <td>-75</td>
      <td>...</td>
      <td>2432.432373</td>
      <td>4843.063965</td>
      <td>4843.063965</td>
      <td>4843.063965</td>
      <td>55485.0</td>
      <td>55485.0</td>
      <td>55485.0</td>
      <td>180000.0</td>
      <td>180000.0</td>
      <td>180000.0</td>
    </tr>
    <tr>
      <th>100013</th>
      <td>-446976</td>
      <td>-9216</td>
      <td>-4656.0</td>
      <td>-4656</td>
      <td>-4656.0</td>
      <td>-4656</td>
      <td>-96</td>
      <td>-96</td>
      <td>-96.0</td>
      <td>-96</td>
      <td>...</td>
      <td>6350.000000</td>
      <td>7168.346191</td>
      <td>7168.346191</td>
      <td>7168.346191</td>
      <td>153675.0</td>
      <td>153675.0</td>
      <td>153675.0</td>
      <td>157500.0</td>
      <td>157500.0</td>
      <td>157500.0</td>
    </tr>
    <tr>
      <th>100021</th>
      <td>-2890</td>
      <td>-306</td>
      <td>-170.0</td>
      <td>-170</td>
      <td>-170.0</td>
      <td>-170</td>
      <td>-34</td>
      <td>-18</td>
      <td>-18.0</td>
      <td>-18</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>100023</th>
      <td>-480</td>
      <td>-88</td>
      <td>-60.0</td>
      <td>-60</td>
      <td>-60.0</td>
      <td>-60</td>
      <td>-32</td>
      <td>-11</td>
      <td>-11.0</td>
      <td>-11</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 376 columns</p>
</div>




```python
print('Credit by client shape: ', credit_by_client.shape)

train = train.merge(credit_by_client, on='SK_ID_CURR', how='left')
test = test.merge(credit_by_client, on='SK_ID_CURR', how='left')

gc.enable()
del credit, credit_by_client
gc.collect()
```

    Credit by client shape:  (103558, 376)
    




    0




```python
train, test = remove_missing_columns(train, test)
```

    There are 0 columns with greater than 90% missing values.
    

## Installment Payments


```python
installments = pd.read_csv("./home_credit/installments_payments.csv")
installments = convert_types(installments, print_info=True)
installments.head()
```

    Original Memory Usage: 0.87 gb.
    New Memory Usage: 0.49 gb.
    




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
      <th>SK_ID_PREV</th>
      <th>SK_ID_CURR</th>
      <th>NUM_INSTALMENT_VERSION</th>
      <th>NUM_INSTALMENT_NUMBER</th>
      <th>DAYS_INSTALMENT</th>
      <th>DAYS_ENTRY_PAYMENT</th>
      <th>AMT_INSTALMENT</th>
      <th>AMT_PAYMENT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1054186</td>
      <td>161674</td>
      <td>1.0</td>
      <td>6</td>
      <td>-1180.0</td>
      <td>-1187.0</td>
      <td>6948.359863</td>
      <td>6948.359863</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1330831</td>
      <td>151639</td>
      <td>0.0</td>
      <td>34</td>
      <td>-2156.0</td>
      <td>-2156.0</td>
      <td>1716.525024</td>
      <td>1716.525024</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2085231</td>
      <td>193053</td>
      <td>2.0</td>
      <td>1</td>
      <td>-63.0</td>
      <td>-63.0</td>
      <td>25425.000000</td>
      <td>25425.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2452527</td>
      <td>199697</td>
      <td>1.0</td>
      <td>3</td>
      <td>-2418.0</td>
      <td>-2426.0</td>
      <td>24350.130859</td>
      <td>24350.130859</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2714724</td>
      <td>167756</td>
      <td>1.0</td>
      <td>2</td>
      <td>-1383.0</td>
      <td>-1366.0</td>
      <td>2165.040039</td>
      <td>2160.584961</td>
    </tr>
  </tbody>
</table>
</div>




```python
installments_by_client = aggregate_client(installments, group_vars=['SK_ID_PREV', 'SK_ID_CURR'], df_names=['installments', 'client'])
installments_by_client.head()
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
      <th>client_installments_DAYS_ENTRY_PAYMENT_sum_sum</th>
      <th>client_installments_DAYS_INSTALMENT_sum_sum</th>
      <th>client_installments_DAYS_ENTRY_PAYMENT_min_sum</th>
      <th>client_installments_DAYS_INSTALMENT_min_sum</th>
      <th>client_installments_DAYS_ENTRY_PAYMENT_mean_sum</th>
      <th>client_installments_DAYS_INSTALMENT_mean_sum</th>
      <th>client_installments_DAYS_ENTRY_PAYMENT_max_sum</th>
      <th>client_installments_DAYS_INSTALMENT_max_sum</th>
      <th>client_installments_DAYS_INSTALMENT_sum_min</th>
      <th>client_installments_DAYS_ENTRY_PAYMENT_sum_min</th>
      <th>...</th>
      <th>client_installments_AMT_PAYMENT_min_sum</th>
      <th>client_installments_AMT_INSTALMENT_min_sum</th>
      <th>client_installments_AMT_PAYMENT_sum_max</th>
      <th>client_installments_AMT_INSTALMENT_sum_max</th>
      <th>client_installments_AMT_PAYMENT_mean_sum</th>
      <th>client_installments_AMT_INSTALMENT_mean_sum</th>
      <th>client_installments_AMT_INSTALMENT_max_sum</th>
      <th>client_installments_AMT_PAYMENT_max_sum</th>
      <th>client_installments_AMT_PAYMENT_sum_sum</th>
      <th>client_installments_AMT_INSTALMENT_sum_sum</th>
    </tr>
    <tr>
      <th>SK_ID_CURR</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>100001</th>
      <td>-52813.0</td>
      <td>-52598.0</td>
      <td>-15608.0</td>
      <td>-15584.0</td>
      <td>-15365.0</td>
      <td>-15314.0</td>
      <td>-15080.0</td>
      <td>-15044.0</td>
      <td>-8658.0</td>
      <td>-8647.0</td>
      <td>...</td>
      <td>2.774678e+04</td>
      <td>2.774678e+04</td>
      <td>2.925090e+04</td>
      <td>2.925090e+04</td>
      <td>4.119593e+04</td>
      <td>4.119593e+04</td>
      <td>8.153775e+04</td>
      <td>8.153775e+04</td>
      <td>1.528387e+05</td>
      <td>1.528387e+05</td>
    </tr>
    <tr>
      <th>100002</th>
      <td>-113867.0</td>
      <td>-106495.0</td>
      <td>-11153.0</td>
      <td>-10735.0</td>
      <td>-5993.0</td>
      <td>-5605.0</td>
      <td>-931.0</td>
      <td>-475.0</td>
      <td>-5605.0</td>
      <td>-5993.0</td>
      <td>...</td>
      <td>1.757837e+05</td>
      <td>1.757837e+05</td>
      <td>2.196257e+05</td>
      <td>2.196257e+05</td>
      <td>2.196257e+05</td>
      <td>2.196257e+05</td>
      <td>1.008781e+06</td>
      <td>1.008781e+06</td>
      <td>4.172888e+06</td>
      <td>4.172888e+06</td>
    </tr>
    <tr>
      <th>100003</th>
      <td>-367137.0</td>
      <td>-365546.0</td>
      <td>-37757.0</td>
      <td>-37514.0</td>
      <td>-34633.0</td>
      <td>-34454.0</td>
      <td>-31594.0</td>
      <td>-31394.0</td>
      <td>-25740.0</td>
      <td>-25821.0</td>
      <td>...</td>
      <td>1.154108e+06</td>
      <td>1.154108e+06</td>
      <td>1.150977e+06</td>
      <td>1.150977e+06</td>
      <td>1.618865e+06</td>
      <td>1.618865e+06</td>
      <td>4.394102e+06</td>
      <td>4.394102e+06</td>
      <td>1.134881e+07</td>
      <td>1.134881e+07</td>
    </tr>
    <tr>
      <th>100004</th>
      <td>-6855.0</td>
      <td>-6786.0</td>
      <td>-2385.0</td>
      <td>-2352.0</td>
      <td>-2285.0</td>
      <td>-2262.0</td>
      <td>-2181.0</td>
      <td>-2172.0</td>
      <td>-2262.0</td>
      <td>-2285.0</td>
      <td>...</td>
      <td>1.607175e+04</td>
      <td>1.607175e+04</td>
      <td>2.128846e+04</td>
      <td>2.128846e+04</td>
      <td>2.128846e+04</td>
      <td>2.128846e+04</td>
      <td>3.172189e+04</td>
      <td>3.172189e+04</td>
      <td>6.386539e+04</td>
      <td>6.386539e+04</td>
    </tr>
    <tr>
      <th>100005</th>
      <td>-49374.0</td>
      <td>-47466.0</td>
      <td>-6624.0</td>
      <td>-6354.0</td>
      <td>-5486.0</td>
      <td>-5274.0</td>
      <td>-4230.0</td>
      <td>-4194.0</td>
      <td>-5274.0</td>
      <td>-5486.0</td>
      <td>...</td>
      <td>4.331880e+04</td>
      <td>4.331880e+04</td>
      <td>5.616184e+04</td>
      <td>5.616184e+04</td>
      <td>5.616184e+04</td>
      <td>5.616184e+04</td>
      <td>1.589062e+05</td>
      <td>1.589062e+05</td>
      <td>5.054566e+05</td>
      <td>5.054566e+05</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 106 columns</p>
</div>




```python
print('Installments by client shape: ', installments_by_client.shape)

train = train.merge(installments_by_client, on='SK_ID_CURR', how='left')
test = test.merge(installments_by_client, on='SK_ID_CURR', how='left')

gc.enable()
del installments, installments_by_client
gc.collect()
```

    Installments by client shape:  (339587, 106)
    




    20




```python
train, test = remove_missing_columns(train, test)
```

    There are 0 columns with greater than 90% missing values.
    


```python
print('Final Training Shape: ', train.shape)
print('Final Testing Shape: ', test.shape)
```

    Final Training Shape:  (307511, 1125)
    Final Testing Shape:  (48744, 1124)
    


```python
print(f'Final training size: {return_size(train)}')
print(f'Final testing size: {return_size(test)}')
```

    Final training size: 2.08
    Final testing size: 0.33
    

## Save All Newly Calculated Features
안타깝게도 생성된 모든 feature를 저장하는 것은 Kaggle 노트북에서 작동하지 않습니다. 개인 컴퓨터에서 코드를 실행해야 합니다.

# Modeling


```python
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
```


```python
def model(features, test_features, encoding="ohe", n_folds=5):
    train_ids = features["SK_ID_CURR"]
    test_ids = test_features["SK_ID_CURR"]
    
    labels = features["TARGET"]
    
    features = features.drop(columns=["SK_ID_CURR"])
    test_features = test_features.drop(columns=["SK_ID_CURR"])
    
    if encoding == "ohe":
        features = pd.get_dummies(features)
        test_features = pd.get_dummies(test_features)
        
        features, test_features = features.align(test_features, join="inner",
                                                 axis=1)
        
        # ohe-hot encoding을 한 경우, 카테고리형 변수를 알려주지 않아도 됨
        cat_indices = "auto"
        
    elif encoding == "le":
        label_encoder = LabelEncdoer()
        
        cat_indices = []
        
        for i, col in enumerate(features):
            if features[col].dtype == "object":
                features[col] = label_encoder.fit_transform(np.array(features[col].astype(str)).resshape((-1, )))
                test_features[col] = label_encoder.fit_transform(np.array(test_features[col].astype(str)).resshape((-1, )))
                
                cat_indices.append(i)
                
    else:
        raise ValueError("Encoding must be either 'ohe' or 'le'")
    
    print('Training Data Shape: ', features.shape)
    print('Testing Data Shape: ', test_features.shape)
    
    feature_names = list(features.columns)
    
    features = np.array(features)
    test_features = np.array(test_features)
    
    k_fold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=50)
    
    feature_importance_values = np.zeros(len(feature_names))
    
    test_predictions = np.zeros(test_features.shape[0])
    
    out_of_fold = np.zeros(features.shape[0])
    
    valid_scores = []
    train_scores = []
    
    for train_indices, valid_indices in k_fold.split(features, labels):
        train_features, train_labels = features[train_indices], labels[train_indices]
        valid_features, valid_labels = features[valid_indices], labels[valid_indices]
        
        model = lgb.LGBMClassifier(n_estimators=10000, objective="binary",
                                   class_weight="balanced",
                                   learning_rate=0.05,
                                   reg_alpha=0.1, reg_lambda=0.1,
                                   subsample=0.8, n_jobs=-1, random_state=50)
        
        model.fit(train_features, train_labels, eval_metric="auc",
                  eval_set=[(valid_features, valid_labels), (train_features, train_labels)],
                  eval_names=["valid", "train"], categorical_feature=cat_indices,
                  early_stopping_rounds=100, verbose=200)
        
        best_iteration = model.best_iteration_
        
        feature_importance_values += model.feature_importances_ / k_fold.n_splits
        
        test_predictions += model.predict_proba(test_features, 
                                                num_iteration=best_iteration)[:, 1] / k_fold.n_splits
        
        out_of_fold[valid_indices] = model.predict_proba(valid_features, 
                                                         num_iteration=best_iteration)[:, 1]
        
        valid_score = model.best_score_["valid"]["auc"]
        train_score = model.best_score_["train"]["auc"]
        
        valid_scores.append(valid_score)
        train_scores.append(train_score)
        
        gc.enable()
        del model, train_features, valid_features
        gc.collect()
    
    submission = pd.DataFrame({"SK_ID_CURR": test_ids,
                               "TARGET": test_predictions})
    
    feature_importances = pd.DataFrame({"feature": feature_names, 
                                        "importance": feature_importance_values})
    
    valid_auc = roc_auc_score(labels, out_of_fold)
    
    # 전체 점수
    valid_scores.append(valid_auc)
    train_scores.append(np.mean(train_scores))
    
    fold_names = list(range(n_folds))
    fold_names.append("overall")
    
    metrics = pd.DataFrame({"fold": fold_names, "train": train_scores,
                            "valid": valid_scores})
    
    return submission, feature_importance_values, metrics
```


```python
submission, fi, metrics = model(train, test)
```

    Training Data Shape:  (307511, 1244)
    Testing Data Shape:  (48744, 1244)
    [200]	train's auc: 0.827847	train's binary_logloss: 0.517851	valid's auc: 0.776498	valid's binary_logloss: 0.53754
    [200]	train's auc: 0.826893	train's binary_logloss: 0.519028	valid's auc: 0.777111	valid's binary_logloss: 0.539778
    [400]	train's auc: 0.862668	train's binary_logloss: 0.479481	valid's auc: 0.778615	valid's binary_logloss: 0.514953
    [200]	train's auc: 0.82652	train's binary_logloss: 0.519854	valid's auc: 0.782785	valid's binary_logloss: 0.536581
    [400]	train's auc: 0.862059	train's binary_logloss: 0.480909	valid's auc: 0.783537	valid's binary_logloss: 0.512251
    [200]	train's auc: 0.827632	train's binary_logloss: 0.518235	valid's auc: 0.780399	valid's binary_logloss: 0.538406
    [400]	train's auc: 0.863706	train's binary_logloss: 0.478578	valid's auc: 0.780408	valid's binary_logloss: 0.513719
    [200]	train's auc: 0.828105	train's binary_logloss: 0.517592	valid's auc: 0.776329	valid's binary_logloss: 0.538545
    [400]	train's auc: 0.863528	train's binary_logloss: 0.478105	valid's auc: 0.776944	valid's binary_logloss: 0.513118
    


```python
metrics
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
      <th>fold</th>
      <th>train</th>
      <th>valid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.838376</td>
      <td>0.777163</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.864993</td>
      <td>0.778851</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0.853780</td>
      <td>0.783790</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0.847372</td>
      <td>0.780880</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0.860250</td>
      <td>0.777134</td>
    </tr>
    <tr>
      <th>5</th>
      <td>overall</td>
      <td>0.852954</td>
      <td>0.779494</td>
    </tr>
  </tbody>
</table>
</div>




```python
submission.to_csv('submission_manualp2.csv', index=False)
```

-------------

참고: [https://www.kaggle.com/code/willkoehrsen/introduction-to-manual-feature-engineering-p2/notebook](https://www.kaggle.com/code/willkoehrsen/introduction-to-manual-feature-engineering-p2/notebook)