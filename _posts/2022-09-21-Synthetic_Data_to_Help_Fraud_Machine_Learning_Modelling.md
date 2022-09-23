---
layout: single
title:  "Fraud Machine Learning Modelling을 돕기 위한 합성 데이터"
categories: ML
tag: [synthetic data, ydata_synthetic, CWGAN-GP]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: false
---
**[공지사항]** ["출처: https://towardsdatascience.com/synthetic-data-to-help-fraud-machine-learning-modelling-c28cdf04e12a"](https://towardsdatascience.com/synthetic-data-to-help-fraud-machine-learning-modelling-c28cdf04e12a)
{: .notice--danger}

# 합성 데이터는 fraud cases를 완하하는데 도움을 줄 수 있습니다

<p align="center"><img src="/assets/images/220923/1.png"></p>

Fraud cases는 모든 비즈니스 산업에서 일반적이며 막대한 재정적 손실을 초래합니다. 작든 크든, 좋든 싫든 모든 기업은 사기 문제에 직면하게 됩니다.

Fraud problem을 완화하기 위해 machine learning fraud detection 연구에 많은 노력을 기울였지만, 아직 완벽한 솔루션은 없습니다. 비즈니스마다 요구 사항이 다르고 데이터가 끊임없이 진화하고 있기 때문입니다.

완벽한 솔루션은 없지만 모델을 개선할 수 있는 몇 가지 방법이 있습니다. 솔루션 중 하나는 합성 데이터를 사용하는 것입니다. 합성 데이터란 무엇이며  fraud detection에 어떻게 도움이 될까요?

이제 시작해봅시다.

# Synthetic Data
합성 데이터는 컴퓨터 기술을 통해 생성된 데이터로 현실 세계에는 존재하지 않습니다. 즉, 합성 데이터는 직접 수집하지 않고, 생성된 데이터로 정의할 수 있습니다.

합성 데이터는 데이터 세계에서 새로운 것이 아닙니다. 그러나 기술이 발전함에 따라 합성 데이터는 더욱 중요해지고 다양한 산업에 영향을 미치고 있습니다. 왜 이런 영향을 미치게 되었는지, 데이터 과학 세계에서 합성 데이터의 여러 application을 살펴보겠습니다:

- 데이터 수집 노력 없이 방대한 양의 데이터 생성


- 실제 상황을 반영하는 데이터 세트 생성


- 데이터 사용 개인 정보 보호


- 아직 발생하지 않은 조건에서의 시뮬레이션


- 데이터 imbalance 완화

합성 데이터에 대한 연구가 아직 진행 중이기 때문에 리스트는 계속 늘어날 것입니다. 요점은 합성 데이터가 데이터 과학에 도움이 되고 업계에 영향을 미친다는 것입니다.

또한, 합성 데이터는 데이터가 생성되고 저장되는 방식에 따라 분류될 수 있습니다. 분류는 다음과 같습니다:

- **Full Synthetic Data:** 원본 데이터를 기반으로 했지만, 원본 데이터를 포함하지 않았습니다. 데이터 세트에는 합성 데이터만 포함되지만, 원본 데이터와 유사한 속성을 지니고 있습니다.


- **Partial Synthetic Data:** variable 수준에서 원본 및 합성 데이터의 조합. 이 category는 민감한 데이터와 같은 특정 변수를 합성 변수로 대체하려는 경우에 자주 사용됩니다.


- **Hybrid Synthetic Data:** 데이터 생성은 실제 데이터와 합성 데이터 모두에서 이뤄집니다. variable 사이의 기본 분포와 관계는 그대로지만, 데이터 세트에는 원본 데이터와 합성 데이터가 모두 포함됩니다.

지금까지 합성 데이터와 그 유용성에 대해 배웠지만, fraud machine learning 개발에 어떻게 도움이 될 수 있을까요? 한발 물러서서 fraud 데이터 세트의 일반적인 사례를 살펴봐야 합니다.

# Fraud Modelling
Fradu 방지 data science 프로젝트의 성공은 비즈니스 전략과 fraud 모델의 두 가지 요소에 달려 있습니다.

앞서 언급했듯이 fraud 사건은 거의 발생하지 않지만, 각 사건은 많은 손실을 초래할 수 있습니다. 이는 fraud 모델링에 imbalance data case를 맞이하게 된다는 것입니다.

연구에 따르면, 합성 데이터는 소수 데이터를 oversampling하고 balanced 데이터 세트를 생성하여 imbalance 문제를 완화하는 데 도움이 될 수 있습니다. 예를 들어,  [paper](https://arxiv.org/pdf/2204.00144.pdf) by Dina et al. (2022)은 CTGAN에 의해 생성된 합성 데이터는 imbalanced 데이터에 대해 훈련된 동일한 ML 모델에 비해 정확도가 8% 향상되었음을 보여줍니다.

합성 데이터 밸런싱을 위한 가장 유명한 전략은 SMOTE이지만, 이 기술은 복잡한 데이터에서 거의 이점을 제공하지 않습니다. 이것이 ML 성능을 높이는 데 도움이 되는 것으로 입증된 GAN 모델을 주로 사용하여 데이터 합성을 위한 다른 접근 방식을 시도하는 이유입니다.

합성 데이터는 장점이 있지만, 연구가 시작된지는 아직 얼마되지 않았기 때문에 합성 데이터를 모델링 프로세스에 적용할 때 다음과 같은 단점에 유의하세요:

- 데이터 복잡성이 증가하면, 생성된 합성 데이터가 실제 population을 나타내지 않을 수 있습니다. 이렇게 하면 모델이 잘못된 insights을 배우고 잘못된 예측을 하게 됩니다.


- 합성 데이터 quality는 데이터 생성에 사용된 데이터 세트에 따라 다릅니다. 잘못된 원본 데이터는 잘못된 합성 데이터를 생성하여 모델의 output이 부정확합니다.

모델링에 합성 데이터를 사용할 때의 위험성과 약점을 이해했다면, imbalanced 데이터가 fraud에 어떻게 도움이 되는지 직접 접근해 보겠습니다.

# Develop a Fraud Model Detection
이 예에서는 Shivam Bansal이 Kaggle에서 제공하는 '[Vehicle Insurance Claim Fraud Detection](https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection?select=fraud_oracle.csv)' 데이터 세트를 사용합니다. 이 데이터 세트에는 claim에 대해 사기를 저지를 고객을 감지하는 비즈니스 문제를 가지고 있습니다.


```python
import pandas as pd
from pandas_profiling import ProfileReport

df = pd.read_csv("fraud_oracle.csv")

profile = ProfileReport(df)
```

<p align="center"><img src="/assets/images/220923/2.png"></p>

전반적으로, 약 33개의 변수와 15420개의 관측치가 있으며 대부분의 데이터는 범주형입니다. 누락된 데이터가 없으므로 누락된 데이터를 처리할 필요가 없습니다. 분포를 보기 위해 변수 target을 확인해봅시다.

<p align="center"><img src="/assets/images/220923/3.png"></p>

위의 summary에서 알 수 있듯이 target 'FraudFound'의 imbalance가 심각합니다. fraud가 아닌 데이터와 비교하여 6%의 데이터(923개의 관측치)만이 fraud였습니다.

다음 파트에서는 차량 보험 사기를 예측하는 분류기 모델을 구축해 보겠습니다. 모든 데이터 세트를 사용하지 않고, training 목적으로 일부 categorical encoding을 수행합니다.


```python
df = df[['AccidentArea', 'Sex', 'MaritalStatus', 'Age', 'Fault', 'PolicyType', 
         'VehicleCategory', 'VehiclePrice', 'Deductible', 'DriverRating',
         'Days_Policy_Accident','Days_Policy_Claim','PastNumberOfClaims', 
         'AgeOfVehicle','BasePolicy', 'FraudFound_P']]

df = pd.get_dummies(df, columns=df.select_dtypes("object").columns,
                    drop_first=True)
```

Data cleaning 후, 모델을 훈련시킵니다.


```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(df.drop("FraudFound_P", axis=1),
                                                    df["FraudFound_P"],
                                                    train_size=0.7,
                                                    stratify=df["FraudFound_P"], # 데이터 비율 유지
                                                    random_state=100)

model = RandomForestClassifier(random_state=100)
model.fit(X_train, y_train)
```




<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>RandomForestClassifier(random_state=100)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">RandomForestClassifier</label><div class="sk-toggleable__content"><pre>RandomForestClassifier(random_state=100)</pre></div></div></div></div></div>



RandomForest를 사용하여 fraud 모델의 성능을 평가해 봅니다.


```python
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
```

                  precision    recall  f1-score   support
    
               0       0.94      0.98      0.96      4349
               1       0.14      0.04      0.07       277
    
        accuracy                           0.93      4626
       macro avg       0.54      0.51      0.51      4626
    weighted avg       0.89      0.93      0.91      4626
    
    

위의 결과에서 알 수 있듯이 대부분의 예측은 fraud가 아닌 경우를 예측합니다. 이 아티클에서는 이미 이러한 경우를 예상했으므로 추가 합성 데이터를 사용하여 모델 성능을 높이려고 합니다.

먼저 패키지를 설치해야 합니다. ydata-synthetic 패키지에서 가져온 모델을 사용하여 작업을 더 쉽게 만들 것입니다. 이 예에서는 [Conditional Wassertein GAN with Gradient Penalty (CWGAN-GP)](https://cameronfabbri.github.io/papers/conditionalWGAN.pdf) 모델을 사용하여 합성 데이터를 생성합니다. 이 모델은 데이터 세트의 balance를 맞추는 데 적합합니다.


```python
# !pip install ydata-synthetic
```

설치 후, CWGAN-GP 모델이 학습할 데이터 세트를 설정합니다. 또한 분할한 train 데이터를 기반으로 합성 데이터를 생성합니다. 그 이유는 테스트 데이터 세트에 합성 데이터를 포함하여 data leaks을 피하고 싶기 때문입니다.


```python
X_train_synth = X_train.copy()
X_train_synth['FraudFound_P'] = y_train
```

모델에 소수의 데이터만 synthesizer에 훈련시키고 싶기 때문에, fraud cases인 경우만 데이터 세트로 선택합니다.


```python
X_train_synth_min = X_train_synth[X_train_synth["FraudFound_P"] == 1].copy()
```

다음 단계는 CWGAN-GP 모델의 개발입니다. parameter를 지정합니다


```python
from ydata_synthetic.synthesizers.regular import cwgangp
from ydata_synthetic.synthesizers import ModelParameters, TrainParameters

synth_model = cwgangp.model.CWGANGP

# parameter
noise_dim = 61
dim = 128
batch_size = 128
log_step = 100
epochs = 200
learning_rate = 5e-4
beta_1 = 0.5
beta_2 = 0.9
models_dir = './cache'

gan_args = ModelParameters(batch_size=batch_size, lr=learning_rate, betas=(beta_1, beta_2),
                           noise_dim=noise_dim, layers_dim=dim)

train_args = TrainParameters(epochs=epochs, sample_interval=log_step)

synthesizer = synth_model(gan_args, n_critic=10)
```

모델을 학습시키기 위해 어떤 칼럼이 numerical인지 categorical인지 결정해야 합니다. 이 예에서는 모든 칼럼을 numerical로 취급합니다.


```python
synthesizer.fit(data=X_train_synth_min, train_arguments=train_args, 
                num_cols=list(X_train_synth_min.drop('FraudFound_P', axis=1).columns), 
                cat_cols=[], label_cols=['FraudFound_P'])
```

      0%|                                                                                          | 0/200 [00:00<?, ?it/s]

    Number of iterations per epoch: 6
    

      0%|▍                                                                                 | 1/200 [00:07<24:42,  7.45s/it]

    Epoch: 0 | critic_loss: -0.18840967118740082 | gen_loss: 0.03733024373650551
    

      1%|▊                                                                                 | 2/200 [00:07<10:35,  3.21s/it]

    Epoch: 1 | critic_loss: -0.06161585450172424 | gen_loss: 0.08629059791564941
    

      2%|█▏                                                                                | 3/200 [00:07<06:06,  1.86s/it]

    Epoch: 2 | critic_loss: -0.2222920060157776 | gen_loss: 0.13438183069229126
    

      2%|█▋                                                                                | 4/200 [00:08<04:00,  1.23s/it]

    Epoch: 3 | critic_loss: -0.18035899102687836 | gen_loss: 0.15498244762420654
    

      2%|██                                                                                | 5/200 [00:08<02:49,  1.15it/s]

    Epoch: 4 | critic_loss: -0.15652552247047424 | gen_loss: 0.18586257100105286
    

      3%|██▍                                                                               | 6/200 [00:08<02:07,  1.52it/s]

    Epoch: 5 | critic_loss: -0.16978693008422852 | gen_loss: 0.19723007082939148
    

      4%|██▊                                                                               | 7/200 [00:08<01:40,  1.92it/s]

    Epoch: 6 | critic_loss: -0.058145806193351746 | gen_loss: 0.22440189123153687
    

      4%|███▎                                                                              | 8/200 [00:09<01:23,  2.30it/s]

    Epoch: 7 | critic_loss: -0.031163856387138367 | gen_loss: 0.228996142745018
    

      4%|███▋                                                                              | 9/200 [00:09<01:13,  2.61it/s]

    Epoch: 8 | critic_loss: -0.04187111556529999 | gen_loss: 0.22223684191703796
    

      5%|████                                                                             | 10/200 [00:09<01:06,  2.86it/s]

    Epoch: 9 | critic_loss: -0.12917296588420868 | gen_loss: 0.23300811648368835
    

      6%|████▍                                                                            | 11/200 [00:09<01:01,  3.05it/s]

    Epoch: 10 | critic_loss: -0.14195621013641357 | gen_loss: 0.2642074227333069
    

      6%|████▊                                                                            | 12/200 [00:10<00:58,  3.20it/s]

    Epoch: 11 | critic_loss: -0.18270359933376312 | gen_loss: 0.2693111300468445
    

      6%|█████▎                                                                           | 13/200 [00:10<00:56,  3.31it/s]

    Epoch: 12 | critic_loss: -0.15068601071834564 | gen_loss: 0.26711422204971313
    

      7%|█████▋                                                                           | 14/200 [00:10<00:54,  3.38it/s]

    Epoch: 13 | critic_loss: -0.17398734390735626 | gen_loss: 0.2810978293418884
    

      8%|██████                                                                           | 15/200 [00:11<00:53,  3.44it/s]

    Epoch: 14 | critic_loss: -0.17138810455799103 | gen_loss: 0.28365975618362427
    

      8%|██████▍                                                                          | 16/200 [00:11<00:53,  3.43it/s]

    Epoch: 15 | critic_loss: -0.17812904715538025 | gen_loss: 0.25822219252586365
    

      8%|██████▉                                                                          | 17/200 [00:11<00:52,  3.48it/s]

    Epoch: 16 | critic_loss: -0.17729909718036652 | gen_loss: 0.30871883034706116
    

      9%|███████▎                                                                         | 18/200 [00:11<00:51,  3.51it/s]

    Epoch: 17 | critic_loss: -0.17693321406841278 | gen_loss: 0.3022943139076233
    

     10%|███████▋                                                                         | 19/200 [00:12<00:51,  3.50it/s]

    Epoch: 18 | critic_loss: -0.01916174590587616 | gen_loss: 0.31317105889320374
    

     10%|████████                                                                         | 20/200 [00:12<00:51,  3.51it/s]

    Epoch: 19 | critic_loss: -0.16351042687892914 | gen_loss: 0.35547035932540894
    

     10%|████████▌                                                                        | 21/200 [00:12<00:50,  3.51it/s]

    Epoch: 20 | critic_loss: -0.18230712413787842 | gen_loss: 0.32079052925109863
    

     11%|████████▉                                                                        | 22/200 [00:13<00:52,  3.42it/s]

    Epoch: 21 | critic_loss: -0.19166556000709534 | gen_loss: 0.30292344093322754
    

     12%|█████████▎                                                                       | 23/200 [00:13<00:53,  3.33it/s]

    Epoch: 22 | critic_loss: -0.1864338517189026 | gen_loss: 0.3262517750263214
    

     12%|█████████▋                                                                       | 24/200 [00:13<00:52,  3.34it/s]

    Epoch: 23 | critic_loss: -0.16972339153289795 | gen_loss: 0.3279057443141937
    

     12%|██████████▏                                                                      | 25/200 [00:14<00:51,  3.37it/s]

    Epoch: 24 | critic_loss: -0.19312818348407745 | gen_loss: 0.3238387405872345
    

     13%|██████████▌                                                                      | 26/200 [00:14<00:51,  3.40it/s]

    Epoch: 25 | critic_loss: -0.1822628229856491 | gen_loss: 0.34720057249069214
    

     14%|██████████▉                                                                      | 27/200 [00:14<00:50,  3.43it/s]

    Epoch: 26 | critic_loss: -0.16222897171974182 | gen_loss: 0.3588358461856842
    

     14%|███████████▎                                                                     | 28/200 [00:14<00:50,  3.39it/s]

    Epoch: 27 | critic_loss: -0.20526570081710815 | gen_loss: 0.2955209016799927
    

     14%|███████████▋                                                                     | 29/200 [00:15<00:50,  3.41it/s]

    Epoch: 28 | critic_loss: -0.22017785906791687 | gen_loss: 0.32519829273223877
    

     15%|████████████▏                                                                    | 30/200 [00:15<00:49,  3.45it/s]

    Epoch: 29 | critic_loss: -0.028029561042785645 | gen_loss: 0.33196333050727844
    

     16%|████████████▌                                                                    | 31/200 [00:15<00:48,  3.47it/s]

    Epoch: 30 | critic_loss: -0.18671752512454987 | gen_loss: 0.32129910588264465
    

     16%|████████████▉                                                                    | 32/200 [00:16<00:48,  3.48it/s]

    Epoch: 31 | critic_loss: -0.20709607005119324 | gen_loss: 0.32487213611602783
    

     16%|█████████████▎                                                                   | 33/200 [00:16<00:48,  3.48it/s]

    Epoch: 32 | critic_loss: -0.21102604269981384 | gen_loss: 0.3262215852737427
    

     17%|█████████████▊                                                                   | 34/200 [00:16<00:47,  3.48it/s]

    Epoch: 33 | critic_loss: -0.20651109516620636 | gen_loss: 0.3222775161266327
    

     18%|██████████████▏                                                                  | 35/200 [00:16<00:48,  3.43it/s]

    Epoch: 34 | critic_loss: -0.20740610361099243 | gen_loss: 0.33706310391426086
    

     18%|██████████████▌                                                                  | 36/200 [00:17<00:47,  3.43it/s]

    Epoch: 35 | critic_loss: -0.21264103055000305 | gen_loss: 0.34640416502952576
    

     18%|██████████████▉                                                                  | 37/200 [00:17<00:47,  3.44it/s]

    Epoch: 36 | critic_loss: -0.20959866046905518 | gen_loss: 0.3171178698539734
    

     19%|███████████████▍                                                                 | 38/200 [00:17<00:46,  3.45it/s]

    Epoch: 37 | critic_loss: -0.21862775087356567 | gen_loss: 0.3392353951931
    

     20%|███████████████▊                                                                 | 39/200 [00:18<00:47,  3.43it/s]

    Epoch: 38 | critic_loss: -0.20555825531482697 | gen_loss: 0.29389309883117676
    

     20%|████████████████▏                                                                | 40/200 [00:18<00:47,  3.40it/s]

    Epoch: 39 | critic_loss: -0.21553707122802734 | gen_loss: 0.3630827069282532
    

     20%|████████████████▌                                                                | 41/200 [00:18<00:46,  3.44it/s]

    Epoch: 40 | critic_loss: -0.21727615594863892 | gen_loss: 0.3468192219734192
    

     21%|█████████████████                                                                | 42/200 [00:18<00:45,  3.47it/s]

    Epoch: 41 | critic_loss: -0.14304694533348083 | gen_loss: 0.3662793040275574
    

     22%|█████████████████▍                                                               | 43/200 [00:19<00:45,  3.44it/s]

    Epoch: 42 | critic_loss: -0.1817150115966797 | gen_loss: 0.32429155707359314
    

     22%|█████████████████▊                                                               | 44/200 [00:19<00:45,  3.46it/s]

    Epoch: 43 | critic_loss: -0.23568402230739594 | gen_loss: 0.36516010761260986
    

     22%|██████████████████▏                                                              | 45/200 [00:19<00:44,  3.48it/s]

    Epoch: 44 | critic_loss: -0.22753556072711945 | gen_loss: 0.36574888229370117
    

     23%|██████████████████▋                                                              | 46/200 [00:20<00:44,  3.44it/s]

    Epoch: 45 | critic_loss: -0.18554610013961792 | gen_loss: 0.3498827815055847
    

     24%|███████████████████                                                              | 47/200 [00:20<00:45,  3.40it/s]

    Epoch: 46 | critic_loss: -0.22544097900390625 | gen_loss: 0.35783758759498596
    

     24%|███████████████████▍                                                             | 48/200 [00:20<00:44,  3.41it/s]

    Epoch: 47 | critic_loss: -0.23484359681606293 | gen_loss: 0.3798193633556366
    

     24%|███████████████████▊                                                             | 49/200 [00:21<00:44,  3.43it/s]

    Epoch: 48 | critic_loss: -0.223350390791893 | gen_loss: 0.32446756958961487
    

     25%|████████████████████▎                                                            | 50/200 [00:21<00:43,  3.41it/s]

    Epoch: 49 | critic_loss: -0.22570274770259857 | gen_loss: 0.3751850128173828
    

     26%|████████████████████▋                                                            | 51/200 [00:21<00:43,  3.42it/s]

    Epoch: 50 | critic_loss: -0.21425823867321014 | gen_loss: 0.31816259026527405
    

     26%|█████████████████████                                                            | 52/200 [00:21<00:43,  3.40it/s]

    Epoch: 51 | critic_loss: -0.2245754450559616 | gen_loss: 0.3842402994632721
    

     26%|█████████████████████▍                                                           | 53/200 [00:22<00:43,  3.41it/s]

    Epoch: 52 | critic_loss: -0.2378351241350174 | gen_loss: 0.3931001126766205
    

     27%|█████████████████████▊                                                           | 54/200 [00:22<00:43,  3.39it/s]

    Epoch: 53 | critic_loss: -0.22294089198112488 | gen_loss: 0.3855641782283783
    

     28%|██████████████████████▎                                                          | 55/200 [00:22<00:42,  3.39it/s]

    Epoch: 54 | critic_loss: -0.21079280972480774 | gen_loss: 0.38321831822395325
    

     28%|██████████████████████▋                                                          | 56/200 [00:23<00:42,  3.39it/s]

    Epoch: 55 | critic_loss: -0.2221517264842987 | gen_loss: 0.3699430823326111
    

     28%|███████████████████████                                                          | 57/200 [00:23<00:42,  3.38it/s]

    Epoch: 56 | critic_loss: -0.24232064187526703 | gen_loss: 0.40048637986183167
    

     29%|███████████████████████▍                                                         | 58/200 [00:23<00:43,  3.30it/s]

    Epoch: 57 | critic_loss: -0.2409924864768982 | gen_loss: 0.40212133526802063
    

     30%|███████████████████████▉                                                         | 59/200 [00:23<00:42,  3.30it/s]

    Epoch: 58 | critic_loss: -0.24176256358623505 | gen_loss: 0.4005703628063202
    

     30%|████████████████████████▎                                                        | 60/200 [00:24<00:41,  3.35it/s]

    Epoch: 59 | critic_loss: -0.25570499897003174 | gen_loss: 0.423927903175354
    

     30%|████████████████████████▋                                                        | 61/200 [00:24<00:41,  3.37it/s]

    Epoch: 60 | critic_loss: -0.23650819063186646 | gen_loss: 0.3820476531982422
    

     31%|█████████████████████████                                                        | 62/200 [00:24<00:40,  3.40it/s]

    Epoch: 61 | critic_loss: -0.13492217659950256 | gen_loss: 0.40035322308540344
    

     32%|█████████████████████████▌                                                       | 63/200 [00:25<00:40,  3.42it/s]

    Epoch: 62 | critic_loss: -0.22140148282051086 | gen_loss: 0.325278103351593
    

     32%|█████████████████████████▉                                                       | 64/200 [00:25<00:39,  3.44it/s]

    Epoch: 63 | critic_loss: -0.25078997015953064 | gen_loss: 0.4220958650112152
    

     32%|██████████████████████████▎                                                      | 65/200 [00:25<00:39,  3.43it/s]

    Epoch: 64 | critic_loss: -0.25431713461875916 | gen_loss: 0.42779776453971863
    

     33%|██████████████████████████▋                                                      | 66/200 [00:26<00:39,  3.43it/s]

    Epoch: 65 | critic_loss: -0.24193213880062103 | gen_loss: 0.3858298063278198
    

     34%|███████████████████████████▏                                                     | 67/200 [00:26<00:39,  3.39it/s]

    Epoch: 66 | critic_loss: -0.2422177642583847 | gen_loss: 0.3854648470878601
    

     34%|███████████████████████████▌                                                     | 68/200 [00:26<00:39,  3.37it/s]

    Epoch: 67 | critic_loss: -0.2650587260723114 | gen_loss: 0.43304184079170227
    

     34%|███████████████████████████▉                                                     | 69/200 [00:26<00:38,  3.42it/s]

    Epoch: 68 | critic_loss: -0.20429494976997375 | gen_loss: 0.36637356877326965
    

     35%|████████████████████████████▎                                                    | 70/200 [00:27<00:38,  3.42it/s]

    Epoch: 69 | critic_loss: -0.2427869737148285 | gen_loss: 0.3877147436141968
    

     36%|████████████████████████████▊                                                    | 71/200 [00:27<00:37,  3.41it/s]

    Epoch: 70 | critic_loss: -0.24758070707321167 | gen_loss: 0.40981465578079224
    

     36%|█████████████████████████████▏                                                   | 72/200 [00:27<00:37,  3.45it/s]

    Epoch: 71 | critic_loss: -0.2581954598426819 | gen_loss: 0.38129404187202454
    

     36%|█████████████████████████████▌                                                   | 73/200 [00:28<00:36,  3.47it/s]

    Epoch: 72 | critic_loss: -0.24487003684043884 | gen_loss: 0.39252394437789917
    

     37%|█████████████████████████████▉                                                   | 74/200 [00:28<00:36,  3.47it/s]

    Epoch: 73 | critic_loss: -0.25425684452056885 | gen_loss: 0.40929079055786133
    

     38%|██████████████████████████████▍                                                  | 75/200 [00:28<00:36,  3.46it/s]

    Epoch: 74 | critic_loss: -0.2667801082134247 | gen_loss: 0.3995470106601715
    

     38%|██████████████████████████████▊                                                  | 76/200 [00:28<00:35,  3.46it/s]

    Epoch: 75 | critic_loss: -0.26049959659576416 | gen_loss: 0.3829888701438904
    

     38%|███████████████████████████████▏                                                 | 77/200 [00:29<00:35,  3.45it/s]

    Epoch: 76 | critic_loss: -0.23728244006633759 | gen_loss: 0.38839197158813477
    

     39%|███████████████████████████████▌                                                 | 78/200 [00:29<00:35,  3.45it/s]

    Epoch: 77 | critic_loss: -0.23814870417118073 | gen_loss: 0.3790966868400574
    

     40%|███████████████████████████████▉                                                 | 79/200 [00:29<00:34,  3.46it/s]

    Epoch: 78 | critic_loss: -0.2559835612773895 | gen_loss: 0.38460248708724976
    

     40%|████████████████████████████████▍                                                | 80/200 [00:30<00:35,  3.42it/s]

    Epoch: 79 | critic_loss: -0.2578791081905365 | gen_loss: 0.39483532309532166
    

     40%|████████████████████████████████▊                                                | 81/200 [00:30<00:34,  3.41it/s]

    Epoch: 80 | critic_loss: -0.2551480233669281 | gen_loss: 0.4234008491039276
    

     41%|█████████████████████████████████▏                                               | 82/200 [00:30<00:34,  3.40it/s]

    Epoch: 81 | critic_loss: -0.2587549686431885 | gen_loss: 0.4106212854385376
    

     42%|█████████████████████████████████▌                                               | 83/200 [00:30<00:34,  3.36it/s]

    Epoch: 82 | critic_loss: -0.18323181569576263 | gen_loss: 0.35344940423965454
    

     42%|██████████████████████████████████                                               | 84/200 [00:31<00:34,  3.34it/s]

    Epoch: 83 | critic_loss: -0.24634215235710144 | gen_loss: 0.41156867146492004
    

     42%|██████████████████████████████████▍                                              | 85/200 [00:31<00:34,  3.31it/s]

    Epoch: 84 | critic_loss: -0.23395822942256927 | gen_loss: 0.39288395643234253
    

     43%|██████████████████████████████████▊                                              | 86/200 [00:31<00:33,  3.36it/s]

    Epoch: 85 | critic_loss: -0.19759848713874817 | gen_loss: 0.3763793706893921
    

     44%|███████████████████████████████████▏                                             | 87/200 [00:32<00:33,  3.39it/s]

    Epoch: 86 | critic_loss: -0.22596175968647003 | gen_loss: 0.39175358414649963
    

     44%|███████████████████████████████████▋                                             | 88/200 [00:32<00:32,  3.41it/s]

    Epoch: 87 | critic_loss: -0.26906150579452515 | gen_loss: 0.41639018058776855
    

     44%|████████████████████████████████████                                             | 89/200 [00:32<00:32,  3.43it/s]

    Epoch: 88 | critic_loss: -0.2638269066810608 | gen_loss: 0.40709614753723145
    

     45%|████████████████████████████████████▍                                            | 90/200 [00:33<00:31,  3.44it/s]

    Epoch: 89 | critic_loss: -0.24647468328475952 | gen_loss: 0.3741465210914612
    

     46%|████████████████████████████████████▊                                            | 91/200 [00:33<00:31,  3.44it/s]

    Epoch: 90 | critic_loss: -0.24450160562992096 | gen_loss: 0.3891570270061493
    

     46%|█████████████████████████████████████▎                                           | 92/200 [00:33<00:31,  3.44it/s]

    Epoch: 91 | critic_loss: -0.25481364130973816 | gen_loss: 0.4017965793609619
    

     46%|█████████████████████████████████████▋                                           | 93/200 [00:33<00:31,  3.44it/s]

    Epoch: 92 | critic_loss: -0.28516602516174316 | gen_loss: 0.4508054256439209
    

     47%|██████████████████████████████████████                                           | 94/200 [00:34<00:31,  3.41it/s]

    Epoch: 93 | critic_loss: -0.1940636783838272 | gen_loss: 0.39802244305610657
    

     48%|██████████████████████████████████████▍                                          | 95/200 [00:34<00:31,  3.36it/s]

    Epoch: 94 | critic_loss: -0.23319678008556366 | gen_loss: 0.3516239523887634
    

     48%|██████████████████████████████████████▉                                          | 96/200 [00:34<00:31,  3.32it/s]

    Epoch: 95 | critic_loss: -0.2214893102645874 | gen_loss: 0.3705796003341675
    

     48%|███████████████████████████████████████▎                                         | 97/200 [00:35<00:30,  3.33it/s]

    Epoch: 96 | critic_loss: -0.2419338971376419 | gen_loss: 0.416594535112381
    

     49%|███████████████████████████████████████▋                                         | 98/200 [00:35<00:30,  3.37it/s]

    Epoch: 97 | critic_loss: -0.278506875038147 | gen_loss: 0.44215062260627747
    

     50%|████████████████████████████████████████                                         | 99/200 [00:35<00:29,  3.40it/s]

    Epoch: 98 | critic_loss: -0.23439666628837585 | gen_loss: 0.3692026138305664
    

     50%|████████████████████████████████████████                                        | 100/200 [00:35<00:29,  3.40it/s]

    Epoch: 99 | critic_loss: -0.22140789031982422 | gen_loss: 0.3604203760623932
    

     50%|████████████████████████████████████████▍                                       | 101/200 [00:36<00:29,  3.36it/s]

    Epoch: 100 | critic_loss: -0.2289312332868576 | gen_loss: 0.3572380542755127
    

     51%|████████████████████████████████████████▊                                       | 102/200 [00:36<00:28,  3.40it/s]

    Epoch: 101 | critic_loss: -0.23688626289367676 | gen_loss: 0.4007105231285095
    

     52%|█████████████████████████████████████████▏                                      | 103/200 [00:36<00:28,  3.42it/s]

    Epoch: 102 | critic_loss: -0.22634679079055786 | gen_loss: 0.36621057987213135
    

     52%|█████████████████████████████████████████▌                                      | 104/200 [00:37<00:28,  3.41it/s]

    Epoch: 103 | critic_loss: -0.24656540155410767 | gen_loss: 0.3869836926460266
    

     52%|██████████████████████████████████████████                                      | 105/200 [00:37<00:27,  3.40it/s]

    Epoch: 104 | critic_loss: -0.2521984279155731 | gen_loss: 0.3999665081501007
    

     53%|██████████████████████████████████████████▍                                     | 106/200 [00:37<00:27,  3.38it/s]

    Epoch: 105 | critic_loss: -0.2712993919849396 | gen_loss: 0.4148338735103607
    

     54%|██████████████████████████████████████████▊                                     | 107/200 [00:38<00:27,  3.33it/s]

    Epoch: 106 | critic_loss: -0.254154771566391 | gen_loss: 0.38774532079696655
    

     54%|███████████████████████████████████████████▏                                    | 108/200 [00:38<00:27,  3.34it/s]

    Epoch: 107 | critic_loss: -0.22105593979358673 | gen_loss: 0.37799692153930664
    

     55%|███████████████████████████████████████████▌                                    | 109/200 [00:38<00:27,  3.34it/s]

    Epoch: 108 | critic_loss: -0.27111756801605225 | gen_loss: 0.3996857702732086
    

     55%|████████████████████████████████████████████                                    | 110/200 [00:38<00:26,  3.35it/s]

    Epoch: 109 | critic_loss: -0.23431813716888428 | gen_loss: 0.37395644187927246
    

     56%|████████████████████████████████████████████▍                                   | 111/200 [00:39<00:27,  3.26it/s]

    Epoch: 110 | critic_loss: -0.26009538769721985 | gen_loss: 0.3903313875198364
    

     56%|████████████████████████████████████████████▊                                   | 112/200 [00:39<00:26,  3.29it/s]

    Epoch: 111 | critic_loss: -0.26801806688308716 | gen_loss: 0.40040072798728943
    

     56%|█████████████████████████████████████████████▏                                  | 113/200 [00:39<00:26,  3.34it/s]

    Epoch: 112 | critic_loss: -0.2371458113193512 | gen_loss: 0.3767924904823303
    

     57%|█████████████████████████████████████████████▌                                  | 114/200 [00:40<00:25,  3.37it/s]

    Epoch: 113 | critic_loss: -0.2652629017829895 | gen_loss: 0.3943062722682953
    

     57%|██████████████████████████████████████████████                                  | 115/200 [00:40<00:25,  3.38it/s]

    Epoch: 114 | critic_loss: -0.25195518136024475 | gen_loss: 0.3877734839916229
    

     58%|██████████████████████████████████████████████▍                                 | 116/200 [00:40<00:24,  3.38it/s]

    Epoch: 115 | critic_loss: -0.26854845881462097 | gen_loss: 0.4060291051864624
    

     58%|██████████████████████████████████████████████▊                                 | 117/200 [00:41<00:24,  3.39it/s]

    Epoch: 116 | critic_loss: -0.24332481622695923 | gen_loss: 0.36277273297309875
    

     59%|███████████████████████████████████████████████▏                                | 118/200 [00:41<00:24,  3.33it/s]

    Epoch: 117 | critic_loss: -0.2571922540664673 | gen_loss: 0.38936060667037964
    

     60%|███████████████████████████████████████████████▌                                | 119/200 [00:41<00:24,  3.29it/s]

    Epoch: 118 | critic_loss: -0.2639597952365875 | gen_loss: 0.38300246000289917
    

     60%|████████████████████████████████████████████████                                | 120/200 [00:41<00:24,  3.33it/s]

    Epoch: 119 | critic_loss: -0.23315274715423584 | gen_loss: 0.35862717032432556
    

     60%|████████████████████████████████████████████████▍                               | 121/200 [00:42<00:22,  3.52it/s]

    Epoch: 120 | critic_loss: -0.24278020858764648 | gen_loss: 0.35789376497268677
    

     61%|████████████████████████████████████████████████▊                               | 122/200 [00:42<00:21,  3.66it/s]

    Epoch: 121 | critic_loss: -0.21571536362171173 | gen_loss: 0.3576071262359619
    

     62%|█████████████████████████████████████████████████▏                              | 123/200 [00:42<00:20,  3.73it/s]

    Epoch: 122 | critic_loss: -0.251568466424942 | gen_loss: 0.39690518379211426
    

     62%|█████████████████████████████████████████████████▌                              | 124/200 [00:42<00:19,  3.84it/s]

    Epoch: 123 | critic_loss: -0.21602106094360352 | gen_loss: 0.3667871952056885
    

     62%|██████████████████████████████████████████████████                              | 125/200 [00:43<00:19,  3.92it/s]

    Epoch: 124 | critic_loss: -0.20176321268081665 | gen_loss: 0.36355531215667725
    

     63%|██████████████████████████████████████████████████▍                             | 126/200 [00:43<00:18,  3.98it/s]

    Epoch: 125 | critic_loss: -0.24002374708652496 | gen_loss: 0.3674788177013397
    

     64%|██████████████████████████████████████████████████▊                             | 127/200 [00:43<00:18,  4.01it/s]

    Epoch: 126 | critic_loss: -0.23060013353824615 | gen_loss: 0.35755252838134766
    

     64%|███████████████████████████████████████████████████▏                            | 128/200 [00:43<00:17,  4.04it/s]

    Epoch: 127 | critic_loss: -0.23299278318881989 | gen_loss: 0.3673613667488098
    

     64%|███████████████████████████████████████████████████▌                            | 129/200 [00:44<00:17,  4.05it/s]

    Epoch: 128 | critic_loss: -0.2497032880783081 | gen_loss: 0.38848063349723816
    

     65%|████████████████████████████████████████████████████                            | 130/200 [00:44<00:17,  3.97it/s]

    Epoch: 129 | critic_loss: -0.2657921314239502 | gen_loss: 0.38932669162750244
    

     66%|████████████████████████████████████████████████████▍                           | 131/200 [00:44<00:17,  3.91it/s]

    Epoch: 130 | critic_loss: -0.2513227164745331 | gen_loss: 0.3647608757019043
    

     66%|████████████████████████████████████████████████████▊                           | 132/200 [00:44<00:17,  3.83it/s]

    Epoch: 131 | critic_loss: -0.25593090057373047 | gen_loss: 0.38305535912513733
    

     66%|█████████████████████████████████████████████████████▏                          | 133/200 [00:45<00:17,  3.78it/s]

    Epoch: 132 | critic_loss: -0.23889058828353882 | gen_loss: 0.3352544605731964
    

     67%|█████████████████████████████████████████████████████▌                          | 134/200 [00:45<00:17,  3.75it/s]

    Epoch: 133 | critic_loss: -0.21322152018547058 | gen_loss: 0.35588937997817993
    

     68%|██████████████████████████████████████████████████████                          | 135/200 [00:45<00:17,  3.72it/s]

    Epoch: 134 | critic_loss: -0.24366207420825958 | gen_loss: 0.36271294951438904
    

     68%|██████████████████████████████████████████████████████▍                         | 136/200 [00:46<00:17,  3.70it/s]

    Epoch: 135 | critic_loss: -0.22219790518283844 | gen_loss: 0.33838164806365967
    

     68%|██████████████████████████████████████████████████████▊                         | 137/200 [00:46<00:17,  3.67it/s]

    Epoch: 136 | critic_loss: -0.2340601235628128 | gen_loss: 0.3847951292991638
    

     69%|███████████████████████████████████████████████████████▏                        | 138/200 [00:46<00:16,  3.66it/s]

    Epoch: 137 | critic_loss: -0.2620770335197449 | gen_loss: 0.3684349060058594
    

     70%|███████████████████████████████████████████████████████▌                        | 139/200 [00:46<00:16,  3.61it/s]

    Epoch: 138 | critic_loss: -0.2637593150138855 | gen_loss: 0.3788525462150574
    

     70%|████████████████████████████████████████████████████████                        | 140/200 [00:47<00:16,  3.56it/s]

    Epoch: 139 | critic_loss: -0.2961990535259247 | gen_loss: 0.4127715229988098
    

     70%|████████████████████████████████████████████████████████▍                       | 141/200 [00:47<00:16,  3.60it/s]

    Epoch: 140 | critic_loss: -0.2552230656147003 | gen_loss: 0.373776912689209
    

     71%|████████████████████████████████████████████████████████▊                       | 142/200 [00:47<00:15,  3.69it/s]

    Epoch: 141 | critic_loss: -0.24516622722148895 | gen_loss: 0.34251007437705994
    

     72%|█████████████████████████████████████████████████████████▏                      | 143/200 [00:47<00:15,  3.78it/s]

    Epoch: 142 | critic_loss: -0.26412978768348694 | gen_loss: 0.3603253960609436
    

     72%|█████████████████████████████████████████████████████████▌                      | 144/200 [00:48<00:14,  3.86it/s]

    Epoch: 143 | critic_loss: -0.24944782257080078 | gen_loss: 0.35506677627563477
    

     72%|██████████████████████████████████████████████████████████                      | 145/200 [00:48<00:14,  3.89it/s]

    Epoch: 144 | critic_loss: -0.24392768740653992 | gen_loss: 0.3511508107185364
    

     73%|██████████████████████████████████████████████████████████▍                     | 146/200 [00:48<00:13,  3.94it/s]

    Epoch: 145 | critic_loss: -0.2316821664571762 | gen_loss: 0.30226296186447144
    

     74%|██████████████████████████████████████████████████████████▊                     | 147/200 [00:48<00:13,  3.98it/s]

    Epoch: 146 | critic_loss: -0.2833584249019623 | gen_loss: 0.3963167071342468
    

     74%|███████████████████████████████████████████████████████████▏                    | 148/200 [00:49<00:13,  3.96it/s]

    Epoch: 147 | critic_loss: -0.2919508218765259 | gen_loss: 0.40546655654907227
    

     74%|███████████████████████████████████████████████████████████▌                    | 149/200 [00:49<00:12,  4.00it/s]

    Epoch: 148 | critic_loss: -0.2800157070159912 | gen_loss: 0.3772668242454529
    

     75%|████████████████████████████████████████████████████████████                    | 150/200 [00:49<00:12,  3.99it/s]

    Epoch: 149 | critic_loss: -0.2720150649547577 | gen_loss: 0.37494733929634094
    

     76%|████████████████████████████████████████████████████████████▍                   | 151/200 [00:49<00:12,  4.02it/s]

    Epoch: 150 | critic_loss: -0.2640763521194458 | gen_loss: 0.3851677179336548
    

     76%|████████████████████████████████████████████████████████████▊                   | 152/200 [00:50<00:11,  4.03it/s]

    Epoch: 151 | critic_loss: -0.28176337480545044 | gen_loss: 0.3738551437854767
    

     76%|█████████████████████████████████████████████████████████████▏                  | 153/200 [00:50<00:11,  3.97it/s]

    Epoch: 152 | critic_loss: -0.29163628816604614 | gen_loss: 0.35690486431121826
    

     77%|█████████████████████████████████████████████████████████████▌                  | 154/200 [00:50<00:11,  4.00it/s]

    Epoch: 153 | critic_loss: -0.23395287990570068 | gen_loss: 0.2983979880809784
    

     78%|██████████████████████████████████████████████████████████████                  | 155/200 [00:50<00:11,  4.01it/s]

    Epoch: 154 | critic_loss: -0.29053735733032227 | gen_loss: 0.40300220251083374
    

     78%|██████████████████████████████████████████████████████████████▍                 | 156/200 [00:51<00:10,  4.02it/s]

    Epoch: 155 | critic_loss: -0.2815514802932739 | gen_loss: 0.3668445348739624
    

     78%|██████████████████████████████████████████████████████████████▊                 | 157/200 [00:51<00:10,  4.05it/s]

    Epoch: 156 | critic_loss: -0.307656466960907 | gen_loss: 0.4336288571357727
    

     79%|███████████████████████████████████████████████████████████████▏                | 158/200 [00:51<00:10,  4.07it/s]

    Epoch: 157 | critic_loss: -0.26327210664749146 | gen_loss: 0.3497222363948822
    

     80%|███████████████████████████████████████████████████████████████▌                | 159/200 [00:51<00:10,  4.05it/s]

    Epoch: 158 | critic_loss: -0.23013195395469666 | gen_loss: 0.32780539989471436
    

     80%|████████████████████████████████████████████████████████████████                | 160/200 [00:52<00:10,  3.99it/s]

    Epoch: 159 | critic_loss: -0.2967003881931305 | gen_loss: 0.4147905707359314
    

     80%|████████████████████████████████████████████████████████████████▍               | 161/200 [00:52<00:09,  4.03it/s]

    Epoch: 160 | critic_loss: -0.2615257799625397 | gen_loss: 0.3410722315311432
    

     81%|████████████████████████████████████████████████████████████████▊               | 162/200 [00:52<00:09,  4.05it/s]

    Epoch: 161 | critic_loss: -0.255295991897583 | gen_loss: 0.31429529190063477
    

     82%|█████████████████████████████████████████████████████████████████▏              | 163/200 [00:52<00:09,  4.02it/s]

    Epoch: 162 | critic_loss: -0.2733691334724426 | gen_loss: 0.3679298758506775
    

     82%|█████████████████████████████████████████████████████████████████▌              | 164/200 [00:53<00:08,  4.02it/s]

    Epoch: 163 | critic_loss: -0.2262057363986969 | gen_loss: 0.21206404268741608
    

     82%|██████████████████████████████████████████████████████████████████              | 165/200 [00:53<00:08,  3.95it/s]

    Epoch: 164 | critic_loss: -0.24785682559013367 | gen_loss: 0.31219327449798584
    

     83%|██████████████████████████████████████████████████████████████████▍             | 166/200 [00:53<00:08,  3.90it/s]

    Epoch: 165 | critic_loss: -0.30123722553253174 | gen_loss: 0.42210543155670166
    

     84%|██████████████████████████████████████████████████████████████████▊             | 167/200 [00:54<00:08,  3.84it/s]

    Epoch: 166 | critic_loss: -0.2758980691432953 | gen_loss: 0.38147351145744324
    

     84%|███████████████████████████████████████████████████████████████████▏            | 168/200 [00:54<00:08,  3.80it/s]

    Epoch: 167 | critic_loss: -0.316704124212265 | gen_loss: 0.39698055386543274
    

     84%|███████████████████████████████████████████████████████████████████▌            | 169/200 [00:54<00:08,  3.76it/s]

    Epoch: 168 | critic_loss: -0.3260529041290283 | gen_loss: 0.3972119688987732
    

     85%|████████████████████████████████████████████████████████████████████            | 170/200 [00:54<00:08,  3.73it/s]

    Epoch: 169 | critic_loss: -0.31215986609458923 | gen_loss: 0.38455331325531006
    

     86%|████████████████████████████████████████████████████████████████████▍           | 171/200 [00:55<00:07,  3.70it/s]

    Epoch: 170 | critic_loss: -0.3706111013889313 | gen_loss: 0.4608657956123352
    

     86%|████████████████████████████████████████████████████████████████████▊           | 172/200 [00:55<00:07,  3.65it/s]

    Epoch: 171 | critic_loss: -0.310491681098938 | gen_loss: 0.40522217750549316
    

     86%|█████████████████████████████████████████████████████████████████████▏          | 173/200 [00:55<00:07,  3.59it/s]

    Epoch: 172 | critic_loss: -0.3513323962688446 | gen_loss: 0.5187121033668518
    

     87%|█████████████████████████████████████████████████████████████████████▌          | 174/200 [00:55<00:07,  3.58it/s]

    Epoch: 173 | critic_loss: -0.24815236032009125 | gen_loss: 0.3275775909423828
    

     88%|██████████████████████████████████████████████████████████████████████          | 175/200 [00:56<00:06,  3.57it/s]

    Epoch: 174 | critic_loss: -0.28646451234817505 | gen_loss: 0.39188483357429504
    

     88%|██████████████████████████████████████████████████████████████████████▍         | 176/200 [00:56<00:06,  3.56it/s]

    Epoch: 175 | critic_loss: -0.27517643570899963 | gen_loss: 0.34296196699142456
    

     88%|██████████████████████████████████████████████████████████████████████▊         | 177/200 [00:56<00:06,  3.55it/s]

    Epoch: 176 | critic_loss: -0.31533291935920715 | gen_loss: 0.41212502121925354
    

     89%|███████████████████████████████████████████████████████████████████████▏        | 178/200 [00:57<00:06,  3.54it/s]

    Epoch: 177 | critic_loss: -0.31115561723709106 | gen_loss: 0.3898743689060211
    

     90%|███████████████████████████████████████████████████████████████████████▌        | 179/200 [00:57<00:05,  3.53it/s]

    Epoch: 178 | critic_loss: -0.31811240315437317 | gen_loss: 0.39330852031707764
    

     90%|████████████████████████████████████████████████████████████████████████        | 180/200 [00:57<00:05,  3.47it/s]

    Epoch: 179 | critic_loss: -0.296848863363266 | gen_loss: 0.38605380058288574
    

     90%|████████████████████████████████████████████████████████████████████████▍       | 181/200 [00:57<00:05,  3.48it/s]

    Epoch: 180 | critic_loss: -0.29916495084762573 | gen_loss: 0.40815067291259766
    

     91%|████████████████████████████████████████████████████████████████████████▊       | 182/200 [00:58<00:05,  3.49it/s]

    Epoch: 181 | critic_loss: -0.27015480399131775 | gen_loss: 0.35915547609329224
    

     92%|█████████████████████████████████████████████████████████████████████████▏      | 183/200 [00:58<00:04,  3.49it/s]

    Epoch: 182 | critic_loss: -0.2670876383781433 | gen_loss: 0.35377103090286255
    

     92%|█████████████████████████████████████████████████████████████████████████▌      | 184/200 [00:58<00:04,  3.47it/s]

    Epoch: 183 | critic_loss: -0.28833794593811035 | gen_loss: 0.3717002868652344
    

     92%|██████████████████████████████████████████████████████████████████████████      | 185/200 [00:59<00:04,  3.43it/s]

    Epoch: 184 | critic_loss: -0.26421743631362915 | gen_loss: 0.3495456576347351
    

     93%|██████████████████████████████████████████████████████████████████████████▍     | 186/200 [00:59<00:04,  3.36it/s]

    Epoch: 185 | critic_loss: -0.25993645191192627 | gen_loss: 0.37321561574935913
    

     94%|██████████████████████████████████████████████████████████████████████████▊     | 187/200 [00:59<00:03,  3.39it/s]

    Epoch: 186 | critic_loss: -0.3285483717918396 | gen_loss: 0.41389790177345276
    

     94%|███████████████████████████████████████████████████████████████████████████▏    | 188/200 [01:00<00:03,  3.39it/s]

    Epoch: 187 | critic_loss: -0.28951331973075867 | gen_loss: 0.3579261004924774
    

     94%|███████████████████████████████████████████████████████████████████████████▌    | 189/200 [01:00<00:03,  3.36it/s]

    Epoch: 188 | critic_loss: -0.32801833748817444 | gen_loss: 0.41400888562202454
    

     95%|████████████████████████████████████████████████████████████████████████████    | 190/200 [01:00<00:02,  3.38it/s]

    Epoch: 189 | critic_loss: -0.2878578007221222 | gen_loss: 0.36097797751426697
    

     96%|████████████████████████████████████████████████████████████████████████████▍   | 191/200 [01:00<00:02,  3.40it/s]

    Epoch: 190 | critic_loss: -0.274676650762558 | gen_loss: 0.34725967049598694
    

     96%|████████████████████████████████████████████████████████████████████████████▊   | 192/200 [01:01<00:02,  3.42it/s]

    Epoch: 191 | critic_loss: -0.3022336959838867 | gen_loss: 0.39650416374206543
    

     96%|█████████████████████████████████████████████████████████████████████████████▏  | 193/200 [01:01<00:02,  3.40it/s]

    Epoch: 192 | critic_loss: -0.29000529646873474 | gen_loss: 0.37750357389450073
    

     97%|█████████████████████████████████████████████████████████████████████████████▌  | 194/200 [01:01<00:01,  3.38it/s]

    Epoch: 193 | critic_loss: -0.24224022030830383 | gen_loss: 0.3046799302101135
    

     98%|██████████████████████████████████████████████████████████████████████████████  | 195/200 [01:02<00:01,  3.40it/s]

    Epoch: 194 | critic_loss: -0.286218523979187 | gen_loss: 0.3520301580429077
    

     98%|██████████████████████████████████████████████████████████████████████████████▍ | 196/200 [01:02<00:01,  3.41it/s]

    Epoch: 195 | critic_loss: -0.3064529299736023 | gen_loss: 0.38745638728141785
    

     98%|██████████████████████████████████████████████████████████████████████████████▊ | 197/200 [01:02<00:00,  3.33it/s]

    Epoch: 196 | critic_loss: -0.3627226948738098 | gen_loss: 0.5069276094436646
    

     99%|███████████████████████████████████████████████████████████████████████████████▏| 198/200 [01:02<00:00,  3.36it/s]

    Epoch: 197 | critic_loss: -0.31205806136131287 | gen_loss: 0.429940789937973
    

    100%|███████████████████████████████████████████████████████████████████████████████▌| 199/200 [01:03<00:00,  3.40it/s]

    Epoch: 198 | critic_loss: -0.28414756059646606 | gen_loss: 0.3612818419933319
    

    100%|████████████████████████████████████████████████████████████████████████████████| 200/200 [01:03<00:00,  3.15it/s]

    Epoch: 199 | critic_loss: -0.2664364278316498 | gen_loss: 0.3359828591346741
    

    
    

이제 훈련된 모델과 데이터를 합성할 것입니다. 예를 들어, 모델에서 100000개의 샘플 데이터를 합성할 수 있습니다.


```python
synth_data = synthesizer.sample(X_train_synth_min[['FraudFound_P']])

while synth_data.shape[0] <= 100000:
    temp = synthesizer.sample(X_train_synth_min[['FraudFound_P']])
    synth_data = pd.concat([synth_data, temp], axis=0)
```


```python
print(synth_data.shape)
synth_data.sample(50)
```

    (100130, 44)
    




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
      <th>Age</th>
      <th>Deductible</th>
      <th>DriverRating</th>
      <th>AccidentArea_Urban</th>
      <th>Sex_Male</th>
      <th>MaritalStatus_Married</th>
      <th>MaritalStatus_Single</th>
      <th>MaritalStatus_Widow</th>
      <th>Fault_Third Party</th>
      <th>PolicyType_Sedan - Collision</th>
      <th>...</th>
      <th>AgeOfVehicle_3 years</th>
      <th>AgeOfVehicle_4 years</th>
      <th>AgeOfVehicle_5 years</th>
      <th>AgeOfVehicle_6 years</th>
      <th>AgeOfVehicle_7 years</th>
      <th>AgeOfVehicle_more than 7</th>
      <th>AgeOfVehicle_new</th>
      <th>BasePolicy_Collision</th>
      <th>BasePolicy_Liability</th>
      <th>FraudFound_P</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>135</th>
      <td>15</td>
      <td>313</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>397</th>
      <td>19</td>
      <td>276</td>
      <td>-1</td>
      <td>255</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>631</th>
      <td>17</td>
      <td>370</td>
      <td>-1</td>
      <td>255</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>255</td>
      <td>255</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>597</th>
      <td>19</td>
      <td>350</td>
      <td>-1</td>
      <td>255</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>255</td>
      <td>255</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>462</th>
      <td>22</td>
      <td>330</td>
      <td>-1</td>
      <td>255</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>18</td>
      <td>350</td>
      <td>-1</td>
      <td>255</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>255</td>
      <td>255</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>185</th>
      <td>19</td>
      <td>355</td>
      <td>-1</td>
      <td>255</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>387</th>
      <td>19</td>
      <td>293</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>496</th>
      <td>29</td>
      <td>323</td>
      <td>-1</td>
      <td>255</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>469</th>
      <td>10</td>
      <td>288</td>
      <td>-1</td>
      <td>255</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>500</th>
      <td>43</td>
      <td>382</td>
      <td>-2</td>
      <td>254</td>
      <td>0</td>
      <td>254</td>
      <td>1</td>
      <td>0</td>
      <td>254</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>255</td>
      <td>0</td>
      <td>0</td>
      <td>255</td>
      <td>255</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>34</th>
      <td>27</td>
      <td>285</td>
      <td>-1</td>
      <td>255</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>104</th>
      <td>14</td>
      <td>389</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>28</th>
      <td>16</td>
      <td>291</td>
      <td>-1</td>
      <td>255</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>222</th>
      <td>12</td>
      <td>302</td>
      <td>-1</td>
      <td>255</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>593</th>
      <td>25</td>
      <td>295</td>
      <td>-1</td>
      <td>255</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>362</th>
      <td>27</td>
      <td>344</td>
      <td>-1</td>
      <td>255</td>
      <td>0</td>
      <td>255</td>
      <td>1</td>
      <td>0</td>
      <td>255</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>255</td>
      <td>255</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>20</th>
      <td>33</td>
      <td>315</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>629</th>
      <td>12</td>
      <td>352</td>
      <td>-1</td>
      <td>255</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>255</td>
      <td>255</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>274</th>
      <td>25</td>
      <td>358</td>
      <td>-1</td>
      <td>255</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>0</td>
      <td>255</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>255</td>
      <td>255</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>313</th>
      <td>18</td>
      <td>314</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>72</th>
      <td>16</td>
      <td>353</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>407</th>
      <td>18</td>
      <td>328</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>449</th>
      <td>23</td>
      <td>366</td>
      <td>-1</td>
      <td>255</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>255</td>
      <td>255</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>371</th>
      <td>27</td>
      <td>293</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>520</th>
      <td>20</td>
      <td>272</td>
      <td>-1</td>
      <td>255</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>458</th>
      <td>22</td>
      <td>361</td>
      <td>-1</td>
      <td>255</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>255</td>
      <td>255</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>170</th>
      <td>18</td>
      <td>326</td>
      <td>-1</td>
      <td>0</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>382</th>
      <td>23</td>
      <td>387</td>
      <td>-1</td>
      <td>255</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>255</td>
      <td>255</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>183</th>
      <td>22</td>
      <td>381</td>
      <td>-2</td>
      <td>255</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>0</td>
      <td>255</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>255</td>
      <td>255</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>546</th>
      <td>31</td>
      <td>348</td>
      <td>-1</td>
      <td>255</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>255</td>
      <td>255</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>51</th>
      <td>23</td>
      <td>332</td>
      <td>-1</td>
      <td>255</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>111</th>
      <td>19</td>
      <td>345</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>486</th>
      <td>18</td>
      <td>295</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>493</th>
      <td>24</td>
      <td>328</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>253</th>
      <td>28</td>
      <td>411</td>
      <td>-1</td>
      <td>255</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>255</td>
      <td>255</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>398</th>
      <td>15</td>
      <td>377</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>390</th>
      <td>18</td>
      <td>357</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>370</th>
      <td>22</td>
      <td>357</td>
      <td>-1</td>
      <td>255</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>36</th>
      <td>24</td>
      <td>306</td>
      <td>-1</td>
      <td>255</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>244</th>
      <td>10</td>
      <td>290</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>68</th>
      <td>22</td>
      <td>375</td>
      <td>-1</td>
      <td>255</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>255</td>
      <td>255</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>548</th>
      <td>17</td>
      <td>332</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>562</th>
      <td>14</td>
      <td>356</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>149</th>
      <td>24</td>
      <td>330</td>
      <td>-1</td>
      <td>255</td>
      <td>0</td>
      <td>254</td>
      <td>0</td>
      <td>0</td>
      <td>254</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>255</td>
      <td>255</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>139</th>
      <td>12</td>
      <td>299</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>310</th>
      <td>18</td>
      <td>294</td>
      <td>-1</td>
      <td>255</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>158</th>
      <td>24</td>
      <td>302</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>84</th>
      <td>28</td>
      <td>376</td>
      <td>-1</td>
      <td>255</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>230</th>
      <td>30</td>
      <td>280</td>
      <td>-1</td>
      <td>0</td>
      <td>0</td>
      <td>255</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>50 rows × 44 columns</p>
</div>



데이터가 합성되면 이전의 훈련 데이터를 합성 데이터로 채우고, 데이터 세트의 균형을 맞춥니다.


```python
minority_synth_data = synth_data[synth_data["FraudFound_P"] == 1].sample(9502)

X_train_synth_true = pd.concat([X_train_synth, minority_synth_data]).reset_index(drop=True).copy()

X_train_synth_true['FraudFound_P'].value_counts()
```




    0    10148
    1    10148
    Name: FraudFound_P, dtype: int64



위의 예에서 볼 수 있듯이 이전의 imblanced 데이터 세트와 현재의 합성 데이터를 통해서 균형을 맞출 수 있습니다. balanced 데이터로 모델 성능을 훈련하는 것을 살펴보겠습니다.


```python
model.fit(X_train_synth_true.drop('FraudFound_P', axis =1), X_train_synth_true['FraudFound_P'])
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

                  precision    recall  f1-score   support
    
               0       0.94      0.98      0.96      4349
               1       0.17      0.05      0.08       277
    
        accuracy                           0.93      4626
       macro avg       0.56      0.52      0.52      4626
    weighted avg       0.90      0.93      0.91      4626
    
    

원본 데이터와 비교했을 때, balanced 데이터 세트로 모델 성능이 약간 향상되었습니다. features를 제대로 선택하지 않았고, 다른 모델로 실험해보지 않았기 때문에 그다지 성능이 증가하지는 않았습니다. 하지만, 이 간단한 예로 합성 데이터가 fraud 모델링 성능을 높이는 데 도움이 될 수 있음을 증명했습니다.

# Conclusion
Fraud는 비즈니스에서 흔히 발생하는 문제이며 해당 사례를 처리하기 위해 적절한 조치가 필요합니다. 우리가 할 수 있는 조치 중 하나는 fraud 모델링을 사용하여 fraud cases를 정확하게 예측하는 것입니다. 그러나 fraud 모델링 개발은 imbalanced 데이터 문제로 인해 자주 지연됩니다.

다양한 연구에서 합성 데이터가 imbalance 문제를 완화하기 위해 훈련 데이터의 balance을 조정함으로써 모델 성능을 향상시킬 수 있음이 입증되었습니다.

fraud 모델을 개발하는 간단한 실험에서 합성 데이터를 포함하는 balanced 데이터 세트는 원본 데이터 세트보다 약간 더 나은 성능을 보였습니다. 이는 합성 데이터가 fraud 모델링 사례에 도움이 될 수 있음을 증명합니다.
