---
layout: single
title:  "[Feedback Prize - English Language Learning] DeBERTa LayerwiseLR LastLayerReinit in TensorFlow"
categories: Kaggle
tag: [Feedback Prize - English Language Learning]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: false
---

# DeBERTa LLRD + LastLayerReinit with TensorFlow
- MultilabelStratifiedKFold split of the data

- HuggingFace DeBERTaV3 pre-trained model finetuning with Tensorflow

- WeightedLayerPool + MeanPool TensorFlow implementation

- Layer-wise learning rate decay

- Last layer reinitialization or partially reinitialzation


# Imports


```python
import os, gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
print(f'TF version: {tf.__version__}')
import tensorflow_addons as tfa
from tensorflow.keras import layers

import transformers
print(f'transformers version: {transformers.__version__}')
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

import sys
sys.path.append('../input/iterativestratification')
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
```

    TF version: 2.6.4
    transformers version: 4.20.1
    


```python
def set_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
set_seed(42)
```

# Load DataFrame


```python
df = pd.read_csv("../input/feedback-prize-english-language-learning/train.csv")
df.head()
print('\n---------DataFrame Summary---------')
df.info()
```

    
    ---------DataFrame Summary---------
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3911 entries, 0 to 3910
    Data columns (total 8 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   text_id      3911 non-null   object 
     1   full_text    3911 non-null   object 
     2   cohesion     3911 non-null   float64
     3   syntax       3911 non-null   float64
     4   vocabulary   3911 non-null   float64
     5   phraseology  3911 non-null   float64
     6   grammar      3911 non-null   float64
     7   conventions  3911 non-null   float64
    dtypes: float64(6), object(2)
    memory usage: 244.6+ KB
    

# CV Split


```python
N_FOLD = 5
TARGET_COLS = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
skf = MultilabelStratifiedKFold(n_splits=N_FOLD, shuffle=True, random_state=42)

for n, (train_index, val_index) in enumerate(skf.split(df, df[TARGET_COLS])):
    df.loc[val_index, "fold"] = n

df["fold"] = df["fold"].astype(int)
df["fold"].value_counts()
```




    1    783
    0    782
    4    782
    3    782
    2    782
    Name: fold, dtype: int64




```python
df.to_csv("df_folds.csv", index=False)
```

# Model Config


```python
MAX_LENGTH = 512
BATCH_SIZE = 4
DEBERTA_MODEL = "../input/debertav3base"
```

regression task에서 dropout을 비활성화 시켜야 하는 이유: [discussion](https://www.kaggle.com/competitions/commonlitreadabilityprize/discussion/260729)


```python
tokenizer = transformers.AutoTokenizer.from_pretrained(DEBERTA_MODEL)
tokenizer.save_pretrained("./tokenizer")

cfg = transformers.AutoConfig.from_pretrained(DEBERTA_MODEL, output_hidden_states=True)
cfg.hidden_dropout_prob = 0
cfg.attention_probs_dropout_prob = 0
cfg.save_pretrained("./tokenizer")
```

    /opt/conda/lib/python3.7/site-packages/transformers/convert_slow_tokenizer.py:435: UserWarning: The sentencepiece tokenizer that you are converting to a fast tokenizer uses the byte fallback option which is not implemented in the fast tokenizers. In practice this means that the fast version of the tokenizer can produce unknown tokens whereas the sentencepiece version would have converted these unknown tokens into a sequence of byte tokens matching the original piece of text.
      "The sentencepiece tokenizer that you are converting to a fast tokenizer uses the byte fallback option"
    

# Data Process Function
HugggingFace DeBERTa 모델을 사용하려면 pre-training된 DeBERTa 모델이 요구하는 대로 input text를 토큰화해야 합니다.


```python
def deberta_encode(texts, tokenizer=tokenizer):
    input_ids = []
    attention_mask = []
    for text in texts.tolist():
        token = tokenizer(text,
                          max_length=MAX_LENGTH,
                          return_attention_mask=True,
                          return_tensors="np",
                          truncation=True,
                          padding="max_length")
        input_ids.append(token["input_ids"][0])
        attention_mask.append(token["attention_mask"][0])
        
    return np.array(input_ids, dtype="int32"), np.array(attention_mask, dtype="int32")
```


```python
def get_dataset(df):
    inputs = deberta_encode(df["full_text"])
    targets = np.array(df[TARGET_COLS], dtype="float32")
    return inputs, targets
```

# Model

## MeanPool
[CLS] token을 사용하는 대신 padding token을 masking하여 sequence axis를 따라 hidden state의 한 layer를 평균화하는 MeanPool method을 사용합니다.

## WeightedLayerPool
WeightedLayerPool은 training 가능한 weight set를 사용하여 transformer backbone에서 hidden state 세트를 평균화합니다. 여기서는 이것을 구현하기 위해 constraint가 있는 Dense layer를 사용합니다.


```python
class MeanPool(tf.keras.layers.Layer):
    def call(self, inputs, mask=None):
        # inputs: (None, 512, 768)
        
        # (None, 512, 1)
        broadcast_mask = tf.expand_dims(tf.cast(mask, "float32"), -1)
        
        # (None, 768)
        embedding_sum = tf.reduce_sum(inputs * broadcast_mask, axis=1)
        
        # (None, 1)
        mask_sum = tf.reduce_sum(broadcast_mask, axis=1)
        
        mask_sum = tf.math.maximum(mask_sum, tf.constant([1e-9]))
        return embedding_sum / mask_sum
```

WeightedLayerPool weights constraints: sum(w)을 1로 만들기 위한 softmax


```python
class WeightsSumOne(tf.keras.constraints.Constraint):
    def __call__(self, w):
        return tf.nn.softmax(w, axis=0)
```

## Model Design Choice
final representation을 얻는 방법에는 여러 가지가 있지만 DeBERTa의 마지막 4개 layer hidden state를 선택하고, 그 중 MeanPool을 사용하여 sequence axis를 따라 정보를 수집한 다음, training 가능한 weight set와 함께 WeightedLayerPool을 사용하여 model의 depth axis를 따라 정보를 수집합니다. 마지막으로는 regression head

## Last Layer Reinitialization
마지막 transformer encoder block의 reinitialization: Dense kerenl을 위한 GlorotUniform, Dense bias를 위한 Zeros, LayerNorm beta를 위한 Zeros, LayerNorm gamma를 위한 Ones.

## Layer-wise Learning Rate Decay
MultiOptimizer를 사용하여 LLRD를 구현합니다: transformer encoder와 embedding block의 경우 layer-wise decay가 0.9인 초기 learning rate 1e-5, 나머지 model의 경우 1e-4입니다. 모든 learning rate에는 decay rate이 0.3인 ExponentialDecay scheduler가 있습니다.


```python
def get_model():
    input_ids = tf.keras.layers.Input(shape=(MAX_LENGTH,), dtype=tf.int32,
                                      name="input_ids")
    attention_masks = tf.keras.layers.Input(shape=(MAX_LENGTH,), dtype=tf.int32,
                                            name="attention_masks")
    
    deberta_model = transformers.TFAutoModel.from_pretrained(DEBERTA_MODEL,
                                                             config=cfg)
    
    REINIT_LAYERS = 1
    normal_initializer = tf.keras.initializers.GlorotUniform()
    zeros_initializer = tf.keras.initializers.Zeros()
    ones_initializer = tf.keras.initializers.Ones()
    
    for encoder_block in deberta_model.deberta.encoder.layer[-REINIT_LAYERS:]:
        for layer in encoder_block.submodules:
            if isinstance(layer, tf.keras.layers.Dense):
                layer.kernel.assign(normal_initializer(shape=layer.kernel.shape,
                                                       dtype=layer.kernel.dtype))
                if layer.bias is not None:
                    layer.bias.assign(zeros_initializer(shape=layer.bias.shape,
                                                        dtype=layer.bias.dtype))
            
            elif isinstance(layer, tf.keras.layers.LayerNormalization):
                layer.beta.assign(zeros_initializer(shape=layer.beta.shape,
                                                    dtype=layer.beta.dtype))
                layer.gamma.assign(ones_initializer(shape=layer.gamma.shape,
                                                    dtype=layer.gamma.dtype))
    
    deberta_output = deberta_model.deberta(input_ids, attention_mask=attention_masks)
    hidden_states = deberta_output.hidden_states # (None, 512, 768) 여러개
    
    # WeightedLayerPool + MeanPool of the last 4 hidden states
    stack_meanpool = tf.stack([MeanPool()(hidden_s, mask=attention_masks)
                               for hidden_s in hidden_states[-4:]],
                              axis=2) # (None, 768, 4)
    
    weighted_layer_pool = layers.Dense(1, use_bias=False,
                                      kernel_constraint=WeightsSumOne())(stack_meanpool)
    
    weighted_layer_pool = tf.squeeze(weighted_layer_pool, axis=-1)
    
    x = layers.Dense(6, activation="sigmoid")(weighted_layer_pool)
    output = layers.Rescaling(scale=4.0, offset=1.0)(x)
    model = tf.keras.Model(inputs=[input_ids, attention_masks], outputs=output)
    
    # Compile model with Layer-wise Learning Rate Decay
    layer_list = [deberta_model.deberta.embeddings] + list(deberta_model.deberta.encoder.layer)
    layer_list.reverse()
    
    INIT_LR = 1e-5
    LLRDR = 0.9
    LR_SCH_DECAY_STEPS = 1600 # 2 * len(train_df) // BATCH_SIZE
    
    lr_schedules = [tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=INIT_LR * LLRDR ** i,
        decay_steps=LR_SCH_DECAY_STEPS,
        decay_rate=0.3) for i in range(len(layer_list))]
    
    lr_schedule_head = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-4,
        decay_steps=LR_SCH_DECAY_STEPS,
        decay_rate=0.3
    )
    
    optimizers = [tf.keras.optimizers.Adam(learning_rate=lr_sch) for lr_sch in lr_schedules]
    
    optimizers_and_layers = [(tf.keras.optimizers.Adam(learning_rate=lr_schedule_head),
                              model.layers[-4:])] + list(zip(optimizers, layer_list))
    
    optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)
    
    model.compile(optimizer=optimizer,
                  loss="huber_loss",
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model
```


```python
tf.keras.backend.clear_session()
model = get_model()
model.summary()
```

    2022-11-09 06:09:59.493138: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2022-11-09 06:09:59.494217: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2022-11-09 06:09:59.494896: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2022-11-09 06:09:59.495755: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 AVX512F FMA
    To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2022-11-09 06:09:59.496083: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2022-11-09 06:09:59.496771: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2022-11-09 06:09:59.497428: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2022-11-09 06:10:04.029922: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2022-11-09 06:10:04.030780: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2022-11-09 06:10:04.031467: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2022-11-09 06:10:04.032050: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1510] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 15043 MB memory:  -> device: 0, name: Tesla P100-PCIE-16GB, pci bus id: 0000:00:04.0, compute capability: 6.0
    2022-11-09 06:10:14.576277: W tensorflow/python/util/util.cc:348] Sets are not currently considered sequences, but this may change in the future, so consider avoiding using them.
    

    Model: "model"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_ids (InputLayer)          [(None, 512)]        0                                            
    __________________________________________________________________________________________________
    attention_masks (InputLayer)    [(None, 512)]        0                                            
    __________________________________________________________________________________________________
    deberta (TFDebertaV2MainLayer)  TFBaseModelOutput(la 183831552   input_ids[0][0]                  
                                                                     attention_masks[0][0]            
    __________________________________________________________________________________________________
    mean_pool (MeanPool)            (None, 768)          0           deberta[0][9]                    
                                                                     attention_masks[0][0]            
    __________________________________________________________________________________________________
    mean_pool_1 (MeanPool)          (None, 768)          0           deberta[0][10]                   
                                                                     attention_masks[0][0]            
    __________________________________________________________________________________________________
    mean_pool_2 (MeanPool)          (None, 768)          0           deberta[0][11]                   
                                                                     attention_masks[0][0]            
    __________________________________________________________________________________________________
    mean_pool_3 (MeanPool)          (None, 768)          0           deberta[0][12]                   
                                                                     attention_masks[0][0]            
    __________________________________________________________________________________________________
    tf.stack (TFOpLambda)           (None, 768, 4)       0           mean_pool[0][0]                  
                                                                     mean_pool_1[0][0]                
                                                                     mean_pool_2[0][0]                
                                                                     mean_pool_3[0][0]                
    __________________________________________________________________________________________________
    dense (Dense)                   (None, 768, 1)       4           tf.stack[0][0]                   
    __________________________________________________________________________________________________
    tf.compat.v1.squeeze (TFOpLambd (None, 768)          0           dense[0][0]                      
    __________________________________________________________________________________________________
    dense_1 (Dense)                 (None, 6)            4614        tf.compat.v1.squeeze[0][0]       
    __________________________________________________________________________________________________
    rescaling (Rescaling)           (None, 6)            0           dense_1[0][0]                    
    ==================================================================================================
    Total params: 183,836,170
    Trainable params: 183,836,170
    Non-trainable params: 0
    __________________________________________________________________________________________________
    

# 5 Folds Training Loop


```python
valid_rmses = []
for fold in range(N_FOLD):
    print(f'\n-----------FOLD {fold} ------------')
    train_df = df[df["fold"] != fold].reset_index(drop=True)
    valid_df = df[df["fold"] == fold].reset_index(drop=True)

    train_dataset = get_dataset(train_df)
    valid_dataset = get_dataset(valid_df)

    print('Data prepared.')
    print(f'Training data input_ids shape: {train_dataset[0][0].shape} dtype: {train_dataset[0][0].dtype}') 
    print(f'Training data attention_mask shape: {train_dataset[0][1].shape} dtype: {train_dataset[0][1].dtype}')
    print(f'Training data targets shape: {train_dataset[1].shape} dtype: {train_dataset[1].dtype}')
    print(f'Validation data input_ids shape: {valid_dataset[0][0].shape} dtype: {valid_dataset[0][0].dtype}')
    print(f'Validation data attention_mask shape: {valid_dataset[0][1].shape} dtype: {valid_dataset[0][1].dtype}')
    print(f'Validation data targets shape: {valid_dataset[1].shape} dtype: {valid_dataset[1].dtype}')

    tf.keras.backend.clear_session()
    model = get_model()
    print('Model prepared.')

    print('Start training...')
    callbacks = [tf.keras.callbacks.ModelCheckpoint(f"best_model_fold{fold}.h5", monitor="val_loss",
                                                     mode="min", save_best_only=True,
                                                     verbose=1, save_weights_only=True),
                 tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=1e-5, 
                                                  patience=3, verbose=1, mode="min")]
    history = model.fit(x=train_dataset[0], y=train_dataset[1],
                        validation_data=valid_dataset,
                        epochs=10, shuffle=True,
                        batch_size=BATCH_SIZE, callbacks=callbacks)
    valid_rmses.append(np.min(history.history["val_root_mean_squared_error"]))

    print('Training finished.')
    del train_dataset, valid_dataset, train_df, valid_df
    gc.collect()
```

    
    -----------FOLD 0 ------------
    Data prepared.
    Training data input_ids shape: (3129, 512) dtype: int32
    Training data attention_mask shape: (3129, 512) dtype: int32
    Training data targets shape: (3129, 6) dtype: float32
    Validation data input_ids shape: (782, 512) dtype: int32
    Validation data attention_mask shape: (782, 512) dtype: int32
    Validation data targets shape: (782, 6) dtype: float32
    Model prepared.
    Start training...
    Epoch 1/10
    

    2022-11-09 06:10:43.814448: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:185] None of the MLIR Optimization Passes are enabled (registered 2)
    

    783/783 [==============================] - 418s 478ms/step - loss: 0.1230 - root_mean_squared_error: 0.4997 - val_loss: 0.1047 - val_root_mean_squared_error: 0.4588
    
    Epoch 00001: val_loss improved from inf to 0.10474, saving model to best_model_fold0.h5
    Epoch 2/10
    783/783 [==============================] - 368s 470ms/step - loss: 0.1000 - root_mean_squared_error: 0.4485 - val_loss: 0.1070 - val_root_mean_squared_error: 0.4637
    
    Epoch 00002: val_loss did not improve from 0.10474
    Epoch 3/10
    783/783 [==============================] - 367s 469ms/step - loss: 0.0916 - root_mean_squared_error: 0.4289 - val_loss: 0.1011 - val_root_mean_squared_error: 0.4506
    
    Epoch 00003: val_loss improved from 0.10474 to 0.10110, saving model to best_model_fold0.h5
    Epoch 4/10
    783/783 [==============================] - 367s 469ms/step - loss: 0.0865 - root_mean_squared_error: 0.4164 - val_loss: 0.1025 - val_root_mean_squared_error: 0.4537
    
    Epoch 00004: val_loss did not improve from 0.10110
    Epoch 5/10
    783/783 [==============================] - 367s 468ms/step - loss: 0.0832 - root_mean_squared_error: 0.4084 - val_loss: 0.1021 - val_root_mean_squared_error: 0.4527
    
    Epoch 00005: val_loss did not improve from 0.10110
    Epoch 6/10
    783/783 [==============================] - 366s 468ms/step - loss: 0.0813 - root_mean_squared_error: 0.4037 - val_loss: 0.1026 - val_root_mean_squared_error: 0.4539
    
    Epoch 00006: val_loss did not improve from 0.10110
    Epoch 00006: early stopping
    Training finished.
    
    -----------FOLD 1 ------------
    Data prepared.
    Training data input_ids shape: (3128, 512) dtype: int32
    Training data attention_mask shape: (3128, 512) dtype: int32
    Training data targets shape: (3128, 6) dtype: float32
    Validation data input_ids shape: (783, 512) dtype: int32
    Validation data attention_mask shape: (783, 512) dtype: int32
    Validation data targets shape: (783, 6) dtype: float32
    Model prepared.
    Start training...
    Epoch 1/10
    782/782 [==============================] - 383s 447ms/step - loss: 0.1240 - root_mean_squared_error: 0.5017 - val_loss: 0.1072 - val_root_mean_squared_error: 0.4647
    
    Epoch 00001: val_loss improved from inf to 0.10719, saving model to best_model_fold1.h5
    Epoch 2/10
    782/782 [==============================] - 343s 438ms/step - loss: 0.0991 - root_mean_squared_error: 0.4463 - val_loss: 0.1061 - val_root_mean_squared_error: 0.4620
    
    Epoch 00002: val_loss improved from 0.10719 to 0.10606, saving model to best_model_fold1.h5
    Epoch 3/10
    782/782 [==============================] - 343s 438ms/step - loss: 0.0901 - root_mean_squared_error: 0.4250 - val_loss: 0.1068 - val_root_mean_squared_error: 0.4638
    
    Epoch 00003: val_loss did not improve from 0.10606
    Epoch 4/10
    782/782 [==============================] - 343s 438ms/step - loss: 0.0854 - root_mean_squared_error: 0.4138 - val_loss: 0.1054 - val_root_mean_squared_error: 0.4605
    
    Epoch 00004: val_loss improved from 0.10606 to 0.10541, saving model to best_model_fold1.h5
    Epoch 5/10
    782/782 [==============================] - 343s 438ms/step - loss: 0.0820 - root_mean_squared_error: 0.4053 - val_loss: 0.1048 - val_root_mean_squared_error: 0.4591
    
    Epoch 00005: val_loss improved from 0.10541 to 0.10475, saving model to best_model_fold1.h5
    Epoch 6/10
    782/782 [==============================] - 343s 438ms/step - loss: 0.0800 - root_mean_squared_error: 0.4003 - val_loss: 0.1058 - val_root_mean_squared_error: 0.4614
    
    Epoch 00006: val_loss did not improve from 0.10475
    Epoch 7/10
    782/782 [==============================] - 343s 438ms/step - loss: 0.0790 - root_mean_squared_error: 0.3979 - val_loss: 0.1049 - val_root_mean_squared_error: 0.4595
    
    Epoch 00007: val_loss did not improve from 0.10475
    Epoch 8/10
    782/782 [==============================] - 343s 438ms/step - loss: 0.0783 - root_mean_squared_error: 0.3962 - val_loss: 0.1049 - val_root_mean_squared_error: 0.4595
    
    Epoch 00008: val_loss did not improve from 0.10475
    Epoch 00008: early stopping
    Training finished.
    
    -----------FOLD 2 ------------
    Data prepared.
    Training data input_ids shape: (3129, 512) dtype: int32
    Training data attention_mask shape: (3129, 512) dtype: int32
    Training data targets shape: (3129, 6) dtype: float32
    Validation data input_ids shape: (782, 512) dtype: int32
    Validation data attention_mask shape: (782, 512) dtype: int32
    Validation data targets shape: (782, 6) dtype: float32
    Model prepared.
    Start training...
    Epoch 1/10
    783/783 [==============================] - 415s 476ms/step - loss: 0.1229 - root_mean_squared_error: 0.4991 - val_loss: 0.1131 - val_root_mean_squared_error: 0.4780
    
    Epoch 00001: val_loss improved from inf to 0.11313, saving model to best_model_fold2.h5
    Epoch 2/10
    783/783 [==============================] - 367s 469ms/step - loss: 0.0987 - root_mean_squared_error: 0.4453 - val_loss: 0.1061 - val_root_mean_squared_error: 0.4626
    
    Epoch 00002: val_loss improved from 0.11313 to 0.10613, saving model to best_model_fold2.h5
    Epoch 3/10
    783/783 [==============================] - 367s 468ms/step - loss: 0.0906 - root_mean_squared_error: 0.4265 - val_loss: 0.1106 - val_root_mean_squared_error: 0.4731
    
    Epoch 00003: val_loss did not improve from 0.10613
    Epoch 4/10
    783/783 [==============================] - 367s 469ms/step - loss: 0.0856 - root_mean_squared_error: 0.4143 - val_loss: 0.1070 - val_root_mean_squared_error: 0.4647
    
    Epoch 00004: val_loss did not improve from 0.10613
    Epoch 5/10
    783/783 [==============================] - 367s 469ms/step - loss: 0.0823 - root_mean_squared_error: 0.4061 - val_loss: 0.1072 - val_root_mean_squared_error: 0.4654
    
    Epoch 00005: val_loss did not improve from 0.10613
    Epoch 00005: early stopping
    Training finished.
    
    -----------FOLD 3 ------------
    Data prepared.
    Training data input_ids shape: (3129, 512) dtype: int32
    Training data attention_mask shape: (3129, 512) dtype: int32
    Training data targets shape: (3129, 6) dtype: float32
    Validation data input_ids shape: (782, 512) dtype: int32
    Validation data attention_mask shape: (782, 512) dtype: int32
    Validation data targets shape: (782, 6) dtype: float32
    Model prepared.
    Start training...
    Epoch 1/10
    783/783 [==============================] - 414s 475ms/step - loss: 0.1214 - root_mean_squared_error: 0.4957 - val_loss: 0.1091 - val_root_mean_squared_error: 0.4684
    
    Epoch 00001: val_loss improved from inf to 0.10907, saving model to best_model_fold3.h5
    Epoch 2/10
    783/783 [==============================] - 366s 467ms/step - loss: 0.0982 - root_mean_squared_error: 0.4443 - val_loss: 0.1040 - val_root_mean_squared_error: 0.4571
    
    Epoch 00002: val_loss improved from 0.10907 to 0.10395, saving model to best_model_fold3.h5
    Epoch 3/10
    783/783 [==============================] - 365s 467ms/step - loss: 0.0901 - root_mean_squared_error: 0.4251 - val_loss: 0.1034 - val_root_mean_squared_error: 0.4557
    
    Epoch 00003: val_loss improved from 0.10395 to 0.10337, saving model to best_model_fold3.h5
    Epoch 4/10
    783/783 [==============================] - 366s 467ms/step - loss: 0.0849 - root_mean_squared_error: 0.4125 - val_loss: 0.1030 - val_root_mean_squared_error: 0.4550
    
    Epoch 00004: val_loss improved from 0.10337 to 0.10301, saving model to best_model_fold3.h5
    Epoch 5/10
    783/783 [==============================] - 366s 467ms/step - loss: 0.0818 - root_mean_squared_error: 0.4049 - val_loss: 0.1030 - val_root_mean_squared_error: 0.4550
    
    Epoch 00005: val_loss did not improve from 0.10301
    Epoch 6/10
    783/783 [==============================] - 367s 468ms/step - loss: 0.0799 - root_mean_squared_error: 0.4001 - val_loss: 0.1036 - val_root_mean_squared_error: 0.4563
    
    Epoch 00006: val_loss did not improve from 0.10301
    Epoch 7/10
    783/783 [==============================] - 367s 469ms/step - loss: 0.0788 - root_mean_squared_error: 0.3973 - val_loss: 0.1033 - val_root_mean_squared_error: 0.4557
    
    Epoch 00007: val_loss did not improve from 0.10301
    Epoch 00007: early stopping
    Training finished.
    
    -----------FOLD 4 ------------
    Data prepared.
    Training data input_ids shape: (3129, 512) dtype: int32
    Training data attention_mask shape: (3129, 512) dtype: int32
    Training data targets shape: (3129, 6) dtype: float32
    Validation data input_ids shape: (782, 512) dtype: int32
    Validation data attention_mask shape: (782, 512) dtype: int32
    Validation data targets shape: (782, 6) dtype: float32
    Model prepared.
    Start training...
    Epoch 1/10
    783/783 [==============================] - 419s 479ms/step - loss: 0.1217 - root_mean_squared_error: 0.4961 - val_loss: 0.1055 - val_root_mean_squared_error: 0.4612
    
    Epoch 00001: val_loss improved from inf to 0.10550, saving model to best_model_fold4.h5
    Epoch 2/10
    783/783 [==============================] - 368s 471ms/step - loss: 0.0989 - root_mean_squared_error: 0.4459 - val_loss: 0.1040 - val_root_mean_squared_error: 0.4576
    
    Epoch 00002: val_loss improved from 0.10550 to 0.10404, saving model to best_model_fold4.h5
    Epoch 3/10
    783/783 [==============================] - 367s 469ms/step - loss: 0.0913 - root_mean_squared_error: 0.4280 - val_loss: 0.1033 - val_root_mean_squared_error: 0.4558
    
    Epoch 00003: val_loss improved from 0.10404 to 0.10327, saving model to best_model_fold4.h5
    Epoch 4/10
    783/783 [==============================] - 368s 470ms/step - loss: 0.0857 - root_mean_squared_error: 0.4145 - val_loss: 0.1016 - val_root_mean_squared_error: 0.4522
    
    Epoch 00004: val_loss improved from 0.10327 to 0.10158, saving model to best_model_fold4.h5
    Epoch 5/10
    783/783 [==============================] - 369s 472ms/step - loss: 0.0827 - root_mean_squared_error: 0.4072 - val_loss: 0.1017 - val_root_mean_squared_error: 0.4524
    
    Epoch 00005: val_loss did not improve from 0.10158
    Epoch 6/10
    783/783 [==============================] - 368s 470ms/step - loss: 0.0807 - root_mean_squared_error: 0.4022 - val_loss: 0.1022 - val_root_mean_squared_error: 0.4535
    
    Epoch 00006: val_loss did not improve from 0.10158
    Epoch 7/10
    783/783 [==============================] - 367s 469ms/step - loss: 0.0797 - root_mean_squared_error: 0.3997 - val_loss: 0.1016 - val_root_mean_squared_error: 0.4522
    
    Epoch 00007: val_loss did not improve from 0.10158
    Epoch 00007: early stopping
    Training finished.
    


```python
print(f'{len(valid_rmses)} Folds validation RMSE:\n{valid_rmses}')
print(f'Local CV Average score: {np.mean(valid_rmses)}')
```

    5 Folds validation RMSE:
    [0.4505632519721985, 0.459128201007843, 0.46264517307281494, 0.45495015382766724, 0.45215508341789246]
    Local CV Average score: 0.45588837265968324
    

# Inference and Submission


```python
test_df = pd.read_csv("../input/feedback-prize-english-language-learning/test.csv")
test_df.head()
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
      <th>text_id</th>
      <th>full_text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0000C359D63E</td>
      <td>when a person has no experience on a job their...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>000BAD50D026</td>
      <td>Do you think students would benefit from being...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>00367BB2546B</td>
      <td>Thomas Jefferson once states that "it is wonde...</td>
    </tr>
  </tbody>
</table>
</div>




```python
test_dataset = deberta_encode(test_df['full_text'])
```

# 5 Folds ensemble prediction


```python
fold_preds = []
for fold in range(N_FOLD):
    tf.keras.backend.clear_session()
    model = get_model()
    model.load_weights(f"best_model_fold{fold}.h5")
    print(f'\nFold {fold} inference...')
    pred = model.predict(test_dataset, batch_size=BATCH_SIZE)
    fold_preds.append(pred)
    gc.collect()
```

    
    Fold 0 inference...
    
    Fold 1 inference...
    
    Fold 2 inference...
    
    Fold 3 inference...
    
    Fold 4 inference...
    


```python
preds = np.mean(fold_preds, axis=0)
preds = np.clip(preds, 1, 5)
```


```python
sub_df = pd.concat([test_df[["text_id"]], pd.DataFrame(preds, columns=TARGET_COLS)], axis=1)
sub_df.to_csv("submission.csv", index=False)
```


```python
sub_df.head()
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
      <th>text_id</th>
      <th>cohesion</th>
      <th>syntax</th>
      <th>vocabulary</th>
      <th>phraseology</th>
      <th>grammar</th>
      <th>conventions</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0000C359D63E</td>
      <td>2.927250</td>
      <td>2.759098</td>
      <td>3.090907</td>
      <td>2.983907</td>
      <td>2.65453</td>
      <td>2.625527</td>
    </tr>
    <tr>
      <th>1</th>
      <td>000BAD50D026</td>
      <td>2.697297</td>
      <td>2.502063</td>
      <td>2.743973</td>
      <td>2.359509</td>
      <td>2.12162</td>
      <td>2.607434</td>
    </tr>
    <tr>
      <th>2</th>
      <td>00367BB2546B</td>
      <td>3.708724</td>
      <td>3.460325</td>
      <td>3.653716</td>
      <td>3.513729</td>
      <td>3.37132</td>
      <td>3.262542</td>
    </tr>
  </tbody>
</table>
</div>




------------------------
참고

[https://www.kaggle.com/code/electro/deberta-layerwiselr-lastlayerreinit-tensorflow](https://www.kaggle.com/code/electro/deberta-layerwiselr-lastlayerreinit-tensorflow)
