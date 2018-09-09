# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 15:15:46 2018

@author: jjonus
"""

from jfunc import tpot_advhouse,rmsle,drop_cols,normalize
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import math_ops
from sklearn.metrics import make_scorer
import tensorflow as tf
from keras.layers import Dense,Dropout
import numpy as np
import warnings
import os
import math
warnings.filterwarnings('ignore')

# load in data from github

X_train,X_valid,X_test,df_test,df_train,y_train,y_valid = tpot_advhouse(plot = False)

# remove cols
remove = ['Foundation_PConc','MSSubClassGroup_60','MasVnrType_Stone','HasPconc','Electrical_SBrkr','HasVnrStone','LotShape_IR1','MasVnrType_None','LotShape_Reg','KitchenQual']

X_train = drop_cols(X_train,remove)
X_valid = drop_cols(X_valid,remove)
X_test = drop_cols(X_test,remove)

# custom loss function
"""
def rmsle_loss(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y_true[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]
    if not (y_true >= 0).all() and not (y_pred >= 0).all():
            raise ValueError("Mean Squared Logarithmic Error cannot be used when "
                             "targets contain negative values.")
    return (sum(terms_to_sum) * (1.0/len(y_true))) ** 0.5

rmsle_loss = make_scorer(rmsle_loss,greater_is_better=False)   
"""

def rmsle(y_true, y_pred):
  first_log = math_ops.log(K.clip(y_pred, K.epsilon(), None) + 1.)
  second_log = math_ops.log(K.clip(y_true, K.epsilon(), None) + 1.)
  return K.sqrt(K.mean(math_ops.square(first_log - second_log), axis=-1))

# build neural net

X_train = normalize(X_train)
X_valid = normalize(X_valid)

# build network
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(60,activation='linear'))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(3,activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(1,activation = 'linear'))
model.add(tf.keras.layers.Dense(30,activation='linear'))
model.add(tf.keras.layers.Dropout(0.2))
model.compile(optimizer="adam",
              loss=rmsle)

model.fit(np.array(X_train),np.array(y_train),epochs=100,batch_size=50)

val_loss,val_acc = model.evaluate(np.array(X_valid),np.array(y_valid))
print(val_loss,val_acc)

yval = model.predict(X_valid)
yval = model.predict(X_train)
