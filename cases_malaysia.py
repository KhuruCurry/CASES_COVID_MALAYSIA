# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 09:53:58 2022

@author: Khuru
"""

import os
import datetime
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from tensorflow.keras import Input
from tensorflow.keras.utils import plot_model
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM,Dense,Dropout
from tensorflow.keras.callbacks import TensorBoard,EarlyStopping
from modules_for_cases_malaysia import EDA

#%% Static

CSV_PATH = os.path.join(os.getcwd(), 'cases_malaysia_train.csv')

log_dir = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
CASES_LOG_FOLDER_PATH = os.path.join(os.getcwd(),'cases_log',log_dir)
CASES_MODEL_SAVE_PATH = os.path.join(os.getcwd(),'cases_model.h5')

#%% Data Loading
df = pd.read_csv(CSV_PATH)

# Data Inspection
df.info()   
df.describe().T
df.boxplot() 
df.isna().sum() 
eda = EDA()

#Data Cleaning
df['cases_new'] = pd.to_numeric(df['cases_new'],errors='coerce')
df.info()
df.isna().sum()

# use interpolate for NaNs value
df['cases_new'].interpolate(method='polynomial',order=2,inplace=True) # to fill NaN for timeseries data
df.isna().sum()
#just to check if 'cases_new' has NaNs
temp = df['cases_new']
temp.isna().sum()


#Preprocessing
mms = MinMaxScaler()
df = mms.fit_transform(np.expand_dims(df['cases_new'],axis=-1))


X_train = [] #initialize something in empty list'[]' "empty container"
y_train = []

win_size = 30

for i in range(win_size,np.shape(df)[0]): #df.shape[0]
    X_train.append(df[i-win_size:i,0]) #[row,columns]
    y_train.append(df[i,0])
    
X_train = np.array(X_train)
y_train = np.array(y_train)

#%% Model developement

model = Sequential()
model.add(Input(shape=(np.shape(X_train)[1],1))) #input_length,#features
model.add(LSTM(64,return_sequences=(True)))
model.add(Dropout(0.3))
model.add(LSTM(64))
model.add(Dropout(0.2))
model.add(Dense(1,activation='relu')) #Output layer

model.compile(optimizer='adam',loss='mse',metrics='mape')

#callbacks
tensorboard_callback = TensorBoard(log_dir=CASES_LOG_FOLDER_PATH)
early_stopping_callback = EarlyStopping(monitor='loss',patience=5)

X_train = np.expand_dims(X_train,axis=-1)
hist = model.fit(X_train,y_train,batch_size=32,epochs=60,
                 callbacks=[tensorboard_callback,early_stopping_callback])

#%% model saving
model.save(CASES_MODEL_SAVE_PATH)

#%%

plot_model(model,show_layer_names=(True),show_shapes=(True))
#%% Model Evaluation and graph
hist.history.keys()

plt.figure()
plt.plot(hist.history['mape'])
plt.show()

plt.figure()
plt.plot(hist.history['loss'])
plt.show()

#%%
CSV_TEST_PATH = os.path.join(os.getcwd(),'cases_malaysia_train.csv')

test_df = pd.read_csv(CSV_TEST_PATH)
test_df.info()

test_df['cases_new']=pd.to_numeric(test_df['cases_new'],errors='coerce')
test_df.info() # got 1 Nans

# use to interpolate for NaNs value
test_df['cases_new'].interpolate(method='polynomial',order=2,inplace=True) # to fill NaN for timeseries data
test_df.isna().sum() # 0 Nans

test_df = mms.transform(np.expand_dims(test_df.iloc[:,1],axis=-1))
con_test = np.concatenate((df,test_df),axis=0) 
con_test = con_test[-(win_size+len(test_df)):]

X_test = []
for i in range(win_size,len(con_test)):
    X_test.append(con_test[i-win_size:i,0])

X_test = np.array(X_test)

predicted = model.predict(np.expand_dims(X_test,axis=-1))


#%% Ploting line the graph

plt.figure()
plt.plot(test_df,'b',label='actual covid cases')
plt.plot(predicted,'r',label='predicted covid cases')
plt.legend()
plt.show()

plt.figure()
plt.plot(mms.inverse_transform(test_df),'b',label='actual covid cases')
plt.plot(mms.inverse_transform(predicted),'r',label='predicted covid cases')
plt.legend()
plt.show()

#%% MAPE
print((mean_absolute_error(test_df, predicted)/sum(abs(test_df)))*100)