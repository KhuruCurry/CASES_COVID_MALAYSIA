# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 17:09:57 2022

@author: Khuru
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

class EDA():
    def _init_(self):
        pass
        
    def plot_graph(self,df):
        plt.figure()
        plt.plot(df['cases_new']) 
        plt.legend(df['cases_new']) 
        plt.show()