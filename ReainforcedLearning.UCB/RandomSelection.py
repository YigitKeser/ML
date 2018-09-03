# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 21:47:29 2018

@author: Yigit
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv('Ads_CTR_Optimisation.csv')

import random

N = 10000 #Kac kere olusturulacak
d = 10 #Kac rastgele değer oluşturulacak
toplam = 0
secilenler = []
for n in range(0,N):
    ad = random.randrange(d)
    secilenler.append(ad)
    odul = veriler.values[n,ad] #n. satır = 1 ise odul 1
    toplam = toplam + odul

plt.hist(secilenler)
plt.show