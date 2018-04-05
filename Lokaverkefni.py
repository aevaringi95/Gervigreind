#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 11:17:47 2018

@author: aevarjohannesson
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

#Generate data. Check for Nan values.
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

pd.isnull(train)

train.isnull().sum().sum() # there seem to be no NAN values.






