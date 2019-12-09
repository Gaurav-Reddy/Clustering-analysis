# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 15:53:35 2019

@author: Y Gaurav Reddy
"""

# designing a shallow auto encoder

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense
from keras.models import Model

#------------cleaning start------------
water_treatment = pd.read_csv("C:/Users/Y Gaurav Reddy/Documents/GitHub/Clustering-analysis.git/trunk/water-treatment.data", header = None)
df=pd.DataFrame(water_treatment)
df.drop(df.columns[0], inplace = True, axis = 1)  #dropping the 1st coloumn as we dont need it (its just date)

df.columns = ["Q_E", "ZN_E", "PH_E", "DBO_E", "DQO_E", "SS_E",      #rename cols for understanding
    "SSV_E", "SED_E", "COND_E", "PH_P", "DBO_P", "SS_P", "SSV_P", "SED_P", "COND_P",
    "PH_D", "DBO_D", "DQO_D", "SS_D", "SSV_D", "SED_D", "COND_D", "PH_S", "DBO_S",
    "DQO_S", "SS_S", "SSV_S", "SED_S", "COND_S", "RD_DBO_P", "RD_SS_P", "RD_SED_P",
    "RD_DBO_S", "RD_DQO_S", "RD_DBO_G", "RD_DQO_G", "RD_SS_G", "RD_SED_G"]
df=df.replace("?",np.NaN)

print(df.isnull().sum()) #see number of missing values
values=df.values
imputer=Imputer()
transformed_values = imputer.fit_transform(values) #we have filled the missing values with means of col
# count the number of NaN values in each column
print(np.isnan(transformed_values).sum())   # should be ==0
waterData_clean= normalize(transformed_values) 

X_train, X_test= train_test_split(waterData_clean, test_size=0.85)
encoding_dim = 3
ncol = waterData_clean.shape[1]

input_dim = Input(shape = (ncol, ))
# Encoder Layers
encoded1 = Dense(3000, activation = 'relu')(input_dim)
encoded2 = Dense(2750, activation = 'relu')(encoded1)
encoded3 = Dense(2500, activation = 'relu')(encoded2)
encoded4 = Dense(2250, activation = 'relu')(encoded3)
encoded5 = Dense(2000, activation = 'relu')(encoded4)
encoded6 = Dense(1750, activation = 'relu')(encoded5)
encoded7 = Dense(1500, activation = 'relu')(encoded6)
encoded8 = Dense(1250, activation = 'relu')(encoded7)
encoded9 = Dense(1000, activation = 'relu')(encoded8)
encoded10 = Dense(750, activation = 'relu')(encoded9)
encoded11 = Dense(500, activation = 'relu')(encoded10)
encoded12 = Dense(250, activation = 'relu')(encoded11)
encoded13 = Dense(encoding_dim, activation = 'relu')(encoded12)
# Decoder Layers
decoded1 = Dense(250, activation = 'relu')(encoded13)
decoded2 = Dense(500, activation = 'relu')(decoded1)
decoded3 = Dense(750, activation = 'relu')(decoded2)
decoded4 = Dense(1000, activation = 'relu')(decoded3)
decoded5 = Dense(1250, activation = 'relu')(decoded4)
decoded6 = Dense(1500, activation = 'relu')(decoded5)
decoded7 = Dense(1750, activation = 'relu')(decoded6)
decoded8 = Dense(2000, activation = 'relu')(decoded7)
decoded9 = Dense(2250, activation = 'relu')(decoded8)
decoded10 = Dense(2500, activation = 'relu')(decoded9)
decoded11 = Dense(2750, activation = 'relu')(decoded10)
decoded12 = Dense(3000, activation = 'relu')(decoded11)
decoded13 = Dense(ncol, activation = 'sigmoid')(decoded12)
# Combine Encoder and Deocder layers
autoencoder = Model(inputs = input_dim, outputs = decoded13)
autoencoder.compile(optimizer = 'adadelta', loss = 'binary_crossentropy')
autoencoder.summary()
autoencoder.fit(X_train, X_train, nb_epoch = 10, batch_size = 32, shuffle = False, validation_data = (X_test, X_test))
encoder = Model(inputs = input_dim, outputs = encoded13)
encoded_input = Input(shape = (encoding_dim, ))

encoded_train = pd.DataFrame(encoder.predict(waterData_clean))
encoded_train = encoded_train.add_prefix('feature_')


encoded_train.to_csv('train_encoded.csv', index=False)
