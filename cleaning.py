# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 17:26:55 2019

@author: Y Gaurav Reddy
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import normalize



#py 1st file commit
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

#Begin normalizing dataset
#we plane to normalize every sample induvidually
#we normalize everything to a value between 0 and 1 based on description in the readme file
waterData_clean= normalize(transformed_values) 

