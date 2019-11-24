# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 20:49:46 2019

@author: Y Gaurav Reddy
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans 
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist 
import matplotlib.pyplot as plt  

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


#here we start applying CLUSTERING!!!!!!!!!!!!!!!

cost =[] 
for i in range(1, 6): 
    KM = KMeans(n_clusters = i, max_iter = 500) 
    KM.fit(waterData_clean) 
      
    # calculates squared error 
    # for the clustered points 
    cost.append(KM.inertia_)  

KM = KMeans(n_clusters = 5, max_iter = 500) 
KM.fit(waterData_clean)    
print(silhouette_score(waterData_clean, KM.labels_))
    
# Get the cluster labels

print("\nKnn computed successfully\n")

'''
from yellowbrick.cluster import KElbowVisualizer

# Instantiate a scikit-learn K-Means model
model = KMeans(random_state=0)

# Instantiate the KElbowVisualizer with the number of clusters and the metric 
visualizer = KElbowVisualizer(model, k=(2,12),metric='distortion')

# Fit the data and visualize
visualizer.fit(waterData_clean)    
visualizer.poof() '''

#seeing the kmeans plot
'''
for i in range(len(KM.labels_)):
       print(str(i+1)+" "+str(KM.labels_[i]))     #OUTPUT'''


# plot the cost against K values 

plt.plot(range(1, 6), cost, color ='b', linewidth ='3') 
plt.grid()
plt.minorticks_on()
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.grid(which='major', linestyle='-', linewidth='0.5', color='red')
plt.xlabel("Value of K") 
plt.ylabel("Sqaured Error (Cost)") 
plt.ylim(ymin=0)
plt.xlim(xmin=0)
plt.show() # clear the plot 
  
# the point of the elbow is the  
# most optimal value for choosing k 
