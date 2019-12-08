# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 22:02:08 2019

@author: Y Gaurav Reddy
"""
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import normalize
from sklearn import decomposition
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans 


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


#Applying PCA
plt.figure()
pca = decomposition.PCA(n_components=2)             #come to this conclusion
waterData_pc = pca.fit_transform(waterData_clean)
waterData_pc.shape
plt.plot((pca.explained_variance_ratio_))
plt.grid()
plt.minorticks_on()
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
#plt.set_ylim(ymin=0)
plt.grid(which='major', linestyle='-', linewidth='0.5', color='red')
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('explained variance');
cost =[] 
for i in range(2, 20): 
    KM = KMeans(n_clusters = i,random_state=i, init='k-means++') 
    KM.fit(waterData_pc) 
      
    # calculates squared error 
    # for the clustered points 
    
    
    cost.append(KM.inertia_) 
    

    
# Get the cluster labels

print("\nKnn computed successfully\n")



# plot the cost against K values 
from kneed import KneeLocator                                                  #THIS ISA NEW PACKAGE TO LOCATE THE KNEE POINT
kn = KneeLocator(range(2, 20), cost, curve='convex', direction='decreasing')
optimum_K= (kn.knee)


print(optimum_K)

plt.figure()
plt.xlabel('number of clusters k')
plt.ylabel('Sum of squared distances')
plt.grid()
plt.minorticks_on()
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.grid(which='major', linestyle='-', linewidth='0.5', color='red')
plt.plot(range(2,20), cost, 'bx-')
plt.vlines(kn.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')

KM = KMeans(n_clusters = optimum_K, init='k-means++')                           #THIS IS TO RUN THE K MEANS WITH AN OPTIMUM number of CLUSTERS
KM.fit(waterData_pc) 
y_kmeans = KM.predict(waterData_pc)

plt.figure()
plt.scatter(waterData_pc[:, 0], waterData_pc[:, 1], c=y_kmeans, s=50, cmap='viridis')

centers = KM.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);

plt.figure()
plt.scatter(waterData_pc[:,0
    ],waterData_pc[:,1],alpha=.1, color='black')
    
if os.path.isfile('ModiKnn_PCA_outputFile.txt'):                                    #checks is the output file exists and if its previously present it removes it 
    os.remove("ModiKnn_PCA_outputFile.txt")                                         #and helps removing it making a new file
    print("File Removed! A new file will be created with the KNN output \n")


for i in range(len(KM.labels_)):
       f = open("ModiKnn_PCA_outputFile.txt", "a")
       f.write(str(i+1)+" "+str(KM.labels_[i])+"\n")

f.close()
print("Output File created")  