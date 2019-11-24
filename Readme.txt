CS 435/535
Fall 2019
Assignment 2
Assigned on 13 November 2019
Due on 9 December 2019

Total Points are: 150




This assignment focuses on clustering study, and in particular, the well-known k-means method. The associated data file is named as water-treatment.data, and the documentation file for this data set is given in the file named water-treatment-dataDescription.txt. Specifically, there are 527 data items and each of them is a 38-dimensional vector. Please note that each attribute (i.e., dimension) has a different range of the values. Also please note that there are missing values. Please pay attention to the final output format specified in the description file. For undergrad students, you are required to complete questions 1 – 5 with a full credit of 100 pts; for grad students, you are required to complete all the six questions with a full credit of 150 pts.

1. (20 pts.) Clean up the data set. This includes filling up the missing values and normalizing all the data items. Please state clearly the methods you use for filling up the missing values and normalizing the values in English to answer this question.

2. (20 pts.) It is well-known that the k-means algorithm requires that the number of clusters, k, be given in advance. In this problem, we do not know the k value in advance. Propose a specific termination condition for the modified k-means when searching the true k value. State clearly your proposed condition or method in English.

3. (20 pts.) Implement the modified k-means algorithm with your proposed termination condition and run the algorithm using the water-treatment dataset. Please note that you must use the output format given in the description file. Report your output.

4. (20 pts.) Apply the PCA method you implemented in the first assignment to this dataset. Then apply the implemented modified k-means method above to this reduced data set to report the output. Please follow the same protocol of the output format specified in the description file.

5. (20 pts.) Compare the two clustering results and analyze any differences that you have observed and state why there is such difference if there is or why there is no difference if there is no.

6. (50 pts.) Implement an autoencoder (either shallow or deep) for dimensionality reduction and apply the implemented autoencoder to the given dataset. Report the dimensionality reduction result using the autoencoder and discuss the difference between PCA and autoencoder for dimensionality reduction with this dataset.

