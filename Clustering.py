#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 16:52:01 2024

@author: mehdy
"""
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import KMeans
import numpy as np

img =mpimg.imread('X1.jpg')
img = img.astype('float64')
img = img/255.0
img_r = img.reshape((img.shape[0]*img.shape[1],3))


kmeans = KMeans(n_clusters=3,  init='random', tol=10) # Specifiy number of clusters
kmeans.fit(img_r) # fit input data
labels = kmeans.labels_ # labels of the clustering algorithm
center = kmeans.cluster_centers_ # cluster centers



# Map each pixel to its cluster center color
segmented_img = center[labels].reshape(img.shape)

# Display the original image and the segmentation
# fig, axes = plt.subplots(1, 2, figsize=(12, 6))
# axes[0].imshow(img)
# axes[1].imshow(segmented_img)
# plt.show()



###################################################################################

gmm = GaussianMixture(n_components=3) #Specify number of clusters and covariance shape
gmm.fit(img_r)#  fit input data 
labels = gmm.predict(img_r) #labels of the clustering algorithm
segmentation = labels.reshape((img.shape[0],img.shape[1]))
plt.imshow(segmentation)
Mean = gmm.means_ # Mean for GMM. 
Cov = gmm.covariances_ # Covariance for GMM.
print("Means for GMM clusters:")
print(Mean)

print("\nCovariances for GMM clusters:")
print(Cov)

plt.show()