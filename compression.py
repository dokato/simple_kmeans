import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from kmeans import kmeans, assign_cluster

image = mpimg.imread("magda.png")
p, q, l = image.shape

k = 4
im2 = image.copy().reshape((p*q, l))
labels, centroids =  kmeans(im2, k)

for i in range(k):
    im2[labels==i, :] = centroids[i]

im3 = im2.reshape((p, q, l))

plt.subplot(1,2,1)
plt.imshow(image)
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(im3)
plt.axis('off')
plt.show()

