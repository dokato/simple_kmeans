import numpy as np
import matplotlib.pyplot as plt

def assign_cluster(X, centroids):
    """
    Assignes labels accoriding to the closest Euclidian distance
    to centroid. Distance is computed using numpy.linalg.norm.

    Parameters:
    -----
      *X* : numpy.array
        two dimensional data vector (n, p) where n - observations
        p -fetures
      *centroids*: numpy.array
        data vector with centroids (k, p) where k nr of labels
    Returns:
    --------
      *labels* : numpy.array
        1D vector with assigned labels
    """
    X = np.asarray(X)
    try:
        n, p = X.shape
    except ValueError:
        X = X.reshape(1,len(X))
        n, p = X.shape
    labels = np.zeros(n)
    for k in range(n): 
        labels[k] = np.argmin(np.linalg.norm(X[k]-centroids,axis=1))
    return labels

def kmeans(X, k, eps=0.1, max_iter=200):
    """
    Simple kmeans clustering implementations
    
    Parameters:
    -----
      *X* : numpy.array
        two dimensional data vector (n, p) where n - observations
        p -fetures
      *k*: int
        number of clusterrs
      *eps*: float
        convergance constant - if difference between old and updated
        centroids is smaller than this value algorithm stops.
    Returns:
    --------
      *labels* : numpy.array
        1D vector with assigned labels from 0 to k in position p
    """
    X = np.asarray(X)
    n, p = X.shape

    centroids_old = np.zeros((k, p))
    init_idxs = np.random.choice(np.arange(n), k, replace=False)
    centroids = X[init_idxs, :]
    l = 0
    while True:
        l+=1
        # assignment step
        labels = assign_cluster(X, centroids)
        # update centroids
        centroids_old = centroids.copy()
        for j in range(k):
            centroids[j] = np.mean(X[labels==j], 0)
        if np.sum(np.abs(centroids_old-centroids)) < eps or l==max_iter:
            break
    return labels, centroids

if __name__=='__main__':
     
    mu1 = np.array([0, 1])
    mu2 = np.array([0, 0])
    cov = np.array([[0.01, 0], [0, 0.01]])

    p1 = np.random.multivariate_normal(mu1, cov, 20)
    p2 = np.random.multivariate_normal(mu2, cov, 20)
    p = np.vstack((p1,p2))
    plt.subplot(1,2,1)
    plt.plot(p[:,0], p[:,1], 'go')
    plt.title('before')

    labels, ctns = kmeans(p, 2)
    p1 = p[labels==0]
    p2 = p[labels==1]
    p3 = p[labels==2]
    plt.subplot(1,2,2)
    plt.plot(p1[:,0], p1[:,1], 'go')
    plt.plot(p2[:,0], p2[:,1], 'mo')
    plt.plot(p3[:,0], p3[:,1], 'ro')
    plt.title('after')
    plt.show()
