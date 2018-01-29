# http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/
# pip install -U scikit-learn


# Generate a dataset and plot it
np.random.seed(0)
X, y = sklearn.datasets.make_moons(200, noise=0.20)
plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)