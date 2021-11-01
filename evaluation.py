"""
compare the two models
"""
import matplotlib.pyplot as plt

from PCA_model import *
from autoencoder_model import *

# verify results from the PCA model on test data
anomaly_alldata.plot(title='PCA Model', logy=True, figsize=(10, 6), ylim=[1e-1, 1e3], color=['green', 'red'])
plt.savefig('PCA model')
plt.show()

# results from the autoencoder model on test data
scored.plot(title='autoencoder model', logy=True,  figsize=(10, 6), ylim=[1e-2, 1e2], color=['blue', 'red'])
plt.savefig('autoencoder model')
plt.show()
