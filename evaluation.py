"""
compare the two models
"""
from PCA_model import *
from autoencoder_model import *

# verify the PCA model on test data
anomaly_alldata.plot(logy=True, figsize=(10, 6), ylim=[1e-1, 1e3], color=['green', 'red'])
plt.show()

# results from the autoencoder model
scored.plot(logy=True,  figsize=(10, 6), ylim=[1e-2, 1e2], color=['blue', 'red'])
plt.show()
