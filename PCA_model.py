"""
test PCS model
"""
import matplotlib.pyplot as plt

from utilities import *

data_train = np.array(X_train_PCA.values)
data_test = np.array(X_test_PCA.values)

# calculate the covariance matrix and its inverse
cov_matrix, inv_cov_matrix = cov_matrix(data_train)

# calculate the mean value for the input values in the training set
mean_distr = data_train.mean(axis=0)

dist_test = MahalanobisDist(inv_cov_matrix, mean_distr, data_test)
dist_train = MahalanobisDist(inv_cov_matrix, mean_distr, data_train)
threshold = MD_threshold(dist_train, extreme=True)

plt.figure()
sns.histplot(np.square(dist_train), bins=10)
plt.xlim([0.0, 15])
plt.title('Mahalanobis Index Chi-square distribution')
plt.savefig('Mahalanobis_Index.png')
plt.show()

# visualize the Mahalanobis distance
plt.figure()
sns.histplot(dist_train, bins=10, color='green')
plt.xlim([0.0, 5])
plt.xlabel('Mahalanobis dist')
plt.title('Mahalanobis Distance')
plt.savefig('Mahalanobis_Distance.png')
plt.show()

# save the Mahalanobis distance, the threshold value and “anomaly flag” variable
# for both train and test data in a dataframe
anomaly_train = pd.DataFrame()
anomaly_train['Mob dist'] = dist_train
anomaly_train['Thresh'] = threshold

# If Mob dist above threshold: Flag as anomaly
anomaly_train['Anomaly'] = anomaly_train['Mob dist'] > anomaly_train['Thresh']
anomaly_train.index = X_train_PCA.index

anomaly = pd.DataFrame()
anomaly['Mob dist'] = dist_test
anomaly['Thresh'] = threshold

# If Mob dist above threshold: Flag as anomaly
anomaly['Anomaly'] = anomaly['Mob dist'] > anomaly['Thresh']
anomaly.index = X_test_PCA.index
print(anomaly.head())

anomaly_alldata = pd.concat([anomaly_train, anomaly])
anomaly_alldata.to_csv('data/Anomaly_distance.csv')
