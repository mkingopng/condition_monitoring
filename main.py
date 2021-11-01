"""

"""

import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from numpy.random import seed
from keras.layers import Input, Dropout, Dense
from keras.models import Model, Sequential, load_model
from keras import regularizers
import tensorflow as tf

sns.set(color_codes=True)

data_dir = 'data/2nd_test/2nd_test'
merged_data = pd.DataFrame()

for filename in os.listdir(data_dir):
    # print(filename)
    dataset = pd.read_csv(os.path.join(data_dir, filename), sep='\t')
    dataset_mean_abs = np.array(dataset.abs().mean())
    dataset_mean_abs = pd.DataFrame(dataset_mean_abs.reshape(1, 4))
    dataset_mean_abs.index = [filename]
    merged_data = merged_data.append(dataset_mean_abs)

merged_data.columns = ['Bearing 1', 'Bearing 2', 'Bearing 3', 'Bearing 4']

merged_data.index = pd.to_datetime(merged_data.index, format='%Y.%m.%d.%H.%M.%S')
merged_data = merged_data.sort_index()
merged_data.to_csv('data/merged_dataset_BearingTest_2.csv')
print(merged_data.head())

dataset_train = merged_data['2004-02-12 11:02:39':'2004-02-13 23:52:39']
dataset_test = merged_data['2004-02-13 23:52:39':]
dataset_train.plot(figsize=(12, 6))
plt.show()

scaler = preprocessing.MinMaxScaler()

X_train = pd.DataFrame(scaler.fit_transform(dataset_train),
                       columns=dataset_train.columns,
                       index=dataset_train.index)  # Random shuffle training data
X_train.sample(frac=1)

X_test = pd.DataFrame(scaler.transform(dataset_test),
                      columns=dataset_test.columns,
                      index=dataset_test.index)

pca = PCA(n_components=2, svd_solver='full')

X_train_PCA = pca.fit_transform(X_train)
X_train_PCA = pd.DataFrame(X_train_PCA)
X_train_PCA.index = X_train.index

X_test_PCA = pca.transform(X_test)
X_test_PCA = pd.DataFrame(X_test_PCA)
X_test_PCA.index = X_test.index


# calculate the covariance matrix


def cov_matrix(data, verbose=False):
    covariance_matrix = np.cov(data, rowvar=False)
    if is_pos_def(covariance_matrix):
        inv_covariance_matrix = np.linalg.inv(covariance_matrix)
        if is_pos_def(inv_covariance_matrix):
            return covariance_matrix, inv_covariance_matrix
        else:
            print("Error: Inverse of Covariance Matrix is not positive definite!")
    else:
        print("Error: Covariance Matrix is not positive definite!")


# calculate the Mahalanobis distance


def MahalanobisDist(inv_cov_matrix, mean_distr, data, verbose=False):
    inv_covariance_matrix = inv_cov_matrix
    vars_mean = mean_distr
    diff = data - vars_mean
    md = []
    for i in range(len(diff)):
        md.append(np.sqrt(diff[i].dot(inv_covariance_matrix).dot(diff[i])))
    return md


# detecting outliers


def MD_detectOutliers(dist, extreme=False, verbose=False):
    k = 3. if extreme else 2.
    threshold = np.mean(dist) * k
    outliers = []
    for i in range(len(dist)):
        if dist[i] >= threshold:
            outliers.append(i)  # index of the outlier
    return np.array(outliers)


# calculate threshold for classifying datapoint as anomaly


def MD_threshold(dist, extreme=False, verbose=False):
    k = 3. if extreme else 2.
    threshold = np.mean(dist) * k
    return threshold


# check if matrix is positive


def is_pos_def(A):
    if np.allclose(A, A.T):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False


# set up the PCA model

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
plt.show()

# visualize the Mahalanobis distance
plt.figure()
sns.histplot(dist_train, bins=10, color='green')
plt.xlim([0.0, 5])
plt.xlabel('Mahalanobis dist')
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

# verify the PCA model on test data
anomaly_alldata.plot(logy=True, figsize=(10, 6), ylim=[1e-1, 1e3], color=['green', 'red'])
plt.show()

# defining the autoencoder network
seed(10)
tf.compat.v1.set_random_seed(10)
act_func = 'elu'

# Input layer:
model = Sequential()

# First hidden layer, connected to input vector X.
model.add(Dense(10, activation=act_func, kernel_regularizer=regularizers.l2(0.0), input_shape=(X_train.shape[1],)))

model.add(Dense(2, activation=act_func))

model.add(Dense(10, activation=act_func))

model.add(Dense(X_train.shape[1]))

model.compile(loss='mse', optimizer='adam')

# Train model for 100 epochs, batch size of 10:
NUM_EPOCHS = 100
BATCH_SIZE = 10

history = model.fit(np.array(X_train), np.array(X_train),
                    batch_size=BATCH_SIZE,
                    epochs=NUM_EPOCHS,
                    validation_split=0.05,
                    verbose=1)

# visualise training/validation loss
plt.plot(history.history['loss'], 'b', label='Training loss')
plt.plot(history.history['val_loss'], 'r', label='Validation loss')
plt.legend(loc='upper right')
plt.xlabel('Epochs')
plt.ylabel('Loss, [mse]')
plt.ylim([0, .1])
plt.show()

# distribution of loss function in the training set
X_pred = model.predict(np.array(X_train))
X_pred = pd.DataFrame(X_pred,
                      columns=X_train.columns)
X_pred.index = X_train.index

scored = pd.DataFrame(index=X_train.index)
scored['Loss_mae'] = np.mean(np.abs(X_pred-X_train), axis=1)

plt.figure()
sns.histplot(scored['Loss_mae'], bins=10, kde=True, color='blue')
plt.xlim([0.0, .5])
plt.show()

# let's try a threshold of 0.3 for flagging anomaly
X_pred = model.predict(np.array(X_test))
X_pred = pd.DataFrame(X_pred, columns=X_test.columns)
X_pred.index = X_test.index

scored = pd.DataFrame(index=X_test.index)
scored['Loss_mae'] = np.mean(np.abs(X_pred-X_test), axis=1)
scored['Threshold'] = 0.3
scored['Anomaly'] = scored['Loss_mae'] > scored['Threshold']
print(scored.head())

# calculate the same metrics for the training set, and merge all data in a single dataframe
X_pred_train = model.predict(np.array(X_train))
X_pred_train = pd.DataFrame(X_pred_train, columns=X_train.columns)
X_pred_train.index = X_train.index

scored_train = pd.DataFrame(index=X_train.index)
scored_train['Loss_mae'] = np.mean(np.abs(X_pred_train-X_train), axis=1)
scored_train['Threshold'] = 0.3
scored_train['Anomaly'] = scored_train['Loss_mae'] > scored_train['Threshold']
scored = pd.concat([scored_train, scored])

# results from the autoencoder model
scored.plot(logy=True,  figsize=(10, 6), ylim=[1e-2, 1e2], color=['blue', 'red'])
plt.show()
