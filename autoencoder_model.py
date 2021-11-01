"""
test autoencoder model
"""
from utilities import *

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
plt.title('Trainging/Validation Loss')
plt.legend(loc='upper right')
plt.xlabel('Epochs')
plt.ylabel('Loss, [mse]')
plt.ylim([0, .1])
plt.savefig('training_validation_loss.png')
plt.show()

# distribution of loss function in the training set
X_pred = model.predict(np.array(X_train))
X_pred = pd.DataFrame(X_pred, columns=X_train.columns)
X_pred.index = X_train.index

scored = pd.DataFrame(index=X_train.index)
scored['Loss_mae'] = np.mean(np.abs(X_pred-X_train), axis=1)

plt.figure()
sns.histplot(scored['Loss_mae'], bins=10, kde=True, color='blue')
plt.title('MAE Loss')
plt.xlim([0.0, .5])
plt.savefig('MAE_Loss.png')
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
