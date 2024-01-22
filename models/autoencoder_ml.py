import xgboost as xgb
from xgboost import XGBClassifier

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

import imblearn
from termcolor import colored




def autoencode(X_train, X_test, y_train, y_test):
	print (X_train)
	exit()

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_curve, accuracy_score
import matplotlib.pyplot as plt


from keras.layers import Input, Dense
from keras.models import Model, Sequential
from keras.optimizers import Adam


def build_autoencoder(input_dim):
	input_layer = Input(shape=(input_dim,))
	
	# Encoder
	encoder = Sequential([
		Dense(14, activation="relu", input_shape=(input_dim,)),
		Dense(7, activation="relu")
	])
	# Decoder
	decoder = Sequential([
		Dense(14, activation="relu", input_shape=(7,)),
		Dense(input_dim, activation="linear")
	])
	
	autoencoder = Model(inputs=input_layer, outputs=decoder(encoder(input_layer)))
	autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

	return autoencoder, encoder, decoder

def visualize_performance(encoder, decoder, data, title):
	if isinstance(data, pd.DataFrame):
		data = data.values  # Convert DataFrame to NumPy array
	encoded_data = encoder.predict(data)
	decoded_data = decoder.predict(encoded_data)

	plt.plot(data[0], 'b')
	plt.plot(decoded_data[0], 'r')
	plt.title(title)
	plt.show()

def calculate_loss_and_plot(autoencoder, normal_data, anomaly_data):
	# Calculate the loss on normal data
	reconstruction = autoencoder.predict(normal_data)
	train_loss = tf.keras.losses.mae(reconstruction, normal_data)
	plt.hist(train_loss, bins=50)

	# Determine the threshold
	threshold = np.mean(train_loss) + 2*np.std(train_loss)

	# Calculate the loss on anomaly data
	reconstruction_a = autoencoder.predict(anomaly_data)
	train_loss_a = tf.keras.losses.mae(reconstruction_a, anomaly_data)
	plt.hist(train_loss_a, bins=50)
	plt.axvline(threshold, color='r', linewidth=3, linestyle='dashed', label=f'{threshold:.3f}')
	plt.legend(loc='upper right')
	plt.title("Loss on Normal and Anomaly Test Data")
	plt.show()

	# Plot combined histogram
	plt.hist(train_loss, bins=50, label='normal')
	plt.hist(train_loss_a, bins=50, label='anomaly')
	plt.axvline(threshold, color='r', linewidth=3, linestyle='dashed', label=f'{threshold:.3f}')
	plt.legend(loc='upper right')
	plt.title("Combined Normal and Anomaly Loss")
	plt.show()

	# Detecting anomalies
	preds = tf.math.less(train_loss, threshold)
	print("Number of Normal Predictions: ", tf.math.count_nonzero(preds).numpy())



# Function to plot the reconstruction error
def plot_reconstruction_error(error_df, threshold):
	groups = error_df.groupby('True_class')
	fig, ax = plt.subplots()

	for name, group in groups:
		ax.plot(group.index, group.Reconstruction_error, marker='o', ms=3.5, linestyle='',
				label= "Fraud" if name == 1 else "Normal",
				color='red' if name == 1 else 'blue')
	ax.hlines(threshold, ax.get_xlim()[0], ax.get_xlim()[1], colors="green", zorder=100, label='Threshold')
	ax.legend()
	plt.title("Reconstruction error for different classes")
	plt.ylabel("Reconstruction error")
	plt.xlabel("Data point index")
	plt.show()

def autoencode_this(X_train, X_test, X_val, y_test, normal_test_data, anomaly_test_data):
	input_dim = X_train.shape[1] # Number of features
	print('Number of columns = ', input_dim)
	
	# Print shapes of the datasets for verification
	print("X_train shape:", X_train.shape)
	print("normal_test_data shape:", normal_test_data.shape)
	print("X_val shape:", X_val.shape)
	print("anomaly_test_data shape:", anomaly_test_data.shape)

	autoencoder, encoder, decoder = build_autoencoder(input_dim)

	# Train the model
	autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, validation_data=(X_val, X_val), shuffle=True)


	# Predict the reconstructed output using the autoencoder for normal test data
	reconstructed_normal = autoencoder.predict(normal_test_data)
	# Calculate the mean squared reconstruction error for normal test data
	mse_normal = np.mean(np.power(normal_test_data - reconstructed_normal, 2), axis=1)
	error_df_normal = pd.DataFrame({'Reconstruction_error': mse_normal,
									'True_class': y_test[y_test == 0]})

	# Predict the reconstructed output for anomaly test data
	reconstructed_anomaly = autoencoder.predict(anomaly_test_data)
	# Calculate the mean squared reconstruction error for anomaly test data
	mse_anomaly = np.mean(np.power(anomaly_test_data - reconstructed_anomaly, 2), axis=1)
	error_df_anomaly = pd.DataFrame({'Reconstruction_error': mse_anomaly,
									 'True_class': y_test[y_test == 1]})

	# Combine normal and anomaly error dataframes
	error_df = pd.concat([error_df_normal, error_df_anomaly])

	# Determine a threshold (Here, using the 90th percentile of the error of normal transactions)
	threshold = np.percentile(error_df_normal['Reconstruction_error'], 90)

	plot_reconstruction_error(error_df, threshold)

	# Visualize model performance
	visualize_performance(encoder, decoder, normal_test_data, "Model Performance on Normal Data")
	visualize_performance(encoder, decoder, anomaly_test_data, "Model Performance on Anomaly Data")

	# Calculate loss and plot
	calculate_loss_and_plot(autoencoder, normal_test_data, anomaly_test_data)
