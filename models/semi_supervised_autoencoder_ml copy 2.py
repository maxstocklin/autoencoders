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

'''

def train_autoencoder(autoencoder, X_train, X_test, epochs, batch_size):
	history = autoencoder.fit(X_train, X_train, epochs=epochs, batch_size=batch_size,
							  shuffle=True, validation_data=(X_test, X_test), verbose=1)
	return history

def evaluate_autoencoder(autoencoder, X_test, y_test):
	y_pred = autoencoder.predict(X_test)
	mse = np.mean(np.power(X_test - y_pred, 2), axis=1)
	return mse



def temp(X_train, X_test, y_train, y_test):

	input_dim = X_train.shape[1]

	# Build and train the autoencoder
	autoencoder = build_autoencoder(input_dim)
	epochs = 50  # Adjust the number of epochs as needed
	batch_size = 64  # Adjust the batch size as needed

	history = train_autoencoder(autoencoder, X_train, X_test, epochs, batch_size)

	# Evaluate the autoencoder and set a threshold for anomaly detection
	mse_values = evaluate_autoencoder(autoencoder, X_test, y_test)

	# Set a threshold for anomaly detection (e.g., using a percentile of MSE values)
	threshold = np.percentile(mse_values, 95)  # Adjust the percentile as needed

	# Detect anomalies
	y_pred = (mse_values > threshold).astype(int)

	# Calculate metrics (e.g., precision, recall, F1-score, accuracy)
	precision, recall, _ = precision_recall_curve(y_test, y_pred)
	fpr, tpr, _ = roc_curve(y_test, y_pred)
	auc_score = auc(fpr, tpr)
	accuracy = accuracy_score(y_test, y_pred)

	# Print and visualize metrics
	print(f'Precision: {precision[1]:.2f}')
	print(f'Recall: {recall[1]:.2f}')
	print(f'F1-score: {(2 * precision[1] * recall[1]) / (precision[1] + recall[1]):.2f}')
	print(f'Accuracy: {accuracy:.2f}')
	print(f'AUC: {auc_score:.2f}')

	# Plot the ROC curve
	plt.figure()
	plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {auc_score:.2f}')
	plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver Operating Characteristic')
	plt.legend(loc='lower right')
	plt.show()






'''
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam

def build_autoencoder(input_dim):
	input_layer = Input(shape=(input_dim,))
	
	# Encoder
	encoder = Dense(14, activation="relu")(input_layer)
	encoder = Dense(7, activation="relu")(encoder)

	# Decoder
	decoder = Dense(14, activation="relu")(encoder)
	decoder = Dense(input_dim, activation="linear")(decoder)

	autoencoder = Model(inputs=input_layer, outputs=decoder)

	autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
	return autoencoder


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


def semi_autoencode_this(X_train, X_test, X_val, y_test):
	input_dim = X_train.shape[1] # Number of features
	print('cols num = ', input_dim)
	autoencoder = build_autoencoder(input_dim)

	# Train the model
	autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, validation_data=(X_val, X_val), shuffle=True)

	# Predict the reconstructed output using the autoencoder
	reconstructed = autoencoder.predict(X_test)

	# Calculate the mean squared reconstruction error
	mse = np.mean(np.power(X_test - reconstructed, 2), axis=1)

	# Create a DataFrame for visualization
	error_df = pd.DataFrame({'Reconstruction_error': mse,
							'True_class': y_test})

	# Determine a threshold (Here, using the 90th percentile of the error of normal transactions)
	threshold = np.percentile(error_df[error_df['True_class'] == 0]['Reconstruction_error'], 90)


	plot_reconstruction_error(error_df, threshold)

