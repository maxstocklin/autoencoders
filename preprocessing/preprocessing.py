
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

from sklearn.model_selection import train_test_split
from sklearn import metrics

import imblearn
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")


def preprocess_data(dataset):
	X = dataset.drop('Class', axis=1)
	y = dataset['Class']

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	# Oversampling the minority class
	sm = SMOTE(random_state=42)
	X_train, y_train = sm.fit_resample(X_train, y_train)

	# Print the class distribution
	y_train_df = pd.DataFrame(y_train, columns=['Class'])
	class_distribution = y_train_df['Class'].value_counts()
#	print(class_distribution)

	return X_train, X_test, y_train, y_test



from sklearn.preprocessing import StandardScaler, MinMaxScaler

def preprocess_autoencoder(dataset):
    X = dataset.drop('Class', axis=1)
    y = dataset['Class']

    # Fill missing values if any (example using mean)
    # dataset.fillna(dataset.mean(), inplace=True)

    # Filter out normal transactions for training the autoencoder
    normal_data = X[y == 0]

    # Normalizing 'Time' and 'Amount' in the normal_data
    scaler = MinMaxScaler()
    normal_data[['Time', 'Amount']] = scaler.fit_transform(normal_data[['Time', 'Amount']])

    # Splitting the normal data into training and validation sets
    X_train, X_val = train_test_split(normal_data, test_size=0.2, random_state=42)

    # Use the entire dataset (including anomalous data) for testing
    # Apply the same scaling to the test data
    X_test = X.copy()
    X_test[['Time', 'Amount']] = scaler.transform(X_test[['Time', 'Amount']])
    y_test = y

    # Separate normal and anomaly test data for additional analysis if needed
    normal_test_data = X_test[y_test == 0]
    anomaly_test_data = X_test[y_test == 1]

    return X_train, X_val, X_test, y_test, normal_test_data, anomaly_test_data

