import xgboost as xgb
from xgboost import XGBClassifier

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score

import imblearn
from termcolor import colored


def plot_confusion_matrix(y_test, pred):
	# Calculating the confusion matrix
	conf_matrix = confusion_matrix(y_test, pred)

	# Plotting the confusion matrix
	plt.figure(figsize=(10, 7))
	sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues')
	plt.xlabel('Predicted labels')
	plt.ylabel('True labels')
	plt.title('Confusion Matrix')
	plt.show()
 
 



def predict_base(classifier, X_test, y_test):
	pred = classifier.predict(X_test)

	# Assuming you have true labels in 'y_test'
	accuracy = accuracy_score(y_test, pred)
	precision = precision_score(y_test, pred)
	recall = recall_score(y_test, pred)
	f1 = f1_score(y_test, pred)

	print(f"Accuracy: {accuracy:.2f}")
	print(f"Precision: {precision:.2f}")
	print(f"Recall: {recall:.2f}")
	print(f"F1 Score: {f1:.2f}")
	return pred

def predict_f1(classifier, X_test, y_test):

	predicted_probabilities = classifier.predict_proba(X_test)[:, 1]  # Probabilities for the positive class

	# Define a range of threshold values (e.g., from 0.1 to 0.9)
	thresholds = np.arange(0.1, 1.0, 0.1)

	best_threshold = None
	best_f1_score = 0

	# Find the threshold that maximizes the F1-score
	for threshold in thresholds:
		predicted_labels = (predicted_probabilities > threshold).astype(int)
		f1 = f1_score(y_test, predicted_labels)
		
		if f1 > best_f1_score:
			best_f1_score = f1
			best_threshold = threshold

	print(best_threshold)
	print(best_f1_score)

	predicted_labels = (predicted_probabilities > best_threshold).astype(int)

	return predicted_labels



def predict_recall(classifier, X_test):
	predicted_probabilities = classifier.predict_proba(X_test)[:, 1]  # Probabilities for the positive class

	# Find the lowest predicted probability for the positive class
	lowest_positive_prob = np.min(predicted_probabilities)

	print(f"Lowest predicted probability: {lowest_positive_prob:.4f}")

	# Apply the threshold to classify the predictions
	predicted_labels = (predicted_probabilities > lowest_positive_prob).astype(int)
 
	return predicted_labels



def predict_all(classifier, X_test, y_test):
	print(colored('BASIC PREDICT', 'green', attrs=['bold']))
	pred = predict_base(classifier, X_test, y_test)
#	print(confusion_matrix(y_test, pred))

	# Calculate the AUPRC
	auprc = average_precision_score(y_test, pred)

	print("AUPRC:", auprc)

	print(colored('F1 PREDICT', 'green', attrs=['bold']))
	pred_f1 = predict_f1(classifier, X_test, y_test)
#	print(confusion_matrix(y_test, pred_f1))

	auprc = average_precision_score(y_test, pred_f1)

	print("AUPRC:", auprc)

	print(colored('RECALL PREDICT', 'green', attrs=['bold']))
	pred_recall = predict_recall(classifier, X_test)
#	print(confusion_matrix(y_test, pred_recall))


#	plot_confusion_matrix(y_test, pred)






 
classifiers = [
				['XGB :', XGBClassifier()],
				['LGBM :', LGBMClassifier(verbose=-1)],
				['RandomForest :',RandomForestClassifier()], 
				['Neural Network :', MLPClassifier()],
				['LogisticRegression :', LogisticRegression()],
				['ExtraTreesClassifier :', ExtraTreesClassifier()],
				['AdaBoostClassifier :', AdaBoostClassifier()],
				['GradientBoostingClassifier: ', GradientBoostingClassifier()],
				['CatBoost :', CatBoostClassifier(logging_level='Silent')],
				['DecisionTree :',DecisionTreeClassifier()],
				['Naive Bayes :', GaussianNB()],
				['KNeighbours :', KNeighborsClassifier()],
	]

def eval_all(X_train, X_test, y_train, y_test):
	predictions_df = pd.DataFrame()
	predictions_df['actual_labels'] = y_test

	for name,classifier in classifiers:
		classifier = classifier
		classifier.fit(X_train, y_train)
		print(colored(f'{name}', 'blue', attrs=['bold']))

		predict_all(classifier, X_test, y_test)
		print()












def tune_lgbm(X_train, X_test, y_train, y_test):
	param_grid = {
		'num_leaves': [15, 31, 63],  # Number of leaves in each tree
		'max_depth': [3, 5, 7, -1],  # Maximum tree depth (-1 means no limit)
		'learning_rate': [0.05, 0.1, 0.2],  # Learning rate
		'n_estimators': [50, 100, 200],  # Number of boosting iterations
		'subsample': [0.8, 1.0],  # Fraction of samples used for training
		'colsample_bytree': [0.8, 1.0],  # Fraction of features used for training in each tree
	}

	# Create the LightGBM classifier
	lgb_model = LGBMClassifier(random_state=42, verbose=-1)

	# Initialize the GridSearchCV with cross-validation
	grid_search = GridSearchCV(
		estimator=lgb_model,
		param_grid=param_grid,
		scoring='recall',
		cv=3,
		verbose=2,  # Set the verbosity level
		n_jobs=-1,  # Use all available CPU cores for parallel processing
	)

	# Fit the grid search to find the best parameters
	grid_search.fit(X_train, y_train)

	# Get the best parameters and the corresponding model
	best_params = grid_search.best_params_
	best_model = grid_search.best_estimator_

	# Evaluate the best model on the test set
	test_accuracy = best_model.score(X_test, y_test)

	print("Best Parameters:", best_params)
	print("Test Accuracy:", test_accuracy)

#	predict_all(classifier, X_test, y_test)



def train_xgb(X_train, X_test, y_train, y_test):

	classifier = LGBMClassifier(verbose=-1)

	classifier.fit(X_train, y_train)

	predict_all(classifier, X_test, y_test)

#	eval_all(X_train, X_test, y_train, y_test)