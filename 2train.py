import xgboost as xgb
from xgboost import XGBClassifier

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score
from sklearn import metrics

import imblearn

warnings.filterwarnings("ignore")


# Extract feature and target arrays
dataset = pd.read_csv('data/raw/creditcard.csv')

tmp = dataset
X_valid, y_valid = tmp.drop('Class', axis=1), tmp[['Class']]

# Split the data
X_train, y_train = dataset.drop('Class', axis=1), dataset[['Class']]

from imblearn.over_sampling import SMOTE

# Oversampling the minority class
sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)

'''
# Plot Survivors data
y_train.Class.value_counts().plot(kind='bar')
plt.show()
'''

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Create DMatrix for train, validation, and test sets
dtrain_class = xgb.DMatrix(X_train, y_train)
dtest_class = xgb.DMatrix(X_test, y_test)


# Calculating the scale_pos_weight
# This is usually the ratio of number of negative class to the positive class
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

# Initialize XGBoost with scale_pos_weight
model = xgb.XGBClassifier(scale_pos_weight=scale_pos_weight)

# Fit the model
model.fit(X_train, y_train)
pred = model.predict(X_valid)
print(metrics.accuracy_score(pred, y_valid))

from sklearn.metrics import confusion_matrix

# Calculating the confusion matrix
conf_matrix = confusion_matrix(y_valid, pred)

# Plotting the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()
exit()


params = {
    'objective': 'binary:logistic',  # Use 'multi:softmax' for multi-class and set 'num_class'
    'max_depth': 6,
    'learning_rate': 0.01,
	'colsample_bytree' : 0.6, #[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
	'min_child_weight' : 10, #[1, 5, 10, 20, 30, 50, 100],
	'eval_metric': 'auc',  # Specify your evaluation metric here

    # Add other parameters here
}

num_rounds = 1000  # Set a high value
early_stopping_rounds = 50
#evals = [(dtest_class, "validation"), (dtrain_class, "train")]
evals = [(dtest_class, "validation")]  # Use validation set for early stopping

bst = xgb.train(params, 
	dtrain_class, 
    num_rounds,
	evals=evals,
    early_stopping_rounds=early_stopping_rounds,
	verbose_eval=50 # Every 50 rounds
)

# Convert probabilities to binary labels
threshold = 0.5
preds = bst.predict(dtest_class)
pred_labels = (preds > threshold).astype(int)

print("Accuracy:", accuracy_score(y_test, pred_labels))
print("Best number of boosting rounds:", bst.best_iteration)

from sklearn.metrics import confusion_matrix

# Calculating the confusion matrix
conf_matrix = confusion_matrix(y_test, pred_labels)

# Plotting the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()
exit()

