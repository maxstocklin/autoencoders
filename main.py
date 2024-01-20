import sys

sys.path.append("models")
sys.path.append("visualization")
sys.path.append("preprocessing")

# from preprocessing import preprocess_data
from visualization import visualize_data


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

from colorama import Fore, Back, Style, init

warnings.filterwarnings("ignore")





def load_data(path):
    dataset = pd.read_csv(path)
    return dataset

def main():
#	try:
		if len(sys.argv) != 2:
			print("Usage: python test.py <filename>")
			sys.exit(1)
		dataset = pd.read_csv(sys.argv[1])
		visualize_data(dataset)

		X = dataset.drop('Class', axis=1)
		y = dataset['Class']

#	except:
#		print('An exception has occured')


if __name__ == "__main__":
	main()
