from termcolor import colored

import os
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings



def show_data(dataset):
	print(colored('dataset', 'green', attrs=['bold']))
	print(dataset)

	print(colored('dataset.info()', 'green', attrs=['bold']))
	print(dataset.info())

#	print(dataset.tail())
#	print(dataset.head())

	print(colored('Numbers', 'green', attrs=['bold']))
	print(dataset.select_dtypes(include='number'))

	print(colored('Categories', 'green', attrs=['bold']))
	print(dataset.select_dtypes(exclude='number'))


	print(colored('describe', 'green', attrs=['bold']))
	print(dataset.describe())

	print(colored('isnull', 'green', attrs=['bold']))
	print(dataset.isnull().sum())





def plot_data(dataset):
	dataset.plot() # Creates line plots for all numerical columns.
	dataset.hist() # Creates histograms for all numerical columns.

	save_dir = 'files/visualisation/line_plots'
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)  # Create the directory if it doesn't exist

	# Create and save line plots for each numerical column
	for col in dataset.select_dtypes(include=['number']).columns:
		plt.figure(figsize=(10, 6))  # Create a new figure for each plot
		plt.plot(dataset.index, dataset[col])
		plt.title(f'Line Plot of {col}')
		plt.xlabel('Index')
		plt.ylabel(col)
		plt.savefig(os.path.join(save_dir, f'{col}_line_plot.png'))
		plt.close()  # Close the figure to free up resources

	# Show the line plots
	plt.show()

	save_dir = 'files/visualisation/histograms'
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)  # Create the directory if it doesn't exist

	# Create and save histograms for each numerical column
	for col in dataset.select_dtypes(include=['number']).columns:
		plt.figure(figsize=(10, 6))  # Create a new figure for each plot
		plt.hist(dataset[col], bins=20)  # Adjust the number of bins as needed
		plt.title(f'Histogram of {col}')
		plt.xlabel(col)
		plt.ylabel('Frequency')
		plt.savefig(os.path.join(save_dir, f'{col}_histogram.png'))
		plt.close()  # Close the figure to free up resources

	# Show the histograms
	plt.show()


def plot_target(dataset):
	sns.countplot(data=dataset, x='Class')
	plt.title('Class Distribution')
	plt.xlabel('Class')
	plt.ylabel('Count')

	save_dir = 'files/visualisation/target'
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)  # Create the directory if it doesn't exist

	# Save the plot as a PNG file
	plt.savefig(os.path.join(save_dir, 'class_distribution.png'))

#	plt.show()


def plot_features(col, dataset):
	fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # Create a 1x3 subplot grid

	# Boxplot
	sns.boxplot(data=dataset, x='Class', y=col, ax=axes[0])
	axes[0].set_title(f'Distribution of {col} by Class')
	axes[0].set_xlabel('Class')
	axes[0].set_ylabel(f'{col}')

	# KDE Plot
	sns.kdeplot(dataset[dataset['Class'] == 0][col], label='Class 0', shade=True, ax=axes[1])
	sns.kdeplot(dataset[dataset['Class'] == 1][col], label='Class 1', shade=True, ax=axes[1])
	axes[1].set_title(f'Distribution of {col} by Class')
	axes[1].set_xlabel(f'{col}')
	axes[1].set_ylabel('Density')
	axes[1].legend()

	# Histogram
	sns.histplot(data=dataset[dataset['Class'] == 0], x=col, label='Class 0', kde=True, color='blue', alpha=0.5, ax=axes[2])
	sns.histplot(data=dataset[dataset['Class'] == 1], x=col, label='Class 1', kde=True, color='orange', alpha=0.5, ax=axes[2])
	axes[2].set_title(f'Distribution of {col} by Class')
	axes[2].set_xlabel(f'{col}')
	axes[2].set_ylabel('Frequency')
	axes[2].legend()

	plt.tight_layout()  # Ensure plots do not overlap

	save_dir = 'files/visualisation/features'
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)  # Create the directory if it doesn't exist

	# Save the plot as a PNG file
	plt.savefig(os.path.join(save_dir, col + '_distribution.png'))

#	plt.show()


def visualize_data(dataset):
#	show_data(dataset)
	plot_data(dataset)
	exit()
	plot_target(dataset)
	columns = dataset.columns

	for col in columns:
		plot_features(col, dataset)



