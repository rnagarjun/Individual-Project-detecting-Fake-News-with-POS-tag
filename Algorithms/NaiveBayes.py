import math
import pandas as pd
import numpy as np


def train_test_split(dataset, splitRatio):
	"""
	given dataset and a split ratio, splits the dataset into 
	train and test datasets
	"""
	train_dataset = dataset.sample(frac=splitRatio, replace=False, random_state=1)
	indices = train_dataset.index.tolist()
	test_dataset = dataset.drop(indices)
	return train_dataset, test_dataset
 	
def class_probabilities(train_dataset):
	"""
	calculate the probabilites for each class from the train data
	"""
	observation = len(train_dataset)
	class_details = {}
	# get all classes that exist(0 and 1 in this case)
	classes = train_dataset['label'].unique().tolist()
	# get list of features
	feature_list = list(train_dataset)
	# remove label from feature
	feature_list.remove('label')
	#for feature in feature_list:
		# checking if there is atleast 3 unique values for standard deviation to not be zero
		#if len(train_dataset[feature].unique().tolist()) < 4:
		#	train_dataset.drop(train_dataset[feature], axis=1)	
	class_details['classes'] = {}
	for each_class in classes:
		# only get the details for the specific class
		class_dataset = train_dataset.loc[train_dataset['label'] == each_class]
		# remove the label as it's not neccessary for prediction calculation
		class_dataset.drop(class_dataset.columns[len(class_dataset.columns)-1], axis=1, inplace=True)
		# will contains mean and standard deviation corresponding to each feature
		feature_summary = []
		
		for each_feature in feature_list:
			# calculate the mean and std for each features and append to summary
			# mu = class_dataset[each_feature].mean()
			count = class_dataset[each_feature].sum()/observation
			#sigma = class_dataset[each_feature].std()
			feature_summary.append([each_feature, count])
		# converting feature_summary to dataframe for easier manipulations
		class_info = pd.DataFrame(feature_summary, columns=['features', 'count'])
		# store dataframe containing feature summary for each class
		# the class value being the key
		class_details[each_class] = class_info
		class_details['classes'][each_class] = len(class_dataset)/observation
		#print(len(class_dataset)/observation)
		
	print(class_details)
	return class_details			


def check_probability(num_x, prob0, prob1, curr_data):

	probabilities = {}
	#count number of columns that are not zero, divide by numx, then multiple by both pro0 and prob1 then return probs
	temp_sum = 0
	for column in curr_data:
		if float(curr_data[column].values[0]) != 0:
			temp_sum += 1
	prob = temp_sum/num_x
	proby = 1-(prob)
	probabilities[0] = proby*prob0
	probabilities[1] = prob*prob1
	#print(probabilities)
	return probabilities
	
def obtain_prediction(feature_len, total, class_0, class_1, test_dataset):
	predictions = []
	prob0 = class_0/total
	prob1 = class_1/total
	#print(prob0, prob1)
	for i in range(len(test_dataset)):
		curr_data = test_dataset.iloc[[i]]
		each_probabilities = check_probability(feature_len, prob0, prob1, curr_data)
		
		key_max = max(each_probabilities, key=each_probabilities.get)
		predictions.append(key_max)	
	
	return predictions

def test_accuracy(predictions, actual_labels):
	"""
	Test how accurate the prediction was to the test cases labels
	"""
	
	confussion_stats = {'fp':0, 'fn':0, 'tp':0, 'tn':0}
	accurate = 0
	for i in range(len(actual_labels)):
		if actual_labels[i] == int(predictions[i]):
			accurate += 1
			
		if (actual_labels[i] == 0 and int(predictions[i]) == 0):
			confussion_stats['tn'] += 1
		elif (actual_labels[i] == 0 and int(predictions[i]) == 1):
			confussion_stats['fp'] += 1
		elif (actual_labels[i] == 1 and int(predictions[i]) == 0):
			confussion_stats['fn'] += 1
		elif (actual_labels[i] == 1 and int(predictions[i]) == 1):
			confussion_stats['tp'] += 1
	confussion_stats['accuracy'] = (accurate/len(actual_labels)) * 100
	confussion_stats['recall'] = (confussion_stats['tp']/actual_labels.count(1)) * 100
	confussion_stats['percision'] = (confussion_stats['tp']/(confussion_stats['tp']+confussion_stats['fp'])) * 100
	
	return confussion_stats
 

def naive_bayes(train_dataset, test_dataset):
	
	#class_dataset1 = np.count_nonzero(train_dataset[:, -1] == 1)
	#class_dataset0 = train_dataset.shape[0] - class_dataset1
	class_dataset0 = train_dataset.loc[train_dataset['label 36'] == 0]
	class_dataset1 = train_dataset.loc[train_dataset['label 36'] == 1]

	predictions = obtain_prediction(len(list(train_dataset))-1, len(train_dataset), len(class_dataset0), len(class_dataset1), test_dataset)
	accuracy = test_accuracy(predictions, test_dataset['label 36'].tolist())
	return accuracy
 
if __name__ == '__main__':
	# get data from file and store in dataframe
	dataset = pd.read_csv('01_preprocessed.csv')
	# make sure all values are numeric
	dataset = dataset.apply(pd.to_numeric, errors='ignore')
	# The ration to split the train and test data by
	# ex, 0.4 indicate, 40% train, 60% test
	splitRatio = 0.1
	# Actually split data into according parts
	train_dataset, test_dataset = train_test_split(dataset, splitRatio)
	# output split details
	print("Split details:\n Total Size: "+str(len(dataset))+"\n Train Size: "+str(len(train_dataset))+"\n Test Size: "+str(len(test_dataset))+"")
	# calculate probability of each class (fake/real) given training data 
	#probabilities = class_probabilities(train_dataset)
	# predict based off probability and test data
	class_dataset0 = train_dataset.loc[train_dataset['label'] == 0]
	class_dataset1 = train_dataset.loc[train_dataset['label'] == 1]
	predictions = obtain_prediction(len(list(train_dataset))-1, len(train_dataset), len(class_dataset0), len(class_dataset1), test_dataset)
	
	
	# check the accuracy given the predicted results and results from test sample
	# print(predictions)
	accuracy = test_accuracy(predictions, test_dataset['label'].tolist())
	# accuray is a dictionary that contains confussion matrix stats
	print(accuracy)
