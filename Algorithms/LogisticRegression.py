import pandas as pd
import numpy as np

# import warnings filter  
from warnings import simplefilter
simplefilter(action='ignore')

def weightInitialization(n_features):
	"""
	Initializing the weight vector
	"""
	# extra one for bias (bias included in w as w0)
	w = np.zeros((1,n_features+1))
	return w

def train_test_split(dataset, splitRatio):
	"""
	given dataset and a split ratio, splits the dataset into 
	train and test datasets
    """
	train_dataset = dataset.sample(frac=splitRatio, replace=False, random_state=1)
	indices = train_dataset.index.tolist()
	test_dataset = dataset.drop(indices)
	return train_dataset, test_dataset
	
def predict(x, w):
	"""
	Takes in x and weight, calculates the sigmoid formula 
	mentioned in the the documentation.
	"""
	a = np.dot(x, w.T)
	z = (1 / (1 + np.exp(-a)))
	return z

def check_cross_entropy(x, y, w):
	"""
	calculates the cross entropy to assist
	"""
	obs = len(y.T)
	# calls on the rpedict function to find the sigmoid result.
	predictions = predict(x, w)
	# calculate cross entroy
	cross_e1 = -y.T*np.log(predictions)
	cross_e0 = (1-y.T)*np.log(1-predictions)
	cross_e = cross_e1 - cross_e0
	cross_e = cross_e.sum() / obs
	return cross_e
	
def model_train(x, y, w, alpha, num_iter):
	"""
	gradient walk given the step size and the number of iteration
	returns the updated weight for the next step to take
	"""
	entropies = []
	
	i = 0
	while i < num_iter:
		N = len(x)
		predictions = predict(x, w)    
		# calculate gradients for gradient descent 
		gradient = np.dot(x.T,  (predictions - y.T))
		gradient = gradient/N
		# move by the given alpha step, then update weight
		gradient = gradient*alpha
		w -= gradient.T
		# cross entroy to be used to cost
		cross_e = check_cross_entropy(x, y, w)
		entropies.append(cross_e)
		i+=1

	return w

def classify(y_pred):
	"""
	input: takes in array of probabilities
	putput: a list with 0s and 1s
		if probability < 0.5, then put in class 0. etc
	"""
	y_class = []
	for i in y_pred:
		if i >= 0.5:
			y_class.append(1)
		else:
			y_class.append(0)
	return y_class

def test_accuracy(predictions, actual_labels):
	"""
	Test how accurate the prediction was to the test cases labels
	"""
	
	confussion_stats = {'fp':0, 'fn':0, 'tp':0, 'tn':0}
	accurate = 0
	for i in range(len(actual_labels)):
		if actual_labels[i] == int(predictions[i]):
			accurate += 1
		# obtains the confussion matrix stats
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
	
def logistic_regression(train_dataset, test_dataset, alpha, iteration):
	"""
	function that is called by cross validation process
	"""
	y_train = np.array([train_dataset['label 36']])
	train_dataset.drop(train_dataset.columns[len(train_dataset.columns)-1], axis=1, inplace=True)
	x_train = train_dataset.as_matrix()
    # adding 1s to the first column for bias in x_train
	x_train = np.column_stack((np.ones((x_train.shape[0])),x_train))
	# initializing weights and bias
	weight = weightInitialization(len(list(train_dataset)))
	# alpha value; how big of a step to take for gradient descent
	# train model to get the values for w
	w = model_train(x_train, y_train, weight, alpha, iteration)
	
	y_test = np.array([test_dataset['label 36']])
	test_dataset.drop(test_dataset.columns[len(test_dataset.columns)-1], axis=1, inplace=True)
	x_test = test_dataset.as_matrix()


    #x_train = np.concatenate((np.ones((x_train.shape[1],1)),x_train),axis=0)
	x_test = np.column_stack((np.ones((x_test.shape[0])),x_test))

	prediction = predict(x_test, w)
	y_pred = classify(prediction.flatten('F').tolist())
	return(test_accuracy(y_pred, y_test.flatten('F').tolist()))

if __name__ == '__main__':
    dataset = pd.read_csv('01_preprocessed.csv')
    dataset = dataset.apply(pd.to_numeric, errors='ignore')
    splitRatio = 0.9
	# Actually split data into according parts
    train_dataset, test_dataset = train_test_split(dataset, splitRatio)
	# collect class values, then remove them from train_dataset
    y_train = np.array([train_dataset['label']])
    train_dataset.drop(train_dataset.columns[len(train_dataset.columns)-1], axis=1, inplace=True)
    x_train = train_dataset.as_matrix()
    # adding 1s to the first column for bias in x_train
    x_train = np.column_stack((np.ones((x_train.shape[0])),x_train))
	# initializing weights and bias
    weight = weightInitialization(len(list(train_dataset)))
	# alpha value; how big of a step to take for gradient descent
    alpha = 0.00001
	# train model to get the values for w
    w = model_train(x_train, y_train, weight, alpha, 10000)
	
    y_test = np.array([test_dataset['label']])
    test_dataset.drop(test_dataset.columns[len(test_dataset.columns)-1], axis=1, inplace=True)
    x_test = test_dataset.as_matrix()


    #x_train = np.concatenate((np.ones((x_train.shape[1],1)),x_train),axis=0)
    x_test = np.column_stack((np.ones((x_test.shape[0])),x_test))

    prediction = predict(x_test, w)
    y_pred = classify(prediction.flatten('F').tolist())
    print(test_accuracy(y_pred, y_test.flatten('F').tolist()))
