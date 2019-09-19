import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import my_logistic_regression
import my_naive_bayes


def split_data(dataset, splits, Algorithm):
	# data sample
	# prepare cross validation
	kfold = KFold(n_splits = splits)
	# enumerate splits
	i = 0
	for train, test in kfold.split(dataset):
		#train = pd.DataFrame({'R':dataset[train][:,0],'G':dataset[train][:,1],'label':dataset[train][:,35]})
		#test = pd.DataFrame({'R':dataset[test][:,0],'G':dataset[test][:,1],'label':dataset[test][:,35]})
		train = pd.DataFrame(dataset[train], columns = ["label %d" % (i + 1) for i in range(36)])
		test = pd.DataFrame(dataset[test], columns = ["label %d" % (i + 1) for i in range(36)])
		if(Algorithm == 'NB'):
			print(my_naive_bayes.naive_bayes(train, test))
		elif (Algorithm == 'LR'):
			alpha = 0
			iterations = 0
			if i == 0:
				alpha = float(input("Please enter the alpha value: "))
				iterations = int(input("Please enter the number of iterations: "))
			print(my_logistic_regression.logistic_regression(train, test, alpha, iterations))
			i+= 1
	


def cross_validation(dataset, splits, algo):
	df1 = pd.read_csv(dataset)
	df1 = df1.apply(pd.to_numeric, errors='ignore')

	df1['label'] = pd.to_numeric(df1['label'])
	#df = convert(df)
	data = df1.as_matrix()
	split_data(data, splits, algo)


	
if __name__ == '__main__':
	split = input("Enter the number of splits for K-Fold: ")
	algo = input("Type NB for Naive Bayes, LR for For Logistic Regression: ")
	cross_validation('01_preprocessed.csv', int(split), algo)
