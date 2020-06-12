
import pandas as pd
import numpy as np

def calculate_mean_and_center(dataset):
	for column in dataset:
		mu = dataset[column].mean()
		dataset[column] -= mu
	return dataset

if __name__ == '__main__':

	dataset = pd.read_csv('preprocessed_data.csv')
	dataset.drop(dataset.columns[len(dataset.columns)-1], axis=1, inplace=True)
	#print(dataset.cov())
	centered = calculate_mean_and_center(dataset)
	# get covarience matrix
	covarience_matrix = centered.cov().values
	# find eigen values and eigen vectors
	eigen_value, eigen_vector = np.linalg.eig(covarience_matrix)
		
	
	# for each eigen value
	for i in range(len(eigen_value)):
		print(eigen_value[i])
	
	eigen_tuples = []
	for i in range(len(eigen_value)):
		eigen_tuples.append((np.abs(eigen_value[i]), eigen_vector[:,i]))
	
	eigen_tuples.sort(key=lambda x: x[0], reverse=True)
	
	
	matrix_w = np.hstack((eigen_tuples[0][1].reshape(40,1), eigen_tuples[1][1].reshape(40,1), eigen_tuples[2][1].reshape(40,1)))
	
	#components = 3
	#matrix_w = np.array([])
	#for i in range(components):
	#	matrix_w = np.hstack((matrix_w, (eigen_tuples[i][1].reshape(40,1)))
	
	print(matrix_w)
	projected = np.dot(matrix_w.T, dataset.T)
	transformed = projected.T
	
	np.savetxt("PCA3.csv", transformed, delimiter=",")
	
	
	
