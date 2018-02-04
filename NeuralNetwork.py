import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import pandas.io.common
import os
import os.path

import scipy as sio
from scipy.io import loadmat

from PIL import Image

from scipy.optimize import minimize

class NeuralNetwork():

	def __init__(self, path_data, path_weigth):

		try:
			self._df = loadmat(path_data)
		except IOError:
			print("Data file mot found")
			raise
		try:
			self._w = loadmat(path_weigth)
		except IOError:
			print("Weigth file mot found")
			raise

		self.X = self._df['X']
		self.X = np.c_[np.ones((self.X.shape[0], 1)), self.X]
		self.y = self._df['y']
		self._l = len(self.y)
		self.theta1 = self._w["Theta1"]
		self.theta2 = self._w["Theta2"]
		self._shapet1 = self.theta1.shape
		self._shapet2 = self.theta1.shape

	def sigmoid(self, data):

		return (1 / (1 + np.exp(-data)))

	def plotNumber(self, X, number):

		fig = plt.figure()
		array = self.X[number, 1:].reshape(-1, 20).T
		plt.imshow(array, cmap=None)
		plt.axis('off')
		plt.show()

	def plotRandomNumber(self, X):

		number = np.random.randint(self._l)
		self.plotNumber(X, number)

	def predict(self, theta1, theta2, X, y):

		# print(theta1.shape)
		# print(X.shape)
		hidden_layer = self.sigmoid(np.dot(theta1, X.T))
		hidden_layer = np.r_[np.ones((1, hidden_layer.shape[1])), hidden_layer]
		output_layer = self.sigmoid(np.dot(theta2, hidden_layer)).T
		predict = np.argmax(output_layer, axis=1)
		predict += 1
		return (predict)

	def accurancy(self, prediction, y):

		print('Accuracy = {0}'.format(np.mean(prediction == y.ravel()) * 100))

def main():
	path_data = os.getcwd() + '/ex3data1.mat'
	path_weigth = os.getcwd() + '/ex3weights.mat'
	NN = NeuralNetwork(path_data, path_weigth)
	NN.plotRandomNumber(NN.X)
	p = NN.predict(NN.theta1, NN.theta2, NN.X, NN.y)
	NN.accurancy(p, NN.y)

if __name__ == '__main__':
    main()
