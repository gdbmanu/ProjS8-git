
import numpy as np
import scipy as sp
import pywt


class WaveImage:
	def __init__(self, image = np.zeros((32,32)), shape = (32, 32)):
		if image.shape != shape :
			image = np.zeros(shape)
		
		self.__shape = image.shape
		
		# Decomposition ondelettes
		coeffs = pywt.wavedec2(image, 'haar')
		h_max = len(coeffs)
		
		# Test puissance de 2
		self.__puiss2 = shape[0] == 2**(h_max - 1) and shape[1] == 2**(h_max - 1)
		
		# L'attribut data contient les vecteurs en position [h][u] 
		self.__data = {-1 : {-1 : coeffs[0][0, 0]}}
		if self.isPuiss2():
			#coeffs[0] est la cste; coeffs[1] le vect a 3 coord
			print 'isPuiss2 = TRUE'
			self.__data[0] = {}
			self.__data[0][(0,0)] = np.append(coeffs[1][0], coeffs[1][1])
			self.__data[0][(0,0)] = np.append(self.__data[0][(0,0)], coeffs[1][2])
			self.__h_max = h_max - 1
			for h in range(1, self.__h_max):
				(i_max, j_max) = coeffs[h + 1][0].shape
				self.__data[h] = {}
				for i in range(i_max):
					for j in range(j_max):
						data = np.append(coeffs[h + 1][0][i][j], coeffs[h + 1][1][i][j])
						data = np.append(data, coeffs[h + 1][2][i][j])
						self.__data[h][(i, j)] = data
		else:
			#coeffs[0] contient un vect a 4 coord
			self.__h_max = h_max
			self.__data[0] = {}
			self.__data[0][(0,0)] = np.append(coeffs[0][0][1], coeffs[0][1][0])
			self.__data[0][(0,0)] = np.append(self.__data[0][(0,0)], coeffs[0][1][1])
			for h in range(1, self.__h_max):
				(i_max, j_max) = coeffs[h][0].shape
				self.__data[h] = {}
				for i in range(i_max):
					for j in range(j_max):
						data = np.append(coeffs[h][0][i][j], coeffs[h][1][i][j])
						data = np.append(data, coeffs[h][2][i][j])	
						self.__data[h][(i, j)] = data
					
	def isPuiss2(self):
		return self.__puiss2
		
	def getData(self):
		return self.__data
		
	def __str__(self):
		h_max = len(self.__data)
		print 'h_max :', h_max, self.__h_max
		s = ''
		for h in range(-1, self.__h_max):
			print h
			s += '***' + str(h) + '***\n'
			s += str(self.__data[h]) + '\n'
		return s
