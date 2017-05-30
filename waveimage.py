
import numpy as np
import scipy as sp
import pywt
import math


class WaveImage:
	
	
	def __init__(self, image = None, shape = (32, 32)):
		
		# Attribut shape
		if image is not None:
			# Decomposition ondelettes
			coeffs = pywt.wavedec2(image, 'haar')
			self.__shape = image.shape
		else:
			self.__shape = shape		
		
		# Attribut h_max : profondeur de l'image
		self.__h_max = min(int(math.log(self.__shape[0], 2)) + 1, 	int(math.log(self.__shape[1], 2)) + 1)
			
		# Attribut data : L'attribut data contient les vecteurs en position [h][u] (dictionnaire)
		if image is not None:
			self.__data = {}
			for h in range(self.__h_max):
				self.__data[h] = {}
				if h == 0:
					(i_max, j_max) = coeffs[h].shape
				else:
					(i_max, j_max) = coeffs[h][0].shape
				for i in range(i_max):
					for j in range(j_max):
						if h == 0:
							data = coeffs[h][i][j]
						else:
							data = coeffs[h][0][i][j]
							for k in range(1,len(coeffs[h])):
								data = np.append(data, coeffs[h][k][i][j])	
						self.__data[h][(i, j)] = data				
		else: # image is None
			self.__data = {}
			for h in range(self.__h_max):
				self.__data[h] = {}
					
		
	def getData(self):
		return self.__data
		
	def getHmax(self):
		return self.__h_max
		
	def getImage(self):
		coeffs = []
		for h in range(self.__h_max):
			if h == 0:
				dim_i = int(math.ceil(self.__shape[0] * 1. / 2**(self.__h_max - h - 1)))
				dim_j = int(math.ceil(self.__shape[1] * 1. / 2**(self.__h_max - h - 1)))
				coeffs_h = np.zeros((dim_i, dim_j))
				for u in self.__data[h]:
					coeffs_h[u[0],u[1]] = self.__data[h][u]
			else:
				dim_i = int(math.ceil(self.__shape[0] * 1. / 2**(self.__h_max - h)))
				dim_j = int(math.ceil(self.__shape[1] * 1. / 2**(self.__h_max - h)))
				coeffs_h = [np.zeros((dim_i, dim_j)), np.zeros((dim_i, dim_j)), np.zeros((dim_i, dim_j))]
				for u in self.__data[h]:
					for k in range(3):
						coeffs_h[k][u[0],u[1]] = self.__data[h][u][k]
			coeffs += [coeffs_h]
		return pywt.waverec2(coeffs, 'haar')	
		
	def add_coeffs(self, waveImage, u, h_ref = 0):
		if self.__data[0] == {}:
			self.__data[0][(0,0)] = np.copy(waveImage.getData()[0][(0,0)])
		else:
			v_test = self.__data[0][(0,0)]
			if np.linalg.norm(v_test) < 1e-16:
				for u_0 in waveImage.getData()[0]:
					self.__data[0][u_0] = np.copy(waveImage.getData()[0][u_0])
		for h in range(1, h_ref) :
			h_opp = self.__h_max - h
			i = int(u[0] / 2**h_opp) 
			j = int(u[1] / 2**h_opp)
			#print i, j
			if (i,j) in self.__data[h]:
				v_test = self.__data[h][(i,j)]
				if np.linalg.norm(v_test) < 1e-16:
					self.__data[h][(i,j)] = np.copy(waveImage.getData()[h][(i,j)])
			else: 
				self.__data[h][(i,j)] = np.copy(waveImage.getData()[h][(i,j)])
				
			
		
		
	def __str__(self):
		h_max = len(self.__data)
		s = 'h_max :' + str(self.__h_max) + '\n'
		for h in range(self.__h_max):
			s += '***' + str(h) + '***\n'
			s += str(self.__data[h]) + '\n'
		return s
