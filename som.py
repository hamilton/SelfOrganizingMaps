from math import log
import numpy as np
from distance import euclidean
import sys
from grid import square_grid
from kernel import radial_kernel

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

class SelfOrganizingMap(object):
	"""Self-Organizing Map
	
	"""
	def __init__(self, p, width, height):
		"""
		p is an integer, defining the expected dimension of our future data.
		width is an integer, defining the width of the cell map.
		height is an integer, defining the height of the cell map."""
		self.p = p
		self.width = width
		self.height = height
		self.prototype_vector = np.zeros([width*height, p])
		
		self.map = square_grid(width, height)
		
		self.ymax_ind = width - 1
		self.xmax_ind = height - 1
		self.ymax = width
		self.xmax = height
		self._initialized_prototypes = 0
		self._total_prototypes = width * height
		
		self._base_neighborhood = max(width, height) / 2.0
	
	def online_train(self, data):
		pass
	
	def train_dataset_v01(self, data, distance = euclidean):
		"""
		v0.1 - trains on """
		
		n, p = data.shape
		if p != self.p: 
			raise ValueError, "data dimensions do not match what you specified in the constructor."
		
		############################################
		# Initialize by taking random data points. #
		############################################
		
		sample = np.random.permutation(range(n))[0:(self.width * self.height)]
		# set the various prototype vectors to these.
		for p, i in enumerate(sample):
			self.prototype_vector[p,:] = data[i,:]
		
		# Permute data point indices, and then loop over them.
		
		data_points = np.random.permutation(range(n))
		iterations = len(data_points)
		self.iterations = iterations
		alpha = 1
		
		t = 0
		
		for i in data_points:
			# get data point, initialize local variables.
			x_i = np.array(data[i,:])[0] # the [0] turns this into a 1d array.
			
			# find nearest prototype. Returns int index.
			best_matching_unit = self.find_BMU(x_i)
			
			# get the prototype's neighbors. returns list of int indices.
			prototype_neighbors = self.get_neighbors(best_matching_unit, t)
			
			# move all prototype neighbors toward this point.
			for neighbor in prototype_neighbors:
				m_i = self.prototype_vector[neighbor,:]
				self.prototype_vector[neighbor,:] = m_i + alpha * (x_i - m_i)
			# increment bookmarks.
			t += 1
			alpha -= 1 / float(iterations)
		
		# fig = plt.figure()
		# 		ax = Axes3D(fig)
		# 		ax.scatter(self.prototype_vector[:,0],self.prototype_vector[:,1], self.prototype_vector[:,2])
		# 		
		# 		#a = self.map[0:1,:]
		# 		a = self.prototype_vector[0:1,:]
		# 		ax.plot(a[:,0], a[:,1], a[:,2])
		# 		
		# 		plt.show()   
	
	def find_BMU(self, v, distance = euclidean):
		closest_prototype = -1
		closest_distance = -1
		for i, pv in enumerate(self.prototype_vector):
			dist = distance(v, pv)
			if dist < closest_distance or closest_distance == -1:
				closest_distance = dist
				closest_prototype = i
		return closest_prototype
	
	def get_neighbors(self, i, t, distance = euclidean):
		## hard-code the neighborhood kernel, for now :*(
		# calculate the expected radius of our maps.
		l = self.iterations / log(self._base_neighborhood)
		neighborhood_radius = self._base_neighborhood * radial_kernel(t, l)
		bmu = np.array(self.map[i,:])
		neighbors = []
		for j, v in enumerate(self.map):
			v = np.array(v)
			if distance(v, bmu) <= neighborhood_radius and j != i:
				neighbors.append(j)
		return neighbors
	

if __name__ == "__main__":
	
	class_a = np.transpose(np.matrix([
		np.random.normal(4,1,1000),
		np.random.normal(4,1,1000),
		np.random.normal(4,1,1000)
	]))
	class_b = np.transpose(np.matrix([
		np.random.normal(1,1,1000),
		np.random.normal(1,1,1000),
		np.random.normal(1,1,1000)
	]))
	data = np.vstack([class_a, class_b])
	# data = np.transpose(np.matrix([
	# 		np.random.uniform(0,5,2000),
	# 		np.random.uniform(0,5,2000),
	# 		np.random.uniform(0,.0001,2000)
	# 	]))
	# verify data with 3d matplotlib scatterplot fcn.
	som = SelfOrganizingMap(3, 20, 30)
	som.train_dataset_v01(data)
	
	fig = plt.figure()
	ax = Axes3D(fig)
	ax.scatter(data[:,0], data[:,1], data[:,2], color="#666666")
	ax.scatter(som.prototype_vector[:,0], som.prototype_vector[:,1], som.prototype_vector[:,2])
	
	#for i in range(width):
	#	for j in range(height):
	#		
	for i, v1 in enumerate(som.map):
		for j, v2 in enumerate(som.map):
			if (v2[0] == v1[0] + 1 and v2[1] == v1[1]) or \
				(v2[1] == v1[1] + 1 and v2[0] == v1[0]):
				# get prototype equivalents for i and j,
				p1 = som.prototype_vector[i,:]
				p2 = som.prototype_vector[j,:]
				p = np.vstack([p1, p2])
				ax.plot(list(p[:,0]), list(p[:,1]), list(p[:,2]), linewidth=1, color='k')
	
	plt.show()