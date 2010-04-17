from math import log
import numpy as np
from distance import euclidean
import sys
from grid import square_grid, hexagonal_grid
from kernel import radial_kernel

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

class SelfOrganizingMap(object):
	"""Self-Organizing Map (SOM)
	
	Class that encapsulates the following data:
	1. the trained, high-dimension prototypes
	2. their corresponding grid structure.
	
	"""
	def __init__(self, p, width, height, grid_type = square_grid, distance = euclidean):
		"""
		p is an integer, defining the expected dimension of our future data.
		width is an integer, defining the width of the cell map.
		height is an integer, defining the height of the cell map."""
		self.alpha = 1
		
		self.distance = distance
		
		self.p = p
		self.width = width
		self.height = height
		self.prototype_vector = np.zeros([width*height, p])
		
		self.map = grid_type(width, height)
		
		self.grid_type = grid_type
		
		self.ymax_ind = width - 1
		self.xmax_ind = height - 1
		self.ymax = width
		self.xmax = height
		self._initialized_prototypes = 0
		self._total_prototypes = width * height
		
		self._base_neighborhood = max(max(self.map[:,0]), max(self.map[:,1])) / 2.0
	
	def online_train(self, data):
		pass
	
	def train(self, data):
		"""
		v0.1 - trains on """
		
		n, p = data.shape
		if p != self.p: 
			raise ValueError, "data dimensions do not match what you specified in the constructor."
		
		n_proto, p_proto = self.prototype_vector.shape
		if self._initialized_prototypes < n_proto:
			"""
			
			TO IMPLEMENT:
			
			While there is still data left in the data var,
				initialize a prototype with the data.
			if we have not sufficiently initialized everyone,
				store data somehow until we can start training in earnest.
				call it self.data_backlog = {timestamp: matrix, timestamp2: matrix2, ... }
			The moment all the prototypes have been initialized,
				clear the backlog first.
				then keep training on the other data.
			"""
			pass
		
		############################################
		# Initialize by taking random data points. #
		############################################
		
		sample = np.random.permutation(range(n))[0:(self._total_prototypes)]
		# set the various prototype vectors to these.
		for p, i in enumerate(sample):
			self.prototype_vector[p,:] = data[i,:]
		
		# Permute data point indices, and then loop over them.
		
		data_points = np.random.permutation(range(n))
		iterations = len(data_points)
		self.iterations = iterations
		
		t = 0
		
		for i in data_points:
			# get data point, initialize local variables.
			x_i = np.array(data[i,:])[0] # the [0] turns this into a 1d array.
			
			# find nearest prototype to x_i. Return int index.
			best_matching_unit = self.find_BMU(x_i)
			
			# get the prototype's neighbors. returns list of int indices.
			prototype_neighbors = self.get_neighbors(best_matching_unit, t)
			
			# move all prototype neighbors toward this point.
			for neighbor in prototype_neighbors:
				m_i = self.prototype_vector[neighbor,:]
				self.prototype_vector[neighbor,:] = m_i + self.alpha * (x_i - m_i)
			# increment bookmarks.
			t += 1
			self.alpha -= 1 / float(iterations)  
	
	def find_BMU(self, v):
		closest_prototype = -1
		closest_distance = -1
		for i, pv in enumerate(self.prototype_vector):
			dist = self.distance(v, pv)
			if dist < closest_distance or closest_distance == -1:
				closest_distance = dist
				closest_prototype = i
		return closest_prototype
	
	def get_neighbors(self, i, t):
		## hard-code the neighborhood kernel, for now :*(
		# calculate the expected radius of our maps.
		l = self.iterations / log(self._base_neighborhood)
		neighborhood_radius = self._base_neighborhood * radial_kernel(t, l)
		bmu = np.array(self.map[i,:])
		neighbors = []
		for j, v in enumerate(self.map):
			v = np.array(v)
			if self.distance(v, bmu) <= neighborhood_radius and j != i:
				neighbors.append(j)
		return neighbors
	
	def preview_3d_data_and_map(self, data):
		"""This is for demo purposes only.
		Demonstrates the self organizing map embedded on its
		corresponding prototype vectors in R^3."""
		fig = plt.figure()
		ax = Axes3D(fig)
		ax.scatter(data[:,0], data[:,1], data[:,2], color="#666666")
		ax.scatter(self.prototype_vector[:,0], self.prototype_vector[:,1], self.prototype_vector[:,2])
		# 
		# The following code is a bit boilerplate, but it does the job.
		# 
		for i, v1 in enumerate(self.map):
			for j, v2 in enumerate(self.map):
				if    ((v2[0] >= v1[0] and v2[1] >= v1[1]) or \
				      (v2[0] > v1[0] and v2[1] < v1[1])) and \
						self.distance(v2,v1) <= 1.03:
					p1 = self.prototype_vector[i,:]
					p2 = self.prototype_vector[j,:]
					p = np.vstack([p1, p2])
					ax.plot(list(p[:,0]), list(p[:,1]), list(p[:,2]), linewidth=1.5, color='k', alpha=.7)
		
		nn_vec = self.calculate_nearest_prototypes(data)
		n, p = data.shape
		for i in range(n):
			v = np.array(data[i,:])[0]
			m = np.array(self.prototype_vector[nn_vec[i],:])
			vs = np.vstack([v, m])
			ax.plot(list(vs[:,0]),list(vs[:,1]),list(vs[:,2]), linewidth=.5, color='#333333', alpha=.3)
		plt.show()
	
	def calculate_nearest_prototypes(self, data):
		n, p = data.shape
		
		nn_vec = np.zeros(n)
		
		for i in range(n):
			v = np.array(data[i,:])[0]
			bmu_i = self.find_BMU(v)
			nn_vec[i] = bmu_i
		return nn_vec
	
	def plot_SOM(self):
		"""
		TODO:
		1. read literature on expressing maps.
		2. work out particulars for each grid type.
		3. 
		"""
		pass
	
	def format_SOM(self, format="json"):
		"""
		TODO:
		1. implement the json formatting necessary for displaying
		the SOM using protovis or processing.js.
		2. implement csv functionality.
		"""
		pass

def uniform_test_data():
	data = np.transpose(np.matrix([
	 	 	 		np.random.uniform(0,5,1000),
	 	 	 		np.random.uniform(0,5,1000),
	 	 	 		np.random.uniform(0,.0001,1000)
	 	 	 	]))
	return data

def spheres_data():
	class_a = np.transpose(np.matrix([
		np.random.normal(10,1,1000),
		np.random.normal(0,1,1000),
		np.random.normal(10,1,1000)
	]))
	class_b = np.transpose(np.matrix([
		np.random.normal(5,1,1000),
		np.random.normal(5,1,1000),
		np.random.normal(5,1,1000)
	]))
	class_c = np.transpose(np.matrix([
		np.random.normal(1,1,1000),
		np.random.normal(1,1,1000),
		np.random.normal(1,1,1000)
	]))
	data = np.vstack([class_a, class_b, class_c])
	return data

def curve_data():
data = np.transpose(np.matrix([
	np.random.normal(0,1,20000),
	np.random.normal(0,1,20000),
	np.random.normal(0,1,20000)
]))
indices = []
n, p = data.shape
for i in range(n):
	if (euclidean(np.array(data[i,:])[0], np.array([0,0,0])) > .5) and \
	   (data[i,0] > 0 and data[i,1] > 1):
		indices.append(i)
new_data = np.zeros([len(indices), 3])
for i, j in enumerate(indices):
	new_data[i,:] = data[j,:]
new_data = np.matrix(new_data)
	return new_data

def main():
	#data = spheres_data()
	#data = curve_data()
	data = uniform_test_data()
	# verify data with 3d matplotlib scatterplot fcn.
	som = SelfOrganizingMap(3, 20, 20)#, grid_type=hexagonal_grid)
	som.train(data)
	som.preview_3d_data_and_map(data)

if __name__ == "__main__":
	main()
	