import numpy as np

def square_grid(width, height):
	"""
	INPUT: width and height
	OUTPUT: a data set of coordinates in 
	a two dimensional Euclidean space, each
	exactly width and height apart from each other.
	
	if plotted, the coordinates produce a square grid:
	
	*---*---* ...
	|   |   |
	*---*---* ...
	|   |   |
	*---*---* ...
	"""
	grid = np.zeros([width*height, 2])
	i = 0
	for y in range(height):
		for x in range(width):
			grid[i,0] = x# x coord.
			grid[i,1] = y# y coord.
			i += 1
	return grid

def hexagonal_grid(width, height):
	pass
	

if __name__ == "__main__":
	a = square_grid(4,4)
	print a