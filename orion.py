import networkx as nx
import numpy as np
import math
from scipy.optimize import fmin
num_landmarks = 100
initial = 16
D = 5

# Find the k nodes with highest degree centrality
def choose_initial_landmarks(G):
	degreelist = G.degree(G.nodes())
	sortedlist = sorted(degreelist, key=lambda tup: tup[1], reverse=True)
	return sortedlist


def compute_landmark_distances(landmarks, G):
	distances = np.zeros((initial, initial))
	i=0
	for landmark1 in landmarks:
		j=0
		for landmark2 in landmarks:
			distances[i,j] = nx.shortest_path_length(G, landmark1[0], landmark2[0])
			j += 1
		i += 1
	return distances
		
# Objective function to be minimized by fmin
def diff(coordinates, distances):
	difference = 0
	coordinates = coordinates.reshape(initial, D)
	for i in np.arange(initial):
		for j in np.arange(initial):
			summed = sum((coordinates[i,:] - coordinates[j,:])**2)
			euclidean = math.sqrt(summed)
			difference += np.abs(distances[i,j]-summed)
	return difference

# Apply simplex downhill algorithm to optimize coordinates of first 16 landmarks
def simplex_downhill(landmarks, landmark_distances, G):
	coordinates = np.random.rand(initial, D)*3
	fmin(func=diff, x0=coordinates, args=(landmark_distances,), xtol=0.001, ftol=1, maxfun=100000)

# Find coordinates of the other landmarks by using coordinates of first landmark nodes
def find_other_landmark_coords(landmarks, first_coordinates, G):
	pass

# Use BFS to compute distance from all nodes to all landmark nodes
def find_regular_coordinates(landmarks, G):
	pass

def main():
	#G = nx.read_edgelist("twitter-small.csv", delimiter=",", nodetype=str)
	G = nx.gnm_random_graph(500, 1000, seed=42, directed=False)
	
	landmarks = choose_initial_landmarks(G) # Choose all k landmarks	
	landmark_distances = compute_landmark_distances(landmarks[:16], G)
	first_coordinates = simplex_downhill(landmarks[:16], landmark_distances, G) # Only first 16 landmarks
	#more_coordinates = find_other_landmark_coords(landmarks[16,:], first_coordinates, G) # Other landmarks
	#all_coordinates = find_regular_coordinates(landmarks, G)
	# #

if __name__ == "__main__":
	main()
