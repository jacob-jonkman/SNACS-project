import networkx as nx
import numpy as np
import math
from scipy.optimize import fmin
from scipy.spatial.distance import euclidean
import sys
from time import time

num_landmarks = 100
initial = 16
D = 10
xtol = 1
ftol = 1
maxfun = 10000
maxiter = 1000

# Find the k nodes with highest degree centrality
def choose_initial_landmarks(G):
	degreelist = G.degree(G.nodes())
	sortedlist = sorted(degreelist, key=lambda tup:int(tup[1]), reverse=True)
	return sortedlist[:num_landmarks], sortedlist[num_landmarks:]


# Compute actual distance between all landmark nodes using BFS
def compute_landmark_distances(landmarks, num_nodes, G):
	distances = np.zeros((num_landmarks, num_nodes))
	i=0
	for landmark in landmarks:
		#print(i)
		dists = nx.shortest_path_length(G, landmark[0])
		sorted_dists = np.array(sorted(dists.items(), key=lambda x:int(x[0])))
		distances[i,:] = sorted_dists[:,1]
		i+=1
	return distances

	
# Splits the distance matrix into two matrices: 
# -landmarkcols: distance from landmarks to other landmarks
# -regularcols: distance from landmarks to regular nodes
def split_distances(all_distances, landmark_indices, regular_indices):
	landmarkcols = all_distances[:,landmark_indices]
	regularcols = all_distances[:,regular_indices]	
	return landmarkcols, regularcols

	
# Objective function to be minimized by fmin used in initial_coordinates()
def diff_initial(coordinates, distances):
	coordinates = coordinates.reshape(initial, D)
	difference = 0
	for i in np.arange(initial):
		for j in np.arange(initial):
			summed = (coordinates[i,:] - coordinates[j,:])**2
			euclidean = math.sqrt(sum(summed))
			difference += np.abs(distances[i,j]-euclidean)
	return difference


# Apply simplex downhill algorithm to optimize coordinates of first 16 landmarks
def initial_coordinates(landmarks, landmark_distances):
	coordinates = np.random.rand(initial, D)*2
	return fmin(func=diff_initial, x0=coordinates, args=(landmark_distances,), xtol=xtol, ftol=ftol, maxfun=maxfun)


# Objective function to be minimized by fmin used in second_coordinates()
def diff_second(second_coordinates, first_coordinates, distances):
	second_coordinates = second_coordinates.reshape(num_landmarks-initial, D)
	difference = 0
	for i in np.arange(len(second_coordinates)):
		for j in np.arange(len(first_coordinates)):
			sums = (second_coordinates[i,:]-first_coordinates[j,:])**2
			euclidean = math.sqrt(sum(sums))
			difference += np.abs(distances[i,j]-euclidean)
	#print(difference)
	return difference


# Find coordinates of the other landmarks by using coordinates of first landmark nodes
def second_coordinates(first_coordinates, distances):
	coordinates = np.random.rand(num_landmarks-initial, D)*2
	return fmin(func=diff_second, x0=coordinates, args=(first_coordinates,distances,), xtol=xtol, ftol=ftol, maxfun=maxfun)


# Objective function to be minimized by fmin used in regular_coordinates()
def diff_regular(coordinates, landmark_coordinates, distances, num_nodes):
	difference = 0
	coordinates = coordinates.reshape(num_nodes-num_landmarks, D)
	#print(landmark_coordinates.shape)
	anchors = landmark_coordinates[np.random.permutation(num_landmarks)[:initial],:]
	#print(anchors.shape)
	
	for i in np.arange(num_landmarks):
		for j in np.arange(num_nodes-num_landmarks):
			sums = (coordinates[j,:] - landmark_coordinates[i,:])**2
			euclid = math.sqrt(sum(sums))
			#euclid = euclidean(coordinates[i,:], landmark_coordinates[j,:])
			difference += np.abs(distances[i,j] - euclid)
	#print("euclidean:", sys.getsizeof(euclidean), "difference:", sys.getsizeof(difference))
	#print("coordinates in objective func:", sys.getsizeof(coordinates))
	#print(difference)
	#del coordinates
	return difference


def diff_regular_per_node(coordinates, landmark_coordinates, distances, nodenr):
	difference = 0	
	for i in np.arange(num_landmarks):
		sums = (coordinates - landmark_coordinates[i,:])**2
		euclid = math.sqrt(sum(sums))
		difference += np.abs(distances[i] - euclid)
	#print("node:", nodenr, "difference:", difference)
	return difference


# Find coordinates of the regular ndoes by using coordinates of all landmark nodes
def regular_coordinates(landmark_coordinates, distances, num_nodes):
	coordinates = np.random.rand(num_nodes - num_landmarks, D)*3
	for i in np.arange(num_nodes - num_landmarks):
		print("calibrating node:", i)
		coordinates[i,:] = fmin(func=diff_regular_per_node, x0=coordinates[1,:], args=(landmark_coordinates,distances[:,i],i,), xtol=xtol, ftol=ftol, maxfun=maxfun, maxiter=maxiter)
	return coordinates
	
	#return fmin(func=diff_regular, x0=coordinates, args=(landmark_coordinates,distances,num_nodes,), xtol=xtol, ftol=ftol, maxfun=maxfun, maxiter=maxiter)

def main():
	np.random.seed(42)
	
	#G = nx.gnm_random_graph(200, 1000, seed=42, directed=False)
	G = nx.read_edgelist("facebook_combined.txt")
	num_nodes = G.number_of_nodes()
	num_edges = G.number_of_edges()
	print("Succesfully loaded network with", num_nodes, "nodes and", num_edges, "edges.")
	
	t = time()
	
	# Split the nodes in landmark nodes and regular nodes, and find the corresponding indices
	landmarks, regular_nodes = choose_initial_landmarks(G)
	landmark_indices = [int(landmark[0]) for landmark in landmarks]
	regular_indices = [int(node[0]) for node in regular_nodes]
	
	# Find the distance from each landmark nodes to all other nodes
	all_distances = compute_landmark_distances(landmarks, num_nodes, G)
	landmark_distances, regular_distances = split_distances(all_distances, landmark_indices, regular_indices)
	del landmark_indices
	del regular_indices
	
	# Start adding the nodes in three steps:
	# 1: Add the initial landmark nodes
	# 2: Add the remaining landmark nodes
	# 3: Add all the regular nodes
	print("Start finding coordinates for initial", initial, "landmark nodes")
	first_coords = initial_coordinates(landmarks[:initial], landmark_distances[:initial]).reshape(initial,  D)
	
	print("Start finding coordinates for the other", num_landmarks-initial, "landmark nodes")
	second_coords = second_coordinates(first_coords, landmark_distances[initial:num_landmarks]).reshape(num_landmarks-initial, D)
	
	# Combine landmark coordinates in single array
	landmark_coords = np.append(first_coords, second_coords).reshape(num_landmarks, D)
	
	del first_coords
	del second_coords
	
	print("Start finding coordinates for the", num_nodes-num_landmarks, "regular nodes")
	regular_coords = regular_coordinates(landmark_coords, regular_distances, num_nodes).reshape(num_nodes-num_landmarks, D)
	
	# Combine all coordinates in single array
	all_coordinates = np.append(landmark_coords, regular_coords).reshape(num_nodes, D)
	
	np.save("coordinates_facebook_small.npy", all_coordinates)
	
	print("preprocessing time:", t-time()


if __name__ == "__main__":
	main()
