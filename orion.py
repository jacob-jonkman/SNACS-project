import networkx as nx
import numpy as np
import argparse as ap
import math
from scipy.optimize import fmin
from scipy.spatial.distance import euclidean
from time import time
from sklearn.metrics.pairwise import euclidean_distances

num_landmarks = 100
initial = 16
D = 15
xtol = 1
ftol = 1
maxfun = 100000
maxiter = 100000

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
		dists = nx.shortest_path_length(G, landmark[0])
		sorted_dists = np.array(sorted(dists.items(), key=lambda x:int(x[0])))
		distances[i,:] = sorted_dists[:,1]
		i+=1
	return distances


# Splits the distance matrix into two matrices: 
# -landmarkcols: distance from landmarks to other landmarks
# -regularcols: distance from landmarks to regular nodes
def split_distances(all_distances, landmark_indices, regular_indices):
	print(all_distances.shape)
	landmarkcols = all_distances[:,landmark_indices]
	regularcols = all_distances[:,regular_indices]
	return landmarkcols, regularcols

	
# Objective function to be minimized by fmin used in initial_coordinates()
def diff_initial(coordinates, distances):
	coordinates = coordinates.reshape(initial, D)
	euclids = euclidean_distances(coordinates)
	difference = np.abs(np.sum(euclids-distances[:,initial]))
	print(difference)
	return difference


# Apply simplex downhill algorithm to optimize coordinates of first 16 landmarks
def initial_coordinates(landmarks, landmark_distances):
	coordinates = (np.random.rand(initial, D))*2
	return fmin(func=diff_initial, x0=coordinates, args=(landmark_distances,), xtol=xtol, ftol=ftol, maxfun=maxfun)


# Objective function to be minimized by fmin used in second_coordinates()
def diff_second(second_coordinates, first_coordinates, distances):
	second_coordinates = second_coordinates.reshape(num_landmarks-initial, D)
	euclids = euclidean_distances(second_coordinates, first_coordinates)
	difference = np.abs(np.sum(euclids-distances[:,:initial]))	
	print(difference)
	return difference


# Find coordinates of the other landmarks by using coordinates of first landmark nodes
def second_coordinates(first_coordinates, distances):
	coordinates = (np.random.rand(num_landmarks-initial, D))*2
	return fmin(func=diff_second, x0=coordinates, args=(first_coordinates,distances,), xtol=xtol, ftol=ftol, maxfun=maxfun)


def diff_regular_per_node(coordinates, landmark_coordinates, distances, nodenr):
	euclids = euclidean_distances(coordinates.reshape(1,-1), landmark_coordinates)
	difference = np.abs(np.sum(euclids-distances))
	return difference


# Find coordinates of the regular ndoes by using coordinates of all landmark nodes
def regular_coordinates(landmark_coordinates, distances, num_nodes):
	coordinates = (np.random.rand(num_nodes - num_landmarks, D))*3
	for i in np.arange(num_nodes - num_landmarks):
		print("calibrating node:", i)
		coordinates[i,:] = fmin(func=diff_regular_per_node, x0=coordinates[i,:], args=(landmark_coordinates,distances[:,i],i,), xtol=xtol/10, ftol=ftol/10, maxfun=maxfun, maxiter=maxiter)
	return coordinates

def main():	
	t = time()
	
	parser = ap.ArgumentParser(description='Run orion preprocessing.')
	parser.add_argument('fin',
											help='path to the file containing the network')
	parser.add_argument('fout',
											help='path to the .npy-file to write the labels')
	parser.add_argument('--xopt',
											help='xopt to use in nelder-mead',
											default=1)
	parser.add_argument('--fopt',
											help='fopt to use in nelder-mead',
											default=1)
	args = parser.parse_args()
		
	G = nx.read_edgelist(args.fin)
	tot_nodes = G.number_of_nodes()
	tot_edges = G.number_of_edges()
	
	Gcc=sorted(nx.connected_component_subgraphs(G), key = len, reverse=True)
	G=Gcc[0]
	graph = nx.to_dict_of_lists(G)
	connected_nodes = graph.keys()
	
	num_nodes = G.number_of_nodes()
	num_edges = G.number_of_edges()
	
	mapping = dict(zip(connected_nodes, np.arange(num_nodes)))
	G = nx.relabel_nodes(G, mapping)
	
	print("Succesfully loaded network with", num_nodes, "nodes and", num_edges, "edges.")
		
	# Split the nodes in landmark nodes and regular nodes, and find the corresponding indices
	landmarks, regular_nodes = choose_initial_landmarks(G)
	landmark_indices = [int(landmark[0]) for landmark in landmarks]
	regular_indices = [int(node[0]) for node in regular_nodes]
	
	# Find the distance from each landmark nodes to all other nodes
	all_distances = compute_landmark_distances(landmarks, num_nodes, G)
	landmark_distances, regular_distances = split_distances(all_distances, landmark_indices, regular_indices)
	
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
		
	print("Start finding coordinates for the", num_nodes-num_landmarks, "regular nodes")
	regular_coords = regular_coordinates(landmark_coords, regular_distances, num_nodes).reshape(num_nodes-num_landmarks, D)
	
	# Combine all coordinates in single array
	all_coordinates = np.append(landmark_coords, regular_coords).reshape(num_nodes, D)
	
	np.save(args.fout, all_coordinates)
	
	print("preprocessing time:", time()-t, "seconds")


if __name__ == "__main__":
	main()
