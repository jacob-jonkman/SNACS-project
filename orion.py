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
	return sortedlist[:num_landmarks]

# Compute actual distance between all landmark nodes
def compute_landmark_distances(landmarks, G):
	distances = np.zeros((num_landmarks, num_landmarks))
	i=0
	for landmark1 in landmarks:
		j=0
		for landmark2 in landmarks:
			distances[i,j] = nx.shortest_path_length(G, landmark1[0], landmark2[0])
			j += 1
		i += 1
	return distances
		
# Objective function to be minimized by fmin used in initial_coordinates()
def diff_initial(coordinates, distances):
	difference = 0
	coordinates = coordinates.reshape(initial, D)
	
	for i in np.arange(initial):
		for j in np.arange(initial):
			summed = sum((coordinates[i,:] - coordinates[j,:])**2)
			euclidean = math.sqrt(summed)
			difference += np.abs(distances[i,j]-euclidean)
	
	return difference

# Apply simplex downhill algorithm to optimize coordinates of first 16 landmarks
def initial_coordinates(landmarks, landmark_distances):
	coordinates = np.random.rand(initial, D)*3
	return fmin(func=diff_initial, x0=coordinates, args=(landmark_distances,), xtol=1, ftol=10, maxfun=100000)

# Objective function to be minimized by fmin used in second_coordinates()
def diff_second(second_coordinates, first_coordinates, distances):
	second_coordinates = second_coordinates.reshape(num_landmarks-initial, D)
	difference = 0
	
	for i in np.arange(len(second_coordinates)):
		for j in np.arange(len(first_coordinates)):
			sums = (second_coordinates[i,:]-first_coordinates[j,:])**2
			euclidean = round(math.sqrt(sum(sums)))
			difference += np.abs(distances[i,j]-euclidean)
			#print(difference, np.abs(distances[i,j]-euclidean), distances[i,j], euclidean)
	#print(difference)	
	return difference

# Find coordinates of the other landmarks by using coordinates of first landmark nodes
def second_coordinates(second_landmarks, first_coordinates, distances):
	coordinates = np.random.rand(num_landmarks-initial, D)*3
	return fmin(func=diff_second, x0=coordinates, args=(first_coordinates,distances,), xtol=1.0, ftol=10, maxfun=100000)

# Use Dijkstra's algorithm to compute distance from all nodes to all landmark nodes
def find_distances_to_landmarks(landmarks, G):
	num_nodes = G.number_of_nodes()
	distances = np.zeros((num_nodes, num_landmarks))
	nodes = G.nodes()
	
	j=0
	for landmark1 in landmarks:
		i=0
		for node in nodes:
			if nx.has_path(G, landmark1[0], node):
				distances[i,j] = nx.shortest_path_length(G, landmark1[0], node)
			else:
				distances[i,j] = -1
			i += 1
		j += 1
	return distances

def distance(node1, node2, distances):
	dists = np.array(distances)
	print(node1)
	print(node2)
	print
	dist = np.min(distances[node1,:] + distances[node2,:])
	return dist
	

def main():
	np.random.seed(42)
	
	#G = nx.read_edgelist("twitter-small.csv", delimiter=",", nodetype=str)
	G = nx.gnm_random_graph(1000, 1000, seed=42, directed=False)
	num_nodes = G.number_of_nodes()
	
	landmarks = choose_initial_landmarks(G) # Choose all k landmarks	
	landmark_distances = compute_landmark_distances(landmarks, G)
	
	first_coordinates = initial_coordinates(landmarks[:initial], landmark_distances[:initial]).reshape(initial,  D)
	second_coords = second_coordinates(landmarks[initial:num_landmarks], first_coordinates, landmark_distances[initial:num_landmarks]).reshape(num_landmarks-initial, D) # Other landmarks
	
	landmark_coordinates = np.append(first_coordinates, second_coordinates)	
	dists_to_landmarks = find_distances_to_landmarks(landmarks, G)
	
	nodes = G.nodes()
	error = 0
	for node1 in nodes:
		for node2 in nodes:
			dist = distance(node1, node2, dists_to_landmarks)
			if nx.has_path(G, node1, node2):
				error += dist - nx.shortest_path_length(G, node1, node2)
	
			print("Distance between node", node1, "and", node2, "is estimated as:", dist)
			print()
	
	print("total error: ", error, "relative error:", error/(num_nodes**2))

if __name__ == "__main__":
	main()
