import numpy as np
import networkx as nx
import argparse as ap
import math

k = 3000

# Use Dijkstra's algorithm to compute distance from all nodes to all landmark nodes
def find_distances(nodes, G):
	distances = np.zeros((k, k))
	
	j=0
	for node1 in nodes:
		i=0
		for node2 in nodes:
			if nx.has_path(G, node1, node2):
				distances[i,j] = nx.shortest_path_length(G, node1, node2)
			else:
				distances[i,j] = -1
			i += 1
		j += 1
	return distances

def euclidean_distances(nodes, coordinates):
	distances = np.zeros((k, k))
	
	for i in np.arange(k):
		for j in np.arange(k):
			distances[i,j] = math.sqrt(sum((coordinates[i,:] - coordinates[j,:])**2))
	
	return distances
	
def estimate_diameter(coordinates):
	euclideans = np.zeros((k, k))
	
	for i in np.arange(k):
		for j in np.arange(k):
			euclideans[i,j] = math.sqrt(sum((coordinates[i,:] - coordinates[j,:])**2))
	
	return np.max(euclideans)

def main():

	parser = ap.ArgumentParser(description='Run orion preprocessing.')
	parser.add_argument('fin',
											help='path to the file containing the network coordinates')
	
	args = parser.parse_args()
	
	coordinates = np.load(args.fin)	
	print(coordinates)
	
	np.random.seed(42)
	#G = nx.gnm_random_graph(200, 1000, seed=42, directed=False)
	#G = nx.read_edgelist("facebook_combined.txt")
	
	# Randomly choose k nodes to compare
	#nodes = np.random.permutation(G.nodes())[:k]

	# Compute euclidean distance and actual distance #
	#distances = find_distances(nodes, G)
	#euclidean = euclidean_distances(nodes, coordinates)
	
	#print("relative error:", np.sum(abs(distances-euclidean))/k**2)
	print("diameter:", estimate_diameter(coordinates))

if __name__ == "__main__":
	main()
