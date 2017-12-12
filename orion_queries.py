import numpy as np
import networkx as nx
import argparse as ap
import math
from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import euclidean_distances
from time import time

k = 1000

# Use Dijkstra's algorithm to compute distance from all nodes to all landmark nodes
def find_distances(node, G):
	dists = nx.shortest_path_length(G, node)
	sorted_dists = np.array(sorted(dists.items(), key=lambda x:int(x[0])))
	distances = sorted_dists[:,1].astype(int)
	return distances
		
def main():
	t = time()
	
	parser = ap.ArgumentParser(description='Run orion preprocessing.')
	parser.add_argument('fin',
											help='path to the file containing the network')
	parser.add_argument('coords',
											help='path to the file containing the network coordinates')
	
	args = parser.parse_args()
	
	coordinates = np.load(args.coords)	
	
	G = nx.read_edgelist(args.fin)
	tot_nodes = G.number_of_nodes()
	
	# Extract largest connected component
	Gcc=sorted(nx.connected_component_subgraphs(G), key = len, reverse=True)
	G=Gcc[0]
	
	num_nodes = G.number_of_nodes()
	num_edges = G.number_of_edges()
	
	# Rename nodes if G did not consist of a single connected component
	if num_nodes != tot_nodes:
		graph = nx.to_dict_of_lists(G)
		connected_nodes = graph.keys()
		mapping = dict(zip(connected_nodes, np.arange(num_nodes)))
		G = nx.relabel_nodes(G, mapping)

	print("Succesfully loaded network with", num_nodes, "nodes and", num_edges, "edges.")
			
	# Randomly choose k nodes to compare
	nodes = np.random.permutation(G.nodes())[:k]
	nodes_as_indices = np.array([int(x) for x in nodes])
	
	# Compute pair-wise distance with BFS as well as euclidean distance for all k*k node pairs
	diameter = 0
	diff = 0
	i=0
	for node in nodes:
		print(i,"th node")
		i+=1
		distances = find_distances(node, G)[nodes_as_indices]
		euclidean = euclidean_distances(coordinates[int(node)].reshape(1,-1), coordinates[nodes_as_indices,:])
		difference = abs(distances-euclidean[0,:])
		diff += np.sum(np.divide(difference, distances, out=np.zeros_like(difference), where=distances!=0))
		diameter = max(diameter, np.max(euclidean))
	
	print("RAE:", diff/(k*k))
	print("diameter:", diameter)
	print("program took", time() - t, "seconds to run using a sample size of", k)

if __name__ == "__main__":
	main()
