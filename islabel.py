"""
Implementation of the IS-Label (Independent Set based labels) algorithm

Todo:
- Change lists to np arrays (would require calculating upper bounds and
  dealing with unset values, but perhaps worth it)
- Look into optimising label generation

"""
from __future__ import print_function
from __future__ import division

import numpy as np
import networkx as nx
import argparse as ap
import pickle as pi
import time as t


'''
Adds an augmenting edge (n, m) if necessary.
'''
def augment(network, n, m, o):
	new_dist = network[n][m]['weight']
	new_dist += network[m][o]['weight']
	
	if network.has_edge(n, o):
		if network[n][o]['weight'] > new_dist:
			network[n][o]['weight'] = new_dist
	else:
		network.add_edge(n, o, attr_dict={'weight' : new_dist})

'''
Adds all necessary aumenting edges to the graph between the provided
nodes.
'''
def add_augmenting_edges(network, node, neighbours):
	for n in neighbours:
		for m in neighbours:
			if n < m:
				augment(network, n, node, m)

'''
Generates a list of adjacent nodes for each node in the network.
Then it generates a list of these lists, which is sorted by degree
of the nodes (or: the number of adjacent nodes in each list).
'''
def sorted_adjs(network):
	adjacents = list()
	for i in network.nodes():
		for j in np.arange(len(adjacents) + 1):
			if j >= len(adjacents) or network.degree(i) < network.degree(adjacents[j][0]):
				adjacents.insert(j, (i, network.neighbors(i)))
				break
	
	return adjacents

'''
Generates one level in the hierarchy required to calculate the labels.
Such a level corresponds to an L_i in the paper.
'''
def gen_level(network):
	new_level = list()
	gprime = sorted_adjs(network)
	lprime = list()
	adjacents = list()
	
	for node, adj in gprime:
		if not node in lprime:
			new_level.append(node)
			adjacents.append((node, network.neighbors(node)))
			for m in network[node]:
				if not m in lprime:
					lprime.append(m)
	
	return [new_level, adjacents]

'''
Generates the remaining network after removing the passed level.
Corresponds to G_i in the paper.
'''
def gen_subnet(network, level, adjacents):
	subnet = network.copy()
	for adj in adjacents:
		add_augmenting_edges(subnet, adj[0], adj[1])
		
	for node in level:
		subnet.remove_node(node)
	return subnet
	
def sigmacutoff(subnets, sigma):
	if len(subnets) < 2:
		return False
	return nx.number_of_nodes(subnets[-1]) / nx.number_of_nodes(subnets[-2]) > sigma

'''
Generates the hierarchy required to calculate the labels for each node.
'''
def gen_hierarchy(network, sigma):
	adj = [network.neighbors(0)]
	levels = list()
	subnets = [network]
	while len(adj) > 0 and not sigmacutoff(subnets, sigma):
		print(len(subnets), nx.number_of_nodes(subnets[-1]))
		[level, adj] = gen_level(subnets[-1])
		levels.append(level)
		neti = gen_subnet(subnets[-1], level, adj)
		subnets.append(neti)
		
	return levels, [net for net in subnets if net.number_of_nodes() > 0]

'''
Initialises the label of the given node
'''
def init_label(node, subnet):
	label = {node : 0}
	for u in subnet.neighbors(node):
		label[u] = subnet[node][u]['weight']
	return label

'''
Initialises the labels of all nodes in the network
'''
def init_labels(subnets, levels):
	labels = dict()
	for i in np.arange(len(levels)):
		for node in levels[i]:
			labels[node] = init_label(node, subnets[i])
	
	return labels

'''
Generates the labels of all the nodes via top-down vertex labeling and
returns these (requires initialised labels)
'''
def gen_labels(labels, levels, network):
	print("Generating labels. Level:")
	for i in np.arange(len(levels) - 1)[::-1]:
		print(i)
		for v in levels[i]:
			for l in levels[i + 1:]:
				for u in l:
					if u in labels[v]:
						for w, val in labels[u].iteritems():
							if not w in labels[v]:
								labels[v][w] = labels[v][u] + labels[u][w]
							else:
								labels[v][w] = min(labels[v][w], labels[v][u] + labels[u][w])
	return labels

'''
Calls all functions necessary for the preprocessing of the network
Returns the labels, which can then be saved to disk
'''
def preprocess(network, sigma):
	[levels, subnets] = gen_hierarchy(network, sigma)
	print("Hierarchy generated!")
	labels = init_labels(subnets, levels)
	print("Labels initialised!")
	return gen_labels(labels, levels, network), subnets[-1]
	
def save_labels(labels, toplevel, filen):
	print("Saving: ", labels, nx.nodes(toplevel))
	f = open(filen, "w")
	pi.dump((labels, toplevel), f)
	f.close()
	return
	
	
def init_rand(seed):
	np.random.seed(seed)


def parse_args():
	parser = ap.ArgumentParser(description='Run islabel preprocessing.')
	parser.add_argument('fin',
											help='path to the file containing the network')
	parser.add_argument('fout',
											help='path to the file to write the labels')
	parser.add_argument('--directed',
	                    help=('provide this flag to interpret the ' +
	                    'edgelist as a directed network'),
	                    action='store_true')
	parser.add_argument('--weight',
	                    help=('provide this flag to interpret the ' +
	                    'edgelist as a weighted network'),
	                    action='store_true')
	parser.add_argument('--seed',
											help='Seed for the random number generator')
	parser.add_argument('-sigma',
											help='Parameter controlling when the hierarchy ' +
											'generation terminates',
											type=float)
	parser.add_argument('--adjlist',
											help='Switch from read_edgelist to read_adjlist',
											action='store_true')
	return parser.parse_args()


def main():
	args = parse_args()
	data = (('weight', float),)
	if args.sigma == None:
		args.sigma = 1.0
	
	if args.seed:
		init_rand(seed)
	else:
		init_rand(0)
		
	if args.directed:
		if args.adjlist:
			network = nx.read_adjlist(args.fin,
																create_using=nx.DiGraph())
		else:
			network = nx.read_edgelist(args.fin,
																 create_using=nx.DiGraph(), data=data)
		print("Successfully loaded directed network.")
	else:
		if args.adjlist:
			network = nx.read_adjlist(args.fin)
		else:
			network = nx.read_edgelist(args.fin, data=data)
		print("Successfully loaded undirected network.")
	
	if not args.weight:
		for x, y in network.edges():
			network[x][y]['weight'] = 1
	
	
	network = nx.convert_node_labels_to_integers(network)
	
	start = t.time()
	[labels, toplevel] = preprocess(network, args.sigma)
	save_labels(labels, toplevel, args.fout)
	end = t.time()
	print("Time elapsed:", end - start)


if __name__ == "__main__":
	main()
