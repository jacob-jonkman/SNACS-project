"""
Implementation of the IS-Label algorithm

Assumptions:
- Independent sets are chosen randomly with a given maximum set size.
  Not necessarily the best method. Should be investigated.

Todo:
- Investigate optimal IS generation
- gen_label
- Change lists to np arrays (would require calculating upper bounds and
  dealing with unset values, but perhaps worth it)

"""
import numpy as np
import networkx as nx
import argparse as ap


'''
Adds an augmenting edge (n, m) if necessary.
'''
def augment(network, n, m, o):
	new_dist = network[n][m]['weight']
	new_dist += network[m][o]['weight']
	
	if network.has_edge(n, o) and network[n][o]['weight'] > new_dist:
		network[n][o]['weight'] = new_dist
	else:
		network.add_edge(n, o, attr_dict={'weight' : new_dist})

'''
Adds all necessary aumenting edges to the graph after removing the
given node.
'''
def add_augmenting_edges(network, adj):
	for n, m in adj[1], adj[1]:
		if n < m:
			augment(network, n, adj[0], m)

'''
Generates a list of adjacent nodes for each node in the network.
Then it generates a list of these lists, which is sorted by degree
of the nodes (or: the number of adjacent nodes in each list).
'''
def sorted_adjs(network):
	adjacents = list()
	for i in np.arange(network.number_of_nodes()):
		for j in np.arange(len(adjacents) + 1):
			if j > len(adjacents) or len(network[i]) < len(adjacents[j][1]):
				adjacents.insert(j, (i, network[i]))
	
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
			adjacents.append(network[node])
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
		add_augmenting_edges(subnet, adj)
		
	for node in level:
		subnet.remove_node(node)
	return subnet

'''
Generates the hierarchy required to calculate the labels for each node.
'''
def gen_hierarchy(network):
	adj = [network[0]]
	levels = list()
	subnets = [network]
	while len(adj) > 0:
		[level, adj] = gen_level(subnets[-1])
		levels.append(level)
		neti = gen_subnet(subnets[-1], level, adj)
		subnets.append(neti)
		
	return levels, subnets
	
'''
Generates the label of the given node using the given hierarchy
'''
def gen_label(network, hierarchy, node):
	return
	
	
def init_rand(seed):
	np.random.seed(seed)


def parse_args():
	parser = ap.ArgumentParser(description='Run islabel preprocessing.')
	parser.add_argument('fin',
											help='path to the file containing the network')
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
	return parser.parse_args()


def main():
	args = parse_args()
	if args.weight:
		data = (('weight', float),)
	else:
		data = None
		
	if args.seed:
		init_rand(seed)
	else:
		init_rand(0)
		
	if(args.directed):
		network = nx.read_edgelist(args.fin,
															 create_using=nx.DiGraph(), data=data)
		print("Successfully loaded directed network.")
	else:
		network = nx.read_edgelist(args.fin, data=data)
		print("Successfully loaded undirected network.")


if __name__ == "__main__":
	main()
