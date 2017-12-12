
from __future__ import print_function

import numpy as np
import networkx as nx
import argparse as ap
import pickle as pi

'''
Update an element into the given queue. It is placed behind the last
element with a priority <= the given priority. If the value already
exists in the queue, it is moved to the new position. The queue is
traversed from back to front.
TODO:
- Update to binary search
'''
def update(queue, val, prio):
	done = False
	for i in np.arange(len(queue))[::-1]:
		if queue[i][0] == val:
			queue.pop(i)
			if done:
				return
			else:
				done = True
		if queue[i][1] <= prio:
			queue.insert(i + 1, (val, prio))
			if done:
				return
			else:
				done = True
	queue.insert(0, (val, prio))

'''
Push an element into the given queue. It is placed behind the last
element with a priority <= the given priority. 
'''	
def push(queue, val, prio):
	for i in np.arange(len(queue))[::-1]:
		if queue[i][1] <= prio:
			queue.insert(i + 1, (val, prio))
			return
	queue.insert(0, (val, prio))

'''
Initialise and return a min-priority queue containing labels of the
given node and all top level vertices
'''
def initq(graph, toplevel, labels, node):
	queue = list()
	for n in nx.nodes(toplevel):
		push(queue, n, toplevel[node][n]['weight'])
	for key, val in labels[node].iteritems():
		push(queue, val, key)
	for n in nx.nodes(graph):
		if n not in queue:
			push(queue, val, float('inf'))
	return queue
	
'''
Distance calculation via formula (used when labels intersect)
'''
def query_formula(labels, source, target):
	intersection = [v for v in labels[source] if v in labels[target]]
	
	mindist = float('inf')
	for node in intersection:
		dist = labels[source][node] + labels[target][node]
		mindist = min(mindist, dist)
		
	return mindist

'''
Estimate distance via Label-based bi-Dijkstra Search
TODO:
- split up into functions
'''
def bi_dijkstra(graph, toplevel, labels, source, target):
	forwardq = initq(graph, toplevel, labels, source)
	reverseq = initq(graph, toplevel, labels, target)
	mindist = query_formula(labels, source, target)
	
	distances = dict()
	
	qsum = 0
	fmin = None
	rmin = None
	S = ()
	while qsum < mindist:
		[distf, nodef] = forwardq[0]
		[distr, noder] = reverseq[0]
			
		if distf > distr:
			x = target
			xp = source
			v = noder
			dist = distr
			queue = reverseq
		else:
			x = source
			xp = target
			v = nodef
			dist = distf
			queue = forwardq
		queue.pop(0)
		
		distances[(x, v)] = dist
		
		if (x, v) not in S:
			S.append((x, v))
		
		for u in toplevel.neighbors(v):
			newdist = distance[(x, v)] + toplevel[v][u]
			if distance[(x, u)] > newdist:
				distance[(x, u)] = newdist
				update(queue, u, distance[(x, u)])
				if (xp, u) in S:
					mindist = min(mindist, distances[(x, u)] + graph[xp][u])
				
		if not forwardq.empty() and not reverseq.empty():
			qsum = fmin[0] + rmin[0]
		else:
			qsum = float('inf')
	
	return mindist
	
'''
Checks whether we should use the formula (or the bi-dijkstra search)
by calculating whether any of the nodes in the labels is in the top
level
'''
def use_formula(toplevel, labels, source, target):
	source_intersect = False
	for key in labels[source]:
		if key in toplevel.nodes():
			source_intersect = True
			break
	if not source_intersect:
		return False
	
	for key in labels[target]:
		if key in toplevel.nodes():
			return True
	return False
	
'''
Execute distance estimation query
'''
def query(graph, toplevel, labels, source, target):
	if use_formula(toplevel, labels, source, target):
		return query_formula(labels, source, target)
	else:
		return bi_dijkstra(graph, toplevel, labels, source, target)
	
'''
Loads the labels and the top level of the graph from the given file name
Returns this as: [labels, toplevel]
'''
def loadlabels(filen):
	f = open(filen, "r")
	[labels, toplevel] = pi.load(f)
	print("Loaded:", labels, toplevel)
	return toplevel, labels
	
def readgraph(fin, directed, adjlist, weight):
	data = (('weight', float),)
	if directed:
		if adjlist:
			network = nx.read_adjlist(fin,
																create_using=nx.DiGraph())
		else:
			network = nx.read_edgelist(fin,
																 create_using=nx.DiGraph(), data=data)
		print("Successfully loaded directed network.")
	else:
		if adjlist:
			network = nx.read_adjlist(fin)
		else:
			network = nx.read_edgelist(fin, data=data)
		print("Successfully loaded undirected network.")
	
	network = nx.convert_node_labels_to_integers(network, ordering="sorted")
	if not weight:
		for x, y in network.edges():
			network[x][y]['weight'] = 1
	return network
	
def setup(fgraph, flabels, directed=False, adjlist=False, weight=False):
	[toplevel, labels] = loadlabels(flabels)
	graph = readgraph(fgraph, directed, adjlist, weight)
	return [graph, toplevel, labels]
	
def parse_args():
	parser = ap.ArgumentParser(description='Run islabel query.')
	parser.add_argument('fgraph',
											help='path to the file containing the network')
	parser.add_argument('flabels',
											help='File containing the labels')
	parser.add_argument('source',
											help='Source node of the query')
	parser.add_argument('target',
											help='Target node of the query')
	parser.add_argument('--directed',
	                    help=('provide this flag to interpret the ' +
	                    'network as a directed network'),
	                    action='store_true')
	parser.add_argument('--weight',
	                    help=('provide this flag to interpret the ' +
	                    'network as a weighted network'),
	                    action='store_true')
	parser.add_argument('--adjlist',
											help='Switch from read_edgelist to read_adjlist',
											action='store_true')
	return parser.parse_args()

def main():
	args = parse_args()
	[graph, toplevel, labels] = setup(args.fgraph,
																		args.flabels,
																		args.directed,
																		args.adjlist,
																		args.weight)
	res = query(graph, toplevel, labels, int(args.source), int(args.target))
	print(res)

if __name__ == "__main__":
	main()
