
from __future__ import print_function

import numpy as np
import networkx as nx
import argparse as ap
import pickle as pi
import queue as q

'''
Initialise and return a min-priority queue containing labels of the
given node and all top level vertices
'''
def initq(graph, toplevel, labels, node):
	#TODO: use distance as key
	queue = q.PriorityQueue()
	for n in nx.nodes(toplevel):
		queue.push((n, toplevel[node][n]['weight']))
	for key, val in labels[node].iteritems():
		queue.push((val, key))
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
		
	return minres

'''
Estimate distance via Label-based bi-Dijkstra Search
'''
def bi_dijkstra(graph, toplevel, labels, source, target):
	forwardq = initq(graph, toplevel, labels, source)
	reverseq = initq(graph, toplevel, labels, target)
	mindist = query_formula(labels, source, target)
	
	qsum = 0
	fmin = None
	rmin = None
	S = dict()
	while qsum < mindist:
		if qsum == 0:
			[distf, nodef] = forwardq.get()
			[distr, noder] = reverseq.get()
			
		if distf > distr:
			x = target
			xp = source
			v = noder
			dist = distr
			#forwardq.put((distf, nodef))
		else:
			x = source
			xp = target
			v = nodef
			dist = distf
			#reverseq.put((distr, noder))
		
		if v not in S:
			S[v] = dist
		
		for u in toplevel.neighbors(v):
			if 
		
		if not forwardq.empty() and not reverseq.empty():
			fmin = forwardq.get()
			rmin = reverseq.get()
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
	for key, _ in labels[source]:
		if key in toplevel.nodes():
			source_intersect = True
			break
	if !source_intersect:
		return False
	
	for key, _ in labels[target]:
		if key in toplevel.nodes():
			return True
	return False
	
'''
Execute distance estimation query
'''
def query(graph, toplevel, labels, source, target):
	if use_formula(toplevel, labels, source, target):
		return query_formula(labels, source, target)
	else
		return bi_dijkstra(graph, toplevel, labels, source, target)
	
'''
Loads the labels and the top level of the graph from the given file name
Returns this as: [labels, toplevel]
'''
def loadlabels(filen):
	f = open(filen, "r")
	[labels, toplevel] = pi.load(f)
	print("Loaded:", labels, toplevel)
	return labels, toplevel
	

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
	return parser.parse_args()
	

def main():
	args = parse_args()
	[toplevel, labels] = loadlabels(args.flabels)
	#graph = nx.
	res = query(toplevel, labels, int(args.source), int(args.target))
	print(res)
	return res

if __name__ == "__main__":
	main()
