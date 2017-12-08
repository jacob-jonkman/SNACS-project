
from __future__ import print_function

import numpy as np
import networkx as nx
import argparse as ap
import pickle as pi
import queue as q

def initq(graph, toplevel, labels, node):
	#TODO: use distance as key
	queue = q.Queue()
	for n in nx.nodes(toplevel):
		queue.push((n, toplevel[node][n]['weight']))
	for key, val in labels[node].iteritems():
		queue.push((key, val))
	return queue
	
def query_formula(toplevel, labels, source, target):
	intersection = [v for v in labels[source] if v in labels[target]]
	if len(intersection) == 0:
		return None
		
	

'''
Estimate distance via Label-based bi-Dijkstra Search
'''
def query(toplevel, labels, source, target):
	result = query_formula(toplevel, labels, source, target)
	if result != None:
		return result
	
	forwardq = initq(graph, toplevel, labels, source)
	reverseq = initq(graph, toplevel, labels, target)
	
	#dostuff
	
	return result
	
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
	#parser.add_argument('fgraph',
	#										help='path to the file containing the network')
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
	res = query(toplevel, labels, int(args.source), int(args.target))
	print(res)
	return res

if __name__ == "__main__":
	main()
