
from __future__ import print_function
from __future__ import division

import numpy as np
import networkx as nx
import argparse as ap
import time as ti

import islabel as isl
import islabel_query as isq

def testall(G, Gk, labels, fout, verbose=False):
	dists = nx.shortest_path_length(G, weight="weight")
	toterr = 0.0
	n = 0.0
	start = ti.time()
	for s in G.nodes():
		for t in G.nodes():
			if s < t:
				n += 1.0
				estimated = isq.query(G, Gk, labels, s, t)
				if verbose:
					fout.write(str(s) + " -> " + str(t) + ": " + str(dists[s][t]) + " =? " + str(estimated) + " (" + str(toterr) + ")\n")
				toterr += abs(estimated - dists[s][t])
	end = ti.time()
	return toterr / n, (end - start) / n

def test_sample(G, Gk, labels, fout, nnodes, verbose=False):
	
	return

def test(G, Gk, labels, fout, nnodes=None, verbose=False):
	if nnodes == None:
		return testall(G, Gk, labels, fout, verbose)
	else:
		return test_sample(G, Gk, labels, fout, nnodes, verbose)
		

def parse_args():
	parser = ap.ArgumentParser(description='Run islabel query test.')
	parser.add_argument('fgraph',
											help='File containing the network')
	parser.add_argument('flabels',
											help='File containing the labels')
	parser.add_argument('fout',
											help='File to write output')
	parser.add_argument('--verb',
											help='Turn on verbosity',
											action='store_true')
	return parser.parse_args()
	
def main():
	args = parse_args()
	f = open(args.fout, "w")
	[G, Gk, labels] = isq.setup(args.fgraph, args.flabels, weight=True)
	[avge, avgt] = test(G, Gk, labels, f, verbose=args.verb)
	f.write("Avg err: " + str(avge) + " Avg time: " + str(avgt) + "\n")
	f.close()

if __name__ == "__main__":
	main()
