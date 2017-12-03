import networkx as nx
import argparse
import os 
import random
import re

OUTPUT_FILE = os.path.join("..", "gen", "graph.txt") 
EDGE_CREATION_BASE = 0.44

PARSER = argparse.ArgumentParser(prog="GraphGenerator", description="Generate file with graph")
PARSER.add_argument("vertex_count", type=int, help="Provide graph size - vertexes number")
PARSER.add_argument("-r", "--random", action="store_true", help="Creates graph edges with random probabiltiy form 0.44 to 1.0")

def main():
    args = PARSER.parse_args()
    
    if os.path.isfile(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    if args.random:
        graph = nx.fast_gnp_random_graph(args.vertex_count, random.uniform(EDGE_CREATION_BASE, 1.0), random.seed())
    else:
        graph = nx.fast_gnp_random_graph(args.vertex_count, EDGE_CREATION_BASE, random.seed())
        
    graph_matrix = nx.adjacency_matrix(graph)

    with open(OUTPUT_FILE, mode='w') as graph_file:
        graph_file.write(str(args.vertex_count))
        graph_file.write('\n')

        graph_string = re.sub("\[|\]", "", str(graph_matrix.todense()))
        graph_string = re.sub("\n ", "\n", graph_string)
        graph_file.write(graph_string)

main()