##########################################################################
#                                                                        #
#  Filename: random_graph.py                                              #
#  Author: htdv92                                                        #
#  Date created:                                                         #
#  Date Last-Modified:                                                   #
#  Python Version: 3.7                                                   #
#  Dependicies:                                                          #
#  Description: A                                                        #
#                                                                        #
##########################################################################

import random


def load_graph(num_nodes, prob):
    """Returns a dictionary to a random graph with the specified number of nodes
    and edge probability.  The nodes of the graph are numbered 0 to
    num_nodes - 1.  For every pair of nodes, i and j, the pair is considered
    twice: once to add an edge (i,j) with probability prob, and then to add an
    edge (j,i) with probability prob.
    """
    # initialize empty graph
    graph = {}
    for vertex in range(num_nodes):
        graph[vertex] = set()

    # consider each vertex
    for vertex in range(num_nodes):
        out_neighbours = []

        for neighbour in range(num_nodes):
            if vertex != neighbour:
                random_number = random.random()
                if random_number < prob:
                    out_neighbours += [neighbour]

        # add vertex with list of out_neighbours
        graph[vertex] = set(out_neighbours)
        for new_neighbour in out_neighbours:
            graph[new_neighbour].add(vertex)

    return graph
