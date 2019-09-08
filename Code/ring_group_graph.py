##########################################################################
#                                                                        #
#  Filename: ring_group_graph.py                                          #
#  Author: htdv92                                                        #
#  Date created:                                                         #
#  Date Last-Modified:                                                   #
#  Python Version: 3.7                                                   #
#  Dependicies:                                                          #
#  Description: A                                                        #
#                                                                        #
##########################################################################

import random

"""
A Ring Group Graph is defined by four parameters m, k, p and q and is constructed as follows:
• Create mk vertices. The vertices are partitioned into m groups each of size k. The groups are labelled from 0 to m−1.
• For each pair of distinct vertices u and v, let lu and lv be the labels of the groups that u and v belong to respectively and
– iflu =lv (thatis,uandvareinthesamegroup)orif|lu−lv|=1modm(thatis,uandvare in adjacent groups), add an edge between u and v with probability p,
– otherwise add an edge between u and v with probability q.

m = group number
k = group size
p = 
q = 
"""


def load_graph(m, k, p, q):
    num_nodes = m * k
    ring_graph = {}  # initialize empty graph

    for vertex in range(num_nodes): ring_graph[vertex] = set()  # Add vertices

    for vertex_u in range(num_nodes):
        vertex_v = vertex_u + 1
        for vertex_v in range(vertex_v, num_nodes):
            group_u = vertex_u // k
            group_v = vertex_v // k
            random_number = random.random()
            test = abs(group_u - group_v) % m
            if (group_u == group_v) or (test == 1) or test == (m-1):
                if random_number < p:
                    ring_graph[vertex_u].add(vertex_v)
                    ring_graph[vertex_v].add(vertex_u)
            else:
                if random_number < q:
                    ring_graph[vertex_u].add(vertex_v)
                    ring_graph[vertex_v].add(vertex_u)

    return ring_graph
