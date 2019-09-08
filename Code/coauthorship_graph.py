##########################################################################
#                                                                        #
#  Filename: coauthorship_graph.py                                       #
#  Author: htdv92                                                        #
#  Date created:                                                         #
#  Date Last-Modified:                                                   #
#  Python Version: 3.7                                                   #
#  Dependicies:                                                          #
#  Description: A                                                        #
#                                                                        #
##########################################################################

def load_graph():
    graph_file = open("coauthorship.txt", 'r')

    graph = {}
    nodes = 0

    while True:
        if graph_file.readline()[0] == '*':
            break

    node_line = graph_file.readline()
    while node_line[0] != '*':
        line_split = node_line.strip().split(' ')
        graph[int(line_split[0])] = set()
        node_line = graph_file.readline()

    edge_line = graph_file.readline()
    while True:
        line_split = edge_line.strip().split(' ')
        vertex_u = int(line_split[0])
        vertex_v = int(line_split[1])

        if vertex_u != vertex_v:
            graph[vertex_u].add(vertex_v)
            graph[vertex_v].add(vertex_u)

        edge_line = graph_file.readline()
        if edge_line == '':
            break

    return graph
