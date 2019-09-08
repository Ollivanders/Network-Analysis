##########################################################################
#                                                                        #
#  Filename: brilliance.py                                               #
#  Author: htdv92                                                        #
#  Date created:                                                         #
#  Date Last-Modified:                                                   #
#  Python Version: 3.7                                                   #
#  Dependicies:                                                          #
#  Description: A                                                        #
#                                                                        #
##########################################################################

import operator
import random
import sys

import matplotlib.pyplot as plt
import coauthorship_graph as coauthorship_creation
import numpy as np
import pa_graph as pa_creation
import ring_group_graph as ring_group_creation
import networkx as nx
from matplotlib import mlab, collections


def estimate_vertex_brilliance(graph):
    # Returns a diciontary of the estimated vertex brilliance for each vertex
    # vertex with lowest degree, add and remove neighbours until finished
    graph_x = nx.Graph(graph)
    graph_brilliance = {}
    for vertex_k in graph:
        print(str(vertex_k))
        # Create Subgraph, ignoring the centre vertex v and all other nodes not in the graph
        # subgraph = construct_star_subgraph(graph, vertex).copy()

        subgraph = nx.subgraph(graph_x, nx.all_neighbors(graph_x, vertex_k))

        # graph_brilliance[vertex] = find_independent_set_size(subgraph)
        if len(subgraph) > 0:
            # subgraph_network = nx.Graph(subgraph)
            graph_brilliance[vertex_k] = len(nx.maximal_independent_set(subgraph))
            # graph_brilliance[vertex_k] = find_maximal_independent_set_size(graph, vertex_k)

    return vertex_brilliance_to_rate(graph_brilliance)


def vertex_brilliance_to_rate(brilliance_by_vertex):
    brilliance_frequency = {}

    for vertex in brilliance_by_vertex:
        brilliance = brilliance_by_vertex[vertex]

        if brilliance in brilliance_frequency:
            brilliance_frequency[brilliance] += 1
        else:
            brilliance_frequency[brilliance] = 1

    return brilliance_frequency


# Don't build subgraph
def find_maximal_independent_set_size(graph, vertex_k):
    neighbours = graph[vertex_k].copy()
    degrees = {}

    for neighbour in neighbours:
        neighbours_neighbours = graph[neighbour]
        degree_count = 0
        for neighbours_neighbour in neighbours_neighbours:
            if neighbours_neighbours in neighbours:
                degree_count += 1
        degrees[neighbour] = degree_count

    ordered_neighbours_degrees = sorted(degrees.items(), key=operator.itemgetter(1))

    independent_size = 0
    vertex_index = 0
    while len(neighbours) != 0:
        # added_vertex = random.sample(neighbours, 1)[0]
        added_vertex = ordered_neighbours_degrees[vertex_index][0]
        vertex_index += 1

        if added_vertex in neighbours:
            independent_size += 1
            neighbours.remove(added_vertex)
            added_vertex_neighbours = graph[added_vertex]

            for added_vertex_neighbour in added_vertex_neighbours:
                if added_vertex_neighbour in neighbours:
                    neighbours.remove(added_vertex_neighbour)

    return independent_size


def compute_degrees(graph):
    degrees = {}
    for vertex in graph: degrees[vertex] = 0

    for vertex in graph:
        for neighbour in graph[vertex]:
            degrees[neighbour] += 1
    return degrees


def num_edges(graph):
    total_degrees = 0

    for vertex in graph:
        total_degrees += len(graph[vertex])

    return total_degrees / 2


def create_scatter_plot(xlabel, ylabel, title):
    plt.figure()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title != '': plt.title(title)


def create_array(data):
    data_array = list()

    for value in data.keys():
        frequency = int(data[value])
        data_array += [value for _ in range(frequency)]

    return data_array


def plot_normal_distribution(data):
    data_array = create_array(data)

    mean = np.mean(data_array)
    range = np.ptp(data_array)
    variance = np.var(data_array)

    sigma = np.math.sqrt(variance)
    x = np.linspace(mean - 3 * sigma, mean + 3 * sigma, 100)
    plt.plot(x, mlab.normpdf(x, mean, sigma), color=(0, 0, 0, 1))


def degree_distribution(graph):
    # find in_degrees
    degrees = compute_degrees(graph)
    # initialize dictionary for degree distribution
    degree_distribution = {}
    # consider each vertex
    for vertex in degrees:
        # update degree_distribution
        if degrees[vertex] in degree_distribution:
            degree_distribution[degrees[vertex]] += 1
        else:
            degree_distribution[degrees[vertex]] = 1
    return degree_distribution


def plot_brilliance(name, number_of_samples):
    graph = {}

    additional_notes = ''
    normalized_brilliance_samples = [None] * number_of_samples
    brilliance_samples = [None] * number_of_samples

    for i in range(number_of_samples):
        if name == 'coauthorship':
            print("Plot Coauthorship")
            graph = coauthorship_creation.load_graph()  # 56.666666666666664, average degree for co-authorship
        elif name == 'pa_graph':
            print("Plot PA Graph")
            num_nodes = 1559
            out_degree = 35
            graph = pa_creation.load_graph(num_nodes, out_degree)  # total_nodes, out_degree
            additional_notes = ' numNodes' + str(num_nodes) + ' outDegree' + str(
                out_degree) + ' Samples' + str(number_of_samples)
        elif name == 'ring_group_graph':
            print("Plot Ring Group")
            m = 50
            k = 31
            p = 0.08
            q = 0.03
            # m156 k10 p0.08 q0.035
            graph = ring_group_creation.load_graph(m, k, p, q)  # m k p q
            additional_notes = ' m' + str(m) + ' k' + str(k) + ' Samples' + str(number_of_samples)

        print(name + ' Sample: ' + str(i + 1) + ' Number of Edges: ' + str(num_edges(graph)))
        # brilliance = collections.OrderedDict(sorted(estimate_vertex_brilliance(graph).items()))
        brilliance_frequency = estimate_vertex_brilliance(graph)

        normalized_brilliance = {}
        brilliance_sum = 0

        for brilliance in brilliance_frequency:
            brilliance_sum += brilliance_frequency[brilliance]

        for brilliance in brilliance_frequency:
            normalized_brilliance[brilliance] = brilliance_frequency[brilliance] / brilliance_sum

        normalized_brilliance_samples[i] = normalized_brilliance
        brilliance_samples[i] = brilliance_frequency

    # Averaged
    averaged_normalized_brilliance = {}
    # create arrays for plotting
    for sample in normalized_brilliance_samples:
        for brilliance in sample:
            if brilliance not in averaged_normalized_brilliance:
                averaged_normalized_brilliance[brilliance] = 1
            else:
                averaged_normalized_brilliance[brilliance] += 1

    for brilliance in averaged_normalized_brilliance:
        averaged_normalized_brilliance[brilliance] = averaged_normalized_brilliance[
                                                         brilliance] / number_of_samples

    # Averaged
    averaged_brilliance = {}
    # create arrays for plotting
    for sample in brilliance_samples:
        for brilliance in sample:
            if brilliance not in averaged_brilliance:
                averaged_brilliance[brilliance] = sample[brilliance]
            else:
                averaged_brilliance[brilliance] += sample[brilliance]

    for brilliance in averaged_brilliance:
        averaged_brilliance[brilliance] = averaged_brilliance[brilliance] / number_of_samples

    # create arrays for plotting
    x_data = []
    y_data = []
    for brilliance in averaged_brilliance:
        x_data += [brilliance]
        y_data += [averaged_brilliance[brilliance]]

    # create_scatter_plot("Brilliance", "Brilliance Distribution", '')
    # plt.plot(x_data, y_data, marker='.', linestyle='None')  # linestyle='None'
    # plt.savefig('Question2/' + name)
    #
    # plt.loglog(x_data, y_data, marker='.', linestyle='None')  # linestyle='None'
    # plt.savefig('Question2/' + name + 'logplot')

    # create arrays for plotting
    x_data = []
    y_data = []
    for brilliance in averaged_normalized_brilliance:
        x_data += [brilliance]
        y_data += [averaged_normalized_brilliance[brilliance]]

    data_array = create_array(averaged_brilliance)

    mean = np.mean(data_array)
    range_data = np.ptp(data_array)
    variance = np.var(data_array)
    print('edges ' + str(num_edges(graph)))
    print('mean ' + str(mean))
    print('range ' + str(range_data))
    print('variance ' + str(variance))

    create_scatter_plot("Brilliance", "Brilliance Distribution", '')
    if name == 'ring_group_graph':
        plot_normal_distribution(averaged_brilliance)
    plt.plot(x_data, y_data, marker='.', linestyle='None')  # linestyle='None'
    plt.savefig('Question2/' + name + 'normalized')

    create_scatter_plot("Brilliance", "Brilliance Distribution", '')
    plt.loglog(x_data, y_data, marker='.', linestyle='None')  # linestyle='None'
    plt.savefig('Question2/' + name + 'logplot' + 'normalized')


number_of_samples = 10

print()
# plot_brilliance('coauthorship', 1)

print()
# plot_brilliance('pa_graph', number_of_samples)

print()
plot_brilliance('ring_group_graph', number_of_samples)

# Expected degree of each in ring group graph
