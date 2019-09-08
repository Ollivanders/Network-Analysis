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

import collections
import queue
import math
import numpy as np
import matplotlib.pyplot as plt
import ring_group_graph as ring_group_creation
import matplotlib.mlab as mlab


def compute_degrees(graph):
    degrees = {}
    for vertex in graph: degrees[vertex] = 0

    for vertex in graph:
        for neighbour in graph[vertex]:
            degrees[neighbour] += 1
    return degrees


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


def max_dist(graph, source):
    """finds the distance (the length of the shortest path) from the source to
    every other vertex in the same component using breadth-first search, and
    returns the value of the largest distance found"""
    q = queue.Queue()
    found = {}
    distance = {}
    for vertex in graph:  # set up arrays
        found[vertex] = 0  # to record whether a vertex has been discovered
        distance[vertex] = -1  # and its distance from the source
    max_distance = 0
    found[source] = 1  # initialize arrays with values for the source
    distance[source] = 0
    q.put(source)  # put the source in the queue
    while q.empty() == False:
        current = q.get()  # process the vertex at the front of the queue
        for neighbour in graph[current]:  # look at its neighbours
            if found[neighbour] == 0:  # if undiscovered, update arrays and add to the queue
                found[neighbour] = 1
                distance[neighbour] = distance[current] + 1
                max_distance = distance[neighbour]
                q.put(neighbour)
    return max_distance


def compute_diameter(graph):
    """returns the diameter of a graph, by finding, for each vertex, the maximum
    length of a shortest path starting at that vertex, and returning the overall
    maximum"""
    distances = []
    for vertex in graph:  # look at each vertex
        distances += [max_dist(graph, vertex)]  # find the distance to the farthest other vertex
    return max(distances)  # return the maximum value found


def add_to_plot(x_data, y_data, colour):
    # plot degree distribution
    plt.plot(x_data, y_data, marker='.', color=colour, linestyle='None')  # ,linestyle='-'


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

    sigma = math.sqrt(variance)
    x = np.linspace(mean - 3 * sigma, mean + 3 * sigma, 100)
    plt.plot(x, mlab.normpdf(x, mean, sigma), color=(0, 0, 0, 1))


def run_diameter(number_of_samples, number_of_nodes, m, k, q, lower_p, upper_p):
    progress_index = 1
    print('Diameter Graphs')
    samples = [None] * number_of_samples

    diameters = {}
    for p in np.linspace(lower_p, upper_p, number_of_nodes):
        print(str(progress_index) + ' / ' + str(number_of_nodes), )
        progress_index += 1
        for i in range(number_of_samples):
            print(str(progress_index) + ' / ' + str(number_of_nodes) + ' ' + str(i) + '/' + str(number_of_samples), )

            ring_graph = ring_group_creation.load_graph(m, k, p, q)
            samples[i] = compute_diameter(ring_graph)

        diameters[p] = 0
        for sample in samples:
            diameters[p] += sample
        diameters[p] = diameters[p] / number_of_samples

    x_data = []
    y_data = []
    for p_value in diameters:
        x_data += [p_value]
        y_data += [diameters[p_value]]

    add_to_plot(x_data, y_data, (0, 0, 0, 0.5))

    plt.savefig(
        './Question1/Diameter/m' + str(m) + 'k' + str(k) + 'num' + str(number_of_nodes) + '.png')


def num_edges(graph):
    total_degrees = 0

    for vertex in graph:
        total_degrees += len(graph[vertex])

    return total_degrees / 2


def run_distribution(number_of_graphs, num_samples, m, k):
    create_scatter_plot('Degree', 'Normalized Degree Distribution', '')
    progress_index = 0
    print('Distribution Graphs')
    colour = (1, 0, 0, 1)

    normalized_distribution_samples = [None] * num_samples
    distribution_samples = [None] * num_samples

    for q in np.linspace(0, 0.25, number_of_graphs):
        # for i in range(1):
        print()
        q = 0.2
        p = 0.5 - q
        print('P = ' + str(p) + ' Q = ' + str(q))

        edges_sample = list()
        for i in range(num_samples):
            print(str(progress_index + 1) + ' / ' + str(number_of_graphs) + ' Sample: ' + str(
                i + 1) + '/' + str(num_samples))
            ring_graph = ring_group_creation.load_graph(m, k, p, q)
            num_nodes = m * k
            distribution = collections.OrderedDict(
                sorted(degree_distribution(ring_graph).items()))
            plt.figure(1)

            distribution_sum = 0
            for degree in distribution:
                distribution_sum += distribution[degree]

            normalized_distribution = {}
            for degree in distribution:
                normalized_distribution[degree] = distribution[degree] / distribution_sum

            normalized_distribution_samples[i] = normalized_distribution
            distribution_samples[i] = distribution
            edges_sample.append(num_edges(ring_graph))

        # Averaged normalized
        averaged_normalized_distribution = {}
        # create arrays for plotting
        for sample in normalized_distribution_samples:
            for degree in sample:
                if degree not in averaged_normalized_distribution:
                    averaged_normalized_distribution[degree] = sample[degree]
                else:
                    averaged_normalized_distribution[degree] += sample[degree]

        for degree in averaged_normalized_distribution:
            averaged_normalized_distribution[degree] = averaged_normalized_distribution[
                                                           degree] / num_samples

        # Averaged
        averaged_distribution = {}
        # create arrays for plotting
        for sample in distribution_samples:
            for degree in sample:
                if degree not in averaged_distribution:
                    averaged_distribution[degree] = sample[degree]
                else:
                    averaged_distribution[degree] += sample[degree]

        for degree in averaged_distribution:
            averaged_distribution[degree] = averaged_distribution[degree] / num_samples

        x_data = []
        y_data = []
        for degree in averaged_normalized_distribution:
            x_data += [degree]
            y_data += [averaged_normalized_distribution[degree]]

        add_to_plot(x_data, y_data, colour)

        amount_of_green = (1 / number_of_graphs) * progress_index
        colour = (1, amount_of_green, 0)

        progress_index += 1
        # additional data
        data_array = create_array(averaged_distribution)
        plot_normal_distribution(averaged_distribution)

        mean = np.mean(data_array)
        range_data = np.ptp(data_array)
        variance = np.var(data_array)
        print('edges ' + str(np.mean(edges_sample)))
        print('mean ' + str(mean))
        print('range ' + str(range_data))
        print('variance ' + str(variance))

    plt.savefig('./Question1/Distribution/m' + str(m) + 'k' + str(k) + '.png')


# m_distribution = 100
# k_distribution = 100
# num_graphs_distribution = 1
# num_samples_distribution = 10
#
# run_distribution(num_graphs_distribution, num_samples_distribution, m_distribution, k_distribution)

# adapted from https://stackoverflow.com/questions/10138085/python-pylab-plot-normal-distribution#10138308

number_of_samples = 1
m_diameter = 25
k_diameter = 25
q = 0.5
lower_p = q
upper_p = 1.0
num_graphs_diameter = 25

create_scatter_plot('Value of P', 'Diameter', '')  # Diameter is figure 2
run_diameter(number_of_samples, num_graphs_diameter, m_diameter, k_diameter, q, lower_p, upper_p)
