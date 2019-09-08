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
import pandas as pd
import matplotlib.pyplot as plt
import ring_group_graph as ring_group_creation


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


def plot_normal_distribution(data):
    mean = np.mean(list(data.values()))
    df = pd.DataFrame(data)
    variance = df.std()

    sigma = math.sqrt(variance)
    x = np.linspace(mean - 3 * sigma, mean + 3 * sigma, 100)
    plt.plot(x, plt.normpdf(x, mean, sigma))


def run_diameter(number_of_graphs, m, k, q, lower_p, upper_p):
    progress_index = 1
    print('Diameter Graphs')

    for p in np.linspace(lower_p, upper_p, number_of_graphs):
        print(str(progress_index) + ' / ' + str(number_of_graphs), )
        progress_index += 1

        ring_graph, num_nodes = ring_group_creation.load_graph(m, k, p, q)
        plt.figure(2)
        add_to_plot(p, compute_diameter(ring_graph), 'b')

    plt.figure(2)
    plt.savefig(
        './Question1/Diameter/m' + str(m) + ' k' + str(k) + ' Num' + str(number_of_graphs) + '.png')


def run_distribution(number_of_graphs, num_samples, m, k, fixed_q):
    progress_index = 0
    print('Distribution Graphs')
    colour = (1, 0, 0, 1)

    normalized_distribution_samples = [None] * num_samples

    if fixed_q is not None:
        p = 0.5 - fixed_q
        print('P = ' + str(p) + ' Q = ' + str(fixed_q))
        for i in range(num_samples):
            print('Sample: ' + str(i + 1))
            ring_graph, num_nodes = ring_group_creation.load_graph(m, k, p, fixed_q)

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

        # Averaged
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

        x_data = []
        y_data = []
        for degree in averaged_normalized_distribution:
            x_data += [degree]
            y_data += [averaged_normalized_distribution[degree]]

        add_to_plot(x_data, y_data, colour)

        amount_of_green = (1 / number_of_graphs) * progress_index
        colour = (1, amount_of_green, 0)

        progress_index += 1

    else:
        for q in np.linspace(0, 0.25, number_of_graphs):
            print()
            p = 0.5 - q
            print('P = ' + str(p) + ' Q = ' + str(q))

            for i in range(num_samples):
                print(str(progress_index + 1) + ' / ' + str(number_of_graphs) + ' Sample: ' + str(
                    i + 1))
                ring_graph, num_nodes = ring_group_creation.load_graph(m, k, p, q)

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
            # Averaged
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

            x_data = []
            y_data = []
            for degree in averaged_normalized_distribution:
                x_data += [degree]
                y_data += [averaged_normalized_distribution[degree]]

            add_to_plot(x_data, y_data, colour)

            amount_of_green = (1 / number_of_graphs) * progress_index
            colour = (1, amount_of_green, 0)
            # colour = (0.5, 0.5, 0, 1)

            progress_index += 1

    plt.figure(1)
    plot_normal_distribution(averaged_normalized_distribution)
    plt.savefig('./Question1/Distribution/m' + str(m) + ' k' + str(k) + ' Num' + str(
        number_of_graphs) + '.png')


m_distribution = 25
k_distribution = 25
# fixed_q = 0.1
fixed_q = None
num_graphs_distribution = 1
num_samples_distribution = 5

create_scatter_plot('Degree', 'Normalized Degree Distribution', '')
run_distribution(num_graphs_distribution, num_samples_distribution, m_distribution, k_distribution,
                 fixed_q)

# adapted from https://stackoverflow.com/questions/10138085/python-pylab-plot-normal-distribution#10138308

# m_diameter = 20
# k_diameter = 5
# q = 0.2
# lower_p = 0.3
# upper_p = 1.0
# num_graphs_diameter = 50
#
# create_scatter_plot('Value of P', 'Diameter', '')  # Diameter is figure 2
# run_diameter(num_graphs_diameter, m ,k, q, lower_p, upper_p)
