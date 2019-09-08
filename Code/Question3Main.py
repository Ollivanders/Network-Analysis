##########################################################################
#                                                                        #
#  Filename: Question3Main.py                                            #
#  Author: htdv92                                                        #
#  Date created:                                                         #
#  Date Last-Modified:                                                   #
#  Python Version: 3.7                                                   #
#  Dependicies:                                                          #
#  Description: A                                                        #
#                                                                        #
##########################################################################

import collections
import matplotlib.pyplot as plt
import random

import numpy as np
import random_graph as random_graph_creation
import ring_group_graph as ring_group_creation


# Each return the search time
def random_graph_search_all(graph, vertex_s, vertex_t):
    search_time = 0
    current_vertex = vertex_s

    while True:
        neighbours = list(graph[current_vertex])
        random.shuffle(neighbours)

        for neighbour in neighbours:
            search_time += 1
            if neighbour == vertex_t:
                return search_time

        current_vertex = neighbours[0]


def random_graph_search_one(graph, vertex_s, vertex_t):
    search_time = 0
    current_vertex = vertex_s

    while True:
        neighbour = random.sample(graph[current_vertex], 1)[0]
        search_time += 1
        if neighbour == vertex_t:
            return search_time

        current_vertex = neighbour


def random_graph_search_relative(graph, vertex_s, vertex_t, num_nodes, p):
    search_time = 0
    current_vertex = vertex_s
    estimated_degree = int(num_nodes * p)

    while True:
        neighbours = list(graph[current_vertex])
        random.shuffle(neighbours)

        neighbours = neighbours[:estimated_degree]

        for neighbour in neighbours:
            search_time += 1
            if neighbour == vertex_t:
                return search_time

        current_vertex = neighbour


def ring_group_graph_search_group(m, k, p, q, graph, vertex_s, vertex_t):
    search_time = 0
    start_group = group_number(vertex_s, k)
    target_group = group_number(vertex_t, k)
    current_vertex = vertex_s
    break_chance = q / (p * k)

    ordered_groups = list(construct_group_priority_picking(m, k, start_group, target_group))

    while True:
        neighbours = list(graph[current_vertex])
        random.shuffle(neighbours)
        old_group = group_number(current_vertex, k)
        best_group_found = ordered_groups[-1]

        current_vertex = neighbours[0]

        for neighbour in neighbours:
            search_time += 1
            if neighbour == vertex_t:
                return search_time
            else:
                neighbour_group = group_number(neighbour, k)
                test_1 = ordered_groups.index(neighbour_group)
                test_2 = ordered_groups.index(best_group_found)

                if neighbour_group != old_group and (ordered_groups.index(neighbour_group) < ordered_groups.index(best_group_found)):
                    best_group_found = neighbour_group
                    current_vertex = neighbour

            if best_group_found == (ordered_groups[0] or ordered_groups[1] or ordered_groups[2]):
                break

        if random.random() < break_chance:
            current_vertex = neighbours[0]


# Probability of finding the target node at every stage, increases as you search for random graphs, introduce small probability of moving to a random neighbour

def ring_group_graph_search_new(m, k, p, q, graph, vertex_s, vertex_t):
    search_time = 0
    target_group = group_number(vertex_t, k)  #
    break_chance = (q / (p * k * 3)) * 0.01
    current_vertex = vertex_s

    if target_group == 0:
        high_groups = [0, m - 1, 1]
    else:
        high_groups = [target_group, (target_group - 1), (target_group + 1) % m]

    while True:
        neighbours = list(graph[current_vertex])
        random.shuffle(neighbours)
        current_group = group_number(current_vertex, k)
        vertex_changed = False

        if current_group in high_groups:
            for neighbour in neighbours:
                search_time += 1
                neighbour_group = group_number(neighbour, k)
                if neighbour == vertex_t:
                    return search_time
                if neighbour_group in high_groups:
                    current_vertex = neighbour
                    vertex_changed = True
            if not vertex_changed:
                current_vertex = neighbours[0]
        # if random.random() < break_chance:
        #    current_vertex = neighbours[0]
        else:
            search_time += 1
            current_vertex = neighbours[0]


def ring_group_graph_search_partial(m, k, p, q, graph, vertex_s, vertex_t):
    search_time = 0
    target_group = group_number(vertex_t, k)  #
    break_chance = (q / (p * k * 3)) * 0.01
    current_vertex = vertex_s

    if target_group == 0:
        high_groups = [0, m - 1, 1]
    else:
        high_groups = [target_group, (target_group - 1), (target_group + 1) % m]

    while True:
        neighbours = list(graph[current_vertex])
        random.shuffle(neighbours)
        current_group = group_number(current_vertex, k)
        vertex_changed = False

        if current_group in high_groups:
            for neighbour in neighbours:
                search_time += 1
                neighbour_group = group_number(neighbour, k)
                if neighbour == vertex_t:
                    return search_time
                if neighbour_group in high_groups:
                    current_vertex = neighbour
                    vertex_changed = True
            if not vertex_changed:
                current_vertex = neighbours[0]
        if random.random() < break_chance:
           current_vertex = neighbours[0]
        else:
            search_time += 1
            current_vertex = neighbours[0]


def construct_group_priority_picking(m, k, start_group, target_group):
    ordered_groups = list()
    ordered_groups.append(target_group)

    # Find direction between start and target by assessing distance
    clockwise_distance = 0
    counter_clockwise_distance = 0
    distance_calculator_location = start_group

    direction = ''

    while distance_calculator_location != target_group:
        if distance_calculator_location == m - 1:
            distance_calculator_location = 0
        else:
            distance_calculator_location += 1
        clockwise_distance += 1

    distance_calculator_location = start_group
    while distance_calculator_location != target_group:
        if distance_calculator_location == 0:
            distance_calculator_location = (m - 1)
        else:
            distance_calculator_location -= 1
        counter_clockwise_distance += 1

    if clockwise_distance < counter_clockwise_distance:
        direction = 'clockwise'
        ordered_groups.append((target_group - 1) % m)
        ordered_groups.append((target_group + 1) % m)
    else:
        direction = 'counter-clockwise'
        ordered_groups.append((target_group + 1) % m)
        ordered_groups.append((target_group - 1) % m)

    for i in range((m - 3)):
        if direction == 'clockwise':
            ordered_groups.append((target_group - 2 - i) % m)
            if len(ordered_groups) > m:
                ordered_groups.append((target_group + 2 + i) % m)
        else:  # direction == 'counter-clockwise':
            ordered_groups.append((target_group + 2 + i) % m)
            if len(ordered_groups) > m:
                ordered_groups.append((target_group - 2 - i) % m)

    return ordered_groups


def group_number(vertex, k):
    return vertex // k


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


def to_distribution(data):
    distribution = {}

    for value in data:
        if value in distribution.keys():
            distribution[value] += 1
        else:
            distribution[value] = 1

    return distribution


def print_data(data):
    data_array = create_array(data)
    mean = np.mean(data_array)
    range_data = np.ptp(data_array)
    variance = np.var(data_array)
    print('mean ' + str(mean))
    print('range ' + str(range_data))
    print('variance ' + str(variance))
    print()
    return None


def plot_random_search(num_nodes, p, num_graphes, name):
    searches_all = [None] * num_graphes
    searches_relative = [None] * num_graphes
    searches_one = [None] * num_graphes
    vertex_t = 0
    vertex_s = 0

    for i in range(num_graphes):
        # print("Making graph: " + str(i))
        while vertex_t == vertex_s:
            vertex_s = random.randint(1, num_nodes)
            vertex_t = random.randint(1, num_nodes)

        random_graph = random_graph_creation.load_graph(num_nodes, p)  # total_nodes, out_degree

        searches_all[i] = random_graph_search_all(random_graph, vertex_s, vertex_t)
        searches_relative[i] = random_graph_search_relative(random_graph, vertex_s, vertex_t, num_nodes, p)
        searches_one[i] = random_graph_search_one(random_graph, vertex_s, vertex_t)

    searches_frequency_all = to_distribution(searches_all)
    searches_frequency_relative = to_distribution(searches_relative)
    searches_frequency_one = to_distribution(searches_one)

    create_scatter_plot("Search Time", 'Frequency', '')
    x_data = []
    y_data = []
    for search in searches_frequency_all:
        x_data += [search]
        y_data += [searches_frequency_all[search]]
        plt.plot(x_data, y_data, marker='.', linestyle='None', color=(0, 0, 0, 1))  # linestyle='None'
    plt.savefig('Question3/Random/All' + name)

    create_scatter_plot("Search Time", 'Frequency', '')
    x_data = []
    y_data = []
    for search in searches_frequency_relative:
        x_data += [search]
        y_data += [searches_frequency_relative[search]]
        plt.plot(x_data, y_data, marker='.', linestyle='None', color=(0, 0, 0, 1))  # linestyle='None'
    plt.savefig('Question3/Random/Relative' + name)

    create_scatter_plot("Search Time", 'Frequency', '')
    x_data = []
    y_data = []
    for search in searches_frequency_one:
        x_data += [search]
        y_data += [searches_frequency_one[search]]
        plt.plot(x_data, y_data, marker='.', linestyle='None', color=(0, 0, 0, 1))  # linestyle='None'
    plt.savefig('Question3/Random/One' + name)

    print('All')
    print_data(searches_frequency_all)
    print('Relative')
    print_data(searches_frequency_relative)
    print('One')
    print_data(searches_frequency_one)


def plot_ring(m, k, p, q, num_graphes, name):
    num_nodes = m * k
    searches_random = [None] * num_graphes
    searches_group = [None] * num_graphes
    searches_partial = [None] * num_graphes
    vertex_t = 0
    vertex_s = 0

    for i in range(num_graphes):
        print("Making graph: " + str(i))
        while vertex_t == vertex_s:
            vertex_s = random.randint(1, num_nodes)
            vertex_t = random.randint(1, num_nodes)

        ring_graph = ring_group_creation.load_graph(m, k, p, q)  # total_nodes, out_degree

        searches_random[i] = random_graph_search_all(ring_graph, vertex_s, vertex_t)
        searches_group[i] = ring_group_graph_search_group(m, k, p, q, ring_graph, vertex_s, vertex_t)
        searches_partial[i] = ring_group_graph_search_partial(m, k, p, q, ring_graph, vertex_s, vertex_t)

    searches_frequency_random = to_distribution(searches_random)
    searches_frequency_group = to_distribution(searches_group)
    searches_frequency_partial = to_distribution(searches_partial)

    create_scatter_plot("Search Time", 'Frequency', '')
    x_data = []
    y_data = []
    for search in searches_frequency_random:
        x_data += [search]
        y_data += [searches_frequency_random[search]]
        plt.plot(x_data, y_data, marker='.', linestyle='None', color=(0, 0, 0, 1))  # linestyle='None'
    plt.savefig('Question3/Ring/random' + name)

    create_scatter_plot("Search Time", 'Frequency', '')
    x_data = []
    y_data = []
    for search in searches_frequency_group:
        x_data += [search]
        y_data += [searches_frequency_group[search]]
        plt.plot(x_data, y_data, marker='.', linestyle='None', color=(0, 0, 0, 1))  # linestyle='None'
    plt.savefig('Question3/Ring/group' + name)

    create_scatter_plot("Search Time", 'Frequency', '')
    x_data = []
    y_data = []
    for search in searches_frequency_partial:
        x_data += [search]
        y_data += [searches_frequency_partial[search]]
        plt.plot(x_data, y_data, marker='.', linestyle='None', color=(0, 0, 0, 1))  # linestyle='None'
    plt.savefig('Question3/Ring/partial' + name)

    print('Random')
    print_data(searches_frequency_random)
    print('Group')
    print_data(searches_frequency_group)
    print('Partial')
    print_data(searches_frequency_partial)


# num_nodes_random = 100
# random_p = 0.5
# num_random_graphs = 10000
#
# name = '05'
# print(name)
# plot_random_search(num_nodes_random, random_p, num_random_graphs, name)
# name = '025'
# print(name)
# random_p = 0.25
# #plot_random_search(num_nodes_random, random_p, num_random_graphs, name)

m = 10
k = 10
p = 0.4
q = 0.1
num_ring_graphs = 10000
name = 'p04'
search_time_ring = plot_ring(m, k, p, q, num_ring_graphs, name)

# p=q=0.25, random graph comparison, increasing chance of query returning the target
