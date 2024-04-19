"""
Author: Tomasz Hachaj, 2024
Department of Applied Computer Science in AGH University of Krakow, Poland.
https://home.agh.edu.pl/~thachaj/

Source codes for paper:
Rough neighborhood graph: a method for proximity modeling and data clustering

Rough neighborhood-based clustering implementation
"""

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from tqdm import tqdm
import itertools
import math

# Helper function
def target_function(list_cl, dataset_size, par1=0.5):
    my_sum = 0
    k = set()
    for l in list_cl:
        k |= set(l)
        my_sum += len(l)
    # the lower is better - common part of clusters
    val1 = math.fabs(my_sum - dataset_size)
    # the lower is better - data covered
    val2 = math.fabs(my_sum - len(k))
    par2 = 1 - par1
    return (par1 * val1 + par2 * val2)

# Helper function
def check_if_contains(nodes_dict, i, e_id):
    if e_id in nodes_dict.keys():
        for ll in nodes_dict[e_id]:
            if ll[1] == i:
                return True
    return False

# Helper function
def if_contain_edge(nodes_dict, v):
    node2 = nodes_dict[v[1]]
    for v2 in node2:
        if v[0] == v2[1] and v[1] == v2[0]:
            return True
    return  False


# Make Rough neighborhood graph
def make_graph(emb_array_copy, edges_count):
    nodes_dict = {}
    for a in range(emb_array_copy.shape[0]):
        nodes_dict[a] = []
    for i in tqdm(range(emb_array_copy.shape[0]), desc="Calculating graph"):
        yy = np.expand_dims(emb_array_copy[i,], axis=0)
        ww = pairwise_distances(X=emb_array_copy, Y=yy, n_jobs=-1)
        ww = ww[:, 0]
        indexes = np.argsort(ww)
        edges_count_help = 0
        j = 1
        while edges_count_help < edges_count and edges_count_help < indexes.shape[0]:
            e_id = indexes[j]
            dist = -1
            if e_id != i:
                # if True:
                if not check_if_contains(nodes_dict, i, e_id):
                    dist = ww[indexes[j]]
                    nodes_dict[i].append((i, e_id, dist, False))
                    nodes_dict[e_id].append((e_id, i, dist, False))
                else:
                    a = 0
                    stop_loop = False
                    while a < len(nodes_dict[e_id]) and not stop_loop:
                        ll = nodes_dict[e_id][a]
                        if ll[0] == e_id and ll[1] == i:
                            nodes_dict[e_id][a] = (ll[0], ll[1], ll[2], True)
                            dist = ll[2]
                            stop_loop = True
                        a = a + 1

                    a = 0
                    stop_loop = False
                    while a < len(nodes_dict[i]) and not stop_loop:
                        ll = nodes_dict[i][a]
                        if ll[0] == i and ll[1] == e_id:
                            nodes_dict[i][a] = (ll[0], ll[1], ll[2], True)
                            stop_loop = True
                        a = a + 1

                edges_count_help = edges_count_help + 1
            j = j + 1
            # If there are several objects in same distance, we add all of them
            if j < indexes.shape[0] and j > edges_count:
                if ww[indexes[j]] == dist:
                    edges_count_help = edges_count_help - 1
    return nodes_dict

# Helper function
def find_element(my_l, ele):
    try:
        index_value = my_l.index(ele)
    except ValueError:
        index_value = -1
    return index_value

# Helper function
def copy_edges(n, list_to_copy):
    for v in n:
        list_to_copy.append(v)
    return list_to_copy

# Helper function, run clustering on graph
def run_clustering(nodes_dict):
    all_clusters_l = []
    all_clusters_b = []

    id_k = 0
    p_bar = tqdm(range(len(nodes_dict)), 'Walking through graph')
    for k in nodes_dict.keys():
        p_bar.update(1)
        if_contains = False
        for kls in all_clusters_l:
            if k in (kls):
                if_contains = True
        if not if_contains:
            id_k = id_k + 1
            neighborhood = []
            next_neighborhood = []
            claster_l = []
            claster_b = []
            claster_l.append(k)

            copy_edges(nodes_dict[k], neighborhood)
            while len(neighborhood) > 0:
                n = neighborhood.pop(0)
                start_id = n[0]
                end_id = n[1]
                is_lower = n[3]
                if is_lower:
                    if end_id in claster_b:
                        claster_b.remove(end_id)
                    if end_id not in claster_l:
                        claster_l.append(end_id)
                        copy_edges(nodes_dict[end_id], next_neighborhood)
                else:
                    if end_id not in claster_b:
                        claster_b.append(end_id)
                while len(next_neighborhood) > 0:
                    neighborhood.append(next_neighborhood.pop(0))
            all_clusters_l.append(claster_l)
            all_clusters_b.append(claster_b)
    p_bar.refresh()
    return [all_clusters_l, all_clusters_b]

# Helper function
def findsubsets(s, n):
    return [set(i) for i in itertools.combinations(s, n)]

# Helper function, optimize clustering, might remove some clusters
def find_max_clusters(cm, dataset_size, min_size = 0.1, par1 = 0.5):
    cm_help = []
    cm_len = []
    for a in range(len(cm)):
        l = cm[a]
        if len(l) >= dataset_size * min_size:
            cm_help.append(l)
            cm_len.append(len(l))
    if len(cm_len) == 1:
        return cm_help
    messure_res = []
    my_list = [x for x in range(0, len(cm_help))]

    p_bar2 = tqdm(range(len(cm_len)), 'Selecting clusters')

    for a in range(1, len(cm_len) + 1):
        p_bar2.update(1)
        sub_sec = findsubsets(my_list, a)

        for ss in sub_sec:
            cm_list_to_measure = []
            for ll in ss:
                cm_list_to_measure.append(cm_help[ll])
            mmmm = target_function(cm_list_to_measure, dataset_size, par1)
            messure_res.append((mmmm, cm_list_to_measure, ss))
        xxxx = 0
        xxxx += 1

    min_val = float('inf')
    id_help = 0
    for a in range(len(messure_res)):
        xxx = messure_res[a]
        if xxx[0] < min_val:
            id_help = a
            min_val = xxx[0]

    return messure_res[id_help][1]

# Helper function, assigns clusters id to objects
def calculate_clustering(nodes_dict, cm_help):
    classes = np.zeros(len(nodes_dict.keys()), dtype=int)
    c_id = 1
    for a in range(len(cm_help)):
        ccc = cm_help[a]
        for b in range(len(ccc)):
             classes[ccc[b]] = c_id
        c_id += 1
    return classes

# Execute Rough neighborhood-based clustering
def RNBC(X, edges_count, min_size=0.1, par1=0.5,use_lower=True,optimize_clusters=True):
    """

    :param X: dataset
    :param edges_count: number of edges in Rough neighborhood graph
    :param min_size: minimal size of the cluster to be added
    :param par1: scalling parameter in optimization function, should be in [0,1], only used when use_lower=True
    :param use_lower: use only lower approximation of clusters
    :param optimize_clusters: run clusters optimization (might remove some clusters, only used when use_lower=True)
    :return:
    """
    # Make Rough neighborhood graph
    my_graph = make_graph(X, edges_count)
    # Perform clustering, get lower approximation and border
    [all_clusters_lower, all_clusters_border] = run_clustering(my_graph)

    # Remove clusters with size < min_size
    all_clusters_lower_help = []
    all_clusters_border_help = []
    for a in range(len(all_clusters_lower)):
        l = all_clusters_lower[a]
        u = all_clusters_border[a]
        if len(l) >= len(my_graph) * min_size:
            all_clusters_lower_help.append(l)
            all_clusters_border_help.append(u)

    all_clusters = all_clusters_lower_help
    # Update border
    for b in range(len(all_clusters_border_help)):
        for a in range(len(all_clusters)):
            res = [i for i in all_clusters_border_help[b] if i not in all_clusters_lower_help[a]]
            all_clusters_border_help[b] = res

    # Add border if you want to get upper approximation,
    if not use_lower:
        for a in range(len(all_clusters)):
            all_clusters[a] += all_clusters_border_help[a]

    # Select clusters
    if optimize_clusters:
        cm_help = find_max_clusters(all_clusters, len(my_graph), min_size, par1)
    else:
        cm_help = all_clusters

    # Get crisp clustering
    classes = calculate_clustering(my_graph, cm_help)
    # Return crisp clustering
    return classes