#collaborator: Richard Lee 

from networkx.algorithms import components
from networkx.classes.function import nodes 
import networkx as nx
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import collections
import operator


def read_old_edge(file_name):
    G = nx.Graph()
    G = nx.read_edgelist(file_name, create_using= nx.DiGraph(),delimiter='\t', nodetype=str)

    figure(figsize=(10, 10))
    nx.draw(G, with_labels=True)
    plt.show()

def read_new_edge(file_name):
    G = nx.Graph()
    G = nx.read_edgelist(file_name,delimiter='\t', nodetype=str)

    # for each authors 
    connected = []
    for node in G.nodes():
        connected.append([node] + list(G.neighbors(node)))

    print(connected)

    figure(figsize=(10, 10))
    nx.draw(G, with_labels=True)
    plt.show()

def get_num_common_friends(person1, person2, graph):
    f_person1 = set(graph.neighbors(person1)) 
    f_person2 = set(graph.neighbors(person2))
    common_friends = (f_person1).intersection(f_person2)
    return len(common_friends)


def common_friends_number(G, X):
    '''get list of recommendations for author X''' 

    # Get Author X's friends 
    friends = set(G.neighbors(X))

    # Get friend's friends
    fof = set()

    for i in friends:
        fof.update(set(G.neighbors(i)) - friends)

    fof = fof - {X}

    mutual_friends = {} 

    for i in fof:
        count = get_num_common_friends(X, i, G)
        if count >= 1:
            mutual_friends[i] = count

    result = [i[0] for i in sorted(mutual_friends.items(), key=operator.itemgetter(1),reverse=True)]
    print(result)

def jaccard_index(G, X):
    '''recommend by jaccard_index '''

    # Get Author X's friends 
    friends = set(G.neighbors(X))

    others = [(X, i) for i in G.nodes if i != X]

    # Get Jaccard coefficient 
    jacc = nx.jaccard_coefficient(G, others)

    result = []

    for u, v, p in jacc:
        # Recommend friends with jaccard score 
        if p > 0:
            result.append([u, v, p])
    result = [i[1] for i in sorted(result, key=operator.itemgetter(2),reverse=True) if i[1] not in list(friends)+[X]]
    return result

def adamic_adar_index(G, X):
    '''recommend by adamic '''

    # Get Author X's friends 
    friends = set(G.neighbors(X))

    others = [(X, i) for i in G.nodes if i != X]

    # Get Adamic Adar index 
    preds = nx.adamic_adar_index(G, others)

    result = []

    for u, v, p in preds:

        # Recommending friends who have an Adamic Adar score
        if p > 0:
            result.append([u, v, p])

    result = [i[1] for i in sorted(result, key=operator.itemgetter(2),reverse=True) if i[1] not in list(friends)+[X]]
    return result

def read_files(filename):

    f = open(filename, encoding='cp437')
    edges = []
    for i in f.readlines():
        edges.append(i[:-1].split('\t'))
    edges = np.array(edges)
    return edges

old = read_files('old_edges.txt')
nodes = set(old.flatten())
G = nx.Graph()
G.add_nodes_from(nodes, weight=1)
G.add_edges_from(old, year="2010-2016")
new = read_files('new_edges.txt')

#read_old_edge('old_edges.txt')
#read_new_edge('new_edges.txt')
# G = nx.Graph()
# G = nx.read_edgelist('new_edges.txt', delimiter='\t', nodetype=str)
# print("printing now")
# print()
# common_friends_number(G, 'Demis Hassabis')
# print("printing jaccard")
# print()
# print(jaccard_index(G, 'Demis Hassabis'))
# print()
# print("printing adamic")
# print(adamic_adar_index(G, 'Demis Hassabis'))