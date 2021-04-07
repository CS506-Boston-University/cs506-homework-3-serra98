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
    
    sortlist = sorted(mutual_friends.items(), key=operator.itemgetter(1),reverse=True)
    result = [i[0] for i in sortlist]
    return result

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

        # Recommend friends with Adamic 
        if p > 0:
            result.append([u, v, p])

    result = [i[1] for i in sorted(result, key=operator.itemgetter(2),reverse=True) if i[1] not in list(friends)+[X]]
    return result

#read_old_edge('old_edges.txt')
#read_new_edge('new_edges.txt')
# G = nx.Graph()
# G = nx.read_edgelist('old_edges.txt', delimiter='\t', nodetype=str)
# print("printing now")
# print()
#common_friends_number(G, 'Demis Hassabis')
# print("printing jaccard")
# print()
# print(jaccard_index(G, 'Demis Hassabis'))
# print()
# print("printing adamic")
# print(adamic_adar_index(G, 'Demis Hassabis'))




G_old = nx.Graph()
G_old = nx.read_edgelist('old_edges.txt', delimiter='\t', nodetype=str)
track_old = G_old.degree() 
G_new = nx.Graph()
G_new = nx.read_edgelist('new_edges.txt', delimiter='\t', nodetype=str)
track_new = G_new.degree()

#Get Authors with new connections formed more than 10+ in 2017-2018 
actual_new = [node for node,degree in dict(track_new).items() if degree >= 10] 
#old authors 
old_author = [node for node,degree in dict(track_old).items()]
#only get authors that were both in old_edges.txt and new_edges.txt (but degree >= 10  in new edges)
overlapped_actual = set(actual_new).intersection(set(old_author))
# print("printing overlapping authors")
# print(overlapped_actual)

# f2) 1st Approach
accuracy = {}

for i in overlapped_actual:
    
    # take the 10 first recommendations for each user

    #change friends_list to use different method for each accuracy score average 
    friends_list = (common_friends_number(G_old,i))
    #friends_list = (jaccard_index(G_old,i))
    #friends_list = (adamic_adar_index(G_old,i))
    rec = friends_list[0:10]
    # Get Author i's friends 
    friends = set(G_new.neighbors(i))
    actual_friends = friends 
        
    # Get the number formed during 2017-2018
    overlapped = set(rec).intersection(actual_friends)
    accuracy[i] = len(overlapped) / len(actual_friends)
         
# Get Average
score = [x for _, x in accuracy.items()]
average = sum(score) / len(score)
print("each author's accuracy with Score")
print(accuracy)
print("average score of overall accuracy:" , average)

# f2) 2nd Approach 
status = {}

for i in overlapped_actual:
    #change friends_list to use different method for each accuracy rank average 
    #friends_list = common_friends_number(G_old,i)
    #friends_list = (jaccard_index(G_old,i))
    friends_list = (adamic_adar_index(G_old,i))
    # Get Author i's friends 
    friends = set(G_new.neighbors(i))
    actual_friends = friends 
    
    # calculate the rank 
    for i in actual_friends:
        if i in friends_list:
            rank = friends_list.index(i)
        else:
            rank = len(friends_list) + 1 
        status[i] = rank
    
# Get Average 
rank = [x for _, x in status.items()]
average = sum(rank) / len(rank)
print("each author's accuracy with rank")
print(rank)
print("average score of overall accuracy:" , average)

