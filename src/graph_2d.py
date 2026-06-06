# /------------------------------------------------------------/
# **2D Graph Generation and Processing Helpers**
# -----
# *pysovist-dev* under MIT License
# -----
# This scripts includes generation and processing pipelines
# for graphs generated from 2D boundary grids.
# -----
# Best use case: centrality and connectivity measures.
# -----
# What's in the file:
# 1. imports
# 2. graph generation helpers
# 3. graph centrality helpers
# 4. metrics
# /------------------------------------------------------------/

## 1. Imports
import numpy as np
from scipy.spatial import KDTree
from collections import defaultdict, deque

## 2. Graph Generation Helpers
def knn_graph(grid:np.array,k:int=4):
    tree = KDTree(grid)
    _, idx = tree.query(grid,k+1) # 4 nearest neighbors by default, excluding self
    edges = [[(s,j[i]),(j[i],s)] for s,j in zip(idx[:,0],idx[:,1:]) for i in range(j.shape[0])]
    edges = np.unique(np.array(edges).reshape(-1,2),axis=0)
    return edges

## 3. Graph Centrality Helpers
def shortest_paths(grid:np.array,edges:np.array,weighted=True): # Dijkstra's algorithm
    edge_weights = np.linalg.norm(grid[edges[:,1]] - grid[edges[:,0]])
    if not weighted: # use BFS
        bfs(edges)
    else: #TODO: A* algorithm
        return
        #algorithm by Duan et al.?

    return

def bfs(edges:np.array, start:int, target:int):
    graph = defaultdict(list)

    for parent, child in np.asarray(edges):
        graph[int(parent)].append(int(child))

    queue = deque([int(start)])
    parent = {int(start): None}
    while queue:
        node = queue.popleft()
        if node == target:
            break
        for child in graph[node]:
            if child not in parent:
                parent[child] = node
                queue.append(child)
    
    if target not in parent:
        return None

    path = []
    node = int(target)

    while node is not None:
        path.append(node)
        node = parent[node]

    return queue, path[::-1]


## 4. Metrics
def centrality(grid:np.array,edges:np.array,method:str='degree'):
    if method == 'degree':
        cent = [np.argwhere(edges==i).shape[0] for i in edges]
    if method == 'closeness':
        shortest_paths
    if method == 'closeness_w':
        shortest_paths

    return cent
# centrality: betweenness, closeness, eigenvector, degree, harmonic, Katz, Laplacian, harmonic
# view_depth
# view_integration
#1. Angular Choice
#2. Angular Integration
#3. Connectivity
#4. Intelligibility
#5. Mean Depth