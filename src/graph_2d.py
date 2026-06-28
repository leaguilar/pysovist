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
from collections import deque
from functools import reduce
import heapq

## 2. Graph Generation Helpers
def knn_graph(grid:np.array,k:int=4):
    tree = KDTree(grid)
    _, idx = tree.query(grid,k+1) # 4 nearest neighbors by default, excluding self
    edges = [[(s,j[i]),(j[i],s)] for s,j in zip(idx[:,0],idx[:,1:]) for i in range(j.shape[0])]
    edges = np.unique(np.array(edges).reshape(-1,2),axis=0)
    return edges

def adjacency_matrix(grid:np.array,edges:np.array):
    adj = [[] for _ in range(grid.shape[0])]
    for u, v in edges:
        adj[u].append(v)
    return adj

def adjacency_matrix_w(grid:np.array,edges:np.array):
    edge_weights = np.linalg.norm(grid[edges[:,1]] - grid[edges[:,0]],axis=1)
    adj = [[] for _ in range(grid.shape[0])]
    for (u, v), w in zip(edges, edge_weights):
        adj[u].append((v, w))
    return adj

def delaunay():
    #TODO: use SciPy delaunay triangulation to make graph
    return

def segment_intersection_mask(a, b, idx, eps=1e-12):
    p = a[0]
    r = a[1]-a[0]
    q = b[0]
    s = b[1]-b[0]
    denom = r[0]*s[1]-r[1]*s[0]
    if abs(denom)<eps:
        return False  # collinear
    qp = q-p
    t = (qp[0]*s[1]-qp[1]*s[0])/denom
    u = (qp[0]*r[1]-qp[1]*r[0])/denom
    if (0<=t<=1) and (0<=u<=1): #intersection in segment bounds
        return int(idx)
    else:
        return False

def line_fill(a:np.array,b:np.array)->np.array:
    xrange = max(a[0],b[0])-min(a[0],b[0])
    yrange = max(a[1],b[1])-min(a[1],b[1])
    xincrs = np.arange(xrange)/xrange
    yincrs = np.arange(yrange)/yrange
    xs = np.stack([np.array(np.ones(yrange)*a[1]+yincrs*(b[0]-a[0])),np.array(np.ones(yrange)*a[1]+yincrs*(b[1]-a[1]))],axis=1)
    ys = np.stack([np.array(np.ones(xrange)*a[0]+xincrs*(b[0]-a[0])),np.array(np.ones(xrange)*a[0]+xincrs*(b[1]-a[1]))],axis=1)
    return np.unique(np.row_stack([xs.astype(int),ys.astype(int)]),axis=0)

def disconnect_walls(grid:np.array|None,edges:np.array,plan:np.array|None,img:np.array|None,raster:bool=False):
    # cut graph edges coincident with any wall
    if raster == False:
        #vectorized workflow: edges as lines, find intersections, remove if intersection
        # bounding boxes: bb-edge/wall-x/y-lower/upper
        bb_e_x_l = grid[edges,0].min(axis=1)
        bb_e_x_u = grid[edges,0].max(axis=1)
        bb_e_y_l = grid[edges,1].min(axis=1)
        bb_e_y_u = grid[edges,1].max(axis=1)
        bb_w_x_l = plan[...,0].min(axis=1)
        bb_w_x_u = plan[...,0].max(axis=1)
        bb_w_y_l = plan[...,1].min(axis=1)
        bb_w_y_u = plan[...,1].max(axis=1)
        # prune edges outside corresponding bounding boxes
        w_ex_x_l = [np.where(bb_e_x_u<i) for i in bb_w_x_l]
        w_ex_x_u = [np.where(bb_e_x_l>i) for i in bb_w_x_u]
        w_ex_y_l = [np.where(bb_e_y_u<i) for i in bb_w_y_l]
        w_ex_y_u = [np.where(bb_e_y_l>i) for i in bb_w_y_u]
        sets_u = [reduce(np.union1d,arrays) for arrays in zip(w_ex_x_l,w_ex_x_u,w_ex_y_l,w_ex_y_u)]
        sets = [reduce(np.setdiff1d,arrays) for arrays in zip(np.tile(np.arange(edges.shape[0]),len(sets_u)).reshape(len(sets_u),edges.shape[0]),sets_u)]
        # solve intersections for curves within bounding boxes
        masks = [segment_intersection_mask(grid[edges[q]],plan[st],q) for st,i in enumerate(sets) for q in i]
        masks = np.array([element if type(element)==int else -1 for element in masks])
        masks = masks[masks>=0]
        edges_d = np.delete(edges,masks,axis=0)
        return edges_d
    elif raster == True:
        #implementation for rasterized image
        #img: sparse sections from plan
        tree_lines = KDTree(img)
        edges_grid = [line_fill(grid[edge[0]],grid[edge[1]]) for edge in edges]
        intersection = np.array([[e_g[0],e_g[1],i,0] for i,e in enumerate(edges_grid) for e_g in e])
        # find overlapping pixels, delete
        dists,_ = tree_lines.query(intersection[:,:2])
        int_overlap = intersection[dists<1.5] #approx. ≤√2
        int_edges = np.unique(int_overlap[:,2])
        edges_d = np.delete(edges,int_edges,axis=0)
        return edges_d

    #rasterized workflow: find sparse intersections, remove

## 3. Graph Centrality Helpers
def closeness(grid:np.array,edges:np.array,weighted=True): # Dijkstra's algorithm
    #edge_weights = np.linalg.norm(grid[edges[:,1]] - grid[edges[:,0]])
    if not weighted: # use BFS
        adj = adjacency_matrix(grid,edges)
        if len(grid) > 1000:
            closeness = bfs_approx(adj)
        else:
            closeness = bfs_exact(adj)
    else:
        adj = adjacency_matrix_w(grid,edges)
        if len(grid) > 1000:
            closeness = dijkstra_approx(adj)
        else:
            closeness = dijkstra_exact(adj)
    return closeness

def bfs_approx(adj:np.array,k:int=256):
    n = len(adj)
    np.random.seed(0)
    landmarks = np.random.choice(n,min(k,n),replace=False)
    dist_sum = np.zeros(n,dtype=np.float64)
    count = np.zeros(n,dtype=np.int32)
    for s in landmarks:
        dist = np.full(n,-1,dtype=np.int32)
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
        mask = dist >= 0
        dist_sum[mask] += dist[mask]
        count[mask] += 1
    mean_dist = dist_sum / np.maximum(count, 1)
    estimate = 1.0/np.maximum(mean_dist,1e-12)
    return estimate # (V,1) array

def bfs_exact(adj:np.array):
    n = len(adj)
    closeness = np.zeros(n)
    
    for s in range(n):
        dist = np.full(n,-1,dtype=np.int32)
        dist[s] = 0
        queue = deque([s])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u]+1
                    queue.append(v)
        reachable = dist >= 0
        total_dist = dist[reachable].sum()
        r = reachable.sum()
        if total_dist > 0:
            closeness[s] = ((r-1)/total_dist)*((r-1)/(n-1))
    return closeness # (V,1) array

def dijkstra(adj:np.array,source:int):
    n = len(adj)
    dist = np.full(n, np.inf, dtype=np.float64)
    dist[source] = 0.0
    heap = [(0.0, source)]
    while heap:
        du, u = heapq.heappop(heap)
        if du > dist[u]:
            continue
        for v, w in adj[u]:
            alt = du + w
            if alt < dist[v]:
                dist[v] = alt
                heapq.heappush(heap, (alt, v))
    return dist

def dijkstra_exact(adj:np.array):
    n = len(adj)
    closeness = np.zeros(n, dtype=np.float64)

    for s in range(n):
        dist = dijkstra(adj, s)
        reachable = np.isfinite(dist)
        reachable[s] = False
        r = reachable.sum()
        if r == 0:
            closeness[s] = 0.0
            continue
        total_dist = dist[reachable].sum()
        closeness[s] = (r / total_dist) * (r / (n - 1))
    return closeness

def dijkstra_approx(adj:np.array, k=256):
    n = len(adj)

    np.random.seed(0)
    landmarks = np.random.choice(n, min(k, n), replace=False)

    dist_sum = np.zeros(n, dtype=np.float64)
    count = np.zeros(n, dtype=np.int32)

    for s in landmarks:
        dist = dijkstra(adj, s)
        reachable = np.isfinite(dist)
        reachable[s] = False
        dist_sum[reachable] += dist[reachable]
        count[reachable] += 1

    closeness_est = np.zeros(n, dtype=np.float64)
    valid = count > 0
    closeness_est[valid] = count[valid] / dist_sum[valid]
    return closeness_est

def betweenness(grid:np.array,edges:np.array,weighted:bool=True):
    if not weighted: # use BFS
        adj = adjacency_matrix(grid,edges)
        if len(grid) > 1000:
            betweenness = brandes_approx_unw(adj)
        else:
            betweenness = brandes_exact_unw(adj)
    else:
        adj = adjacency_matrix_w(grid,edges)
        if len(grid) > 1000:
            betweenness = brandes_approx(adj)
        else:
            betweenness = brandes_exact(adj)
    return betweenness

def brandes_approx_unw(adj:np.array,k:int=256,normalized:bool=True):
    n = len(adj)
    betweenness = np.zeros(n, dtype=np.float64)

    np.random.seed(0)
    landmarks = np.random.choice(n, min(k, n), replace=False)

    for s in landmarks:
        stack = []
        pred = [[] for _ in range(n)]
        sigma = np.zeros(n, dtype=np.float64)   # number of shortest paths
        sigma[s] = 1.0
        dist = np.full(n, -1, dtype=np.int32)
        dist[s] = 0
        queue = deque([s])
        # Forward BFS pass
        while queue:
            u = queue.popleft()
            stack.append(u)
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
                if dist[v] == dist[u] + 1:
                    sigma[v] += sigma[u]
                    pred[v].append(u)
        # Backward dependency accumulation
        delta = np.zeros(n, dtype=np.float64)
        while stack:
            w = stack.pop()
            if sigma[w] == 0:
                continue
            for v in pred[w]:
                delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != s:
                betweenness[w] += delta[w]

    if len(landmarks) > 0:
        betweenness *= n / len(landmarks)
    if normalized and n > 2:
        betweenness *= 1.0 / ((n - 1) * (n - 2))
    return betweenness
        
def brandes_exact_unw(adj:np.array,normalized:bool=True):
    n = len(adj)
    betweenness = np.zeros(n, dtype=np.float64)

    for s in range(n):
        stack = []
        pred = [[] for _ in range(n)]
        sigma = np.zeros(n, dtype=np.float64)   # number of shortest paths
        sigma[s] = 1.0
        dist = np.full(n, -1, dtype=np.int32)
        dist[s] = 0
        queue = deque([s])
        # Forward BFS pass
        while queue:
            u = queue.popleft()
            stack.append(u)
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
                if dist[v] == dist[u] + 1:
                    sigma[v] += sigma[u]
                    pred[v].append(u)
        # Backward dependency accumulation
        delta = np.zeros(n, dtype=np.float64)
        while stack:
            w = stack.pop()
            for v in pred[w]:
                delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != s:
                betweenness[w] += delta[w]
    if normalized and n > 2:
        betweenness *= 1.0 / ((n - 1) * (n - 2))
    return betweenness

def brandes_exact(adj:np.array,normalized:bool=True):
    n = len(adj)
    betweenness = np.zeros(n, dtype=np.float64)
    for s in range(n):
        stack = []
        pred = [[] for _ in range(n)]
        sigma = np.zeros(n, dtype=np.float64)
        sigma[s] = 1.0
        dist = np.full(n, np.inf, dtype=np.float64)
        dist[s] = 0.0
        pq = [(0.0, s)]
        # Forward Dijkstra pass
        while pq:
            dist_u, u = heapq.heappop(pq)
            if dist_u > dist[u]:
                continue
            stack.append(u)
            for v, weight in adj[u]:
                alt = dist[u] + weight
                if alt < dist[v]:
                    dist[v] = alt
                    heapq.heappush(pq, (alt, v))
                    sigma[v] = sigma[u]
                    pred[v] = [u]
                elif alt == dist[v]:
                    sigma[v] += sigma[u]
                    pred[v].append(u)
        # Backward dependency accumulation
        delta = np.zeros(n, dtype=np.float64)
        while stack:
            w = stack.pop()
            if sigma[w] == 0:
                continue
            for v in pred[w]:
                delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != s:
                betweenness[w] += delta[w]
    # Directed-graph normalization
    if normalized and n > 2:
        betweenness *= 1.0 / ((n - 1) * (n - 2))

    return betweenness

def brandes_approx(adj:np.array,k:int=256,normalized:bool=True):
    n = len(adj)
    betweenness = np.zeros(n, dtype=np.float64)

    np.random.seed(0)
    landmarks = np.random.choice(n, min(k, n), replace=False)

    for s in landmarks:
        stack = []
        pred = [[] for _ in range(n)]
        sigma = np.zeros(n, dtype=np.float64)
        sigma[s] = 1.0
        dist = np.full(n, np.inf, dtype=np.float64)
        dist[s] = 0.0
        pq = [(0.0, s)]
        # Forward Dijkstra pass
        while pq:
            dist_u, u = heapq.heappop(pq)
            if dist_u > dist[u]:
                continue
            stack.append(u)
            for v, weight in adj[u]:
                alt = dist[u] + weight
                if alt < dist[v]:
                    dist[v] = alt
                    heapq.heappush(pq, (alt, v))
                    sigma[v] = sigma[u]
                    pred[v] = [u]
                elif alt == dist[v]:
                    sigma[v] += sigma[u]
                    pred[v].append(u)
        # Backward dependency accumulation
        delta = np.zeros(n, dtype=np.float64)
        while stack:
            w = stack.pop()
            if sigma[w] == 0:
                continue
            for v in pred[w]:
                delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != s:
                betweenness[w] += delta[w]

    if len(landmarks) > 0:
        betweenness *= n / len(landmarks)
    # Directed-graph normalization
    if normalized and n > 2:
        betweenness *= 1.0 / ((n - 1) * (n - 2))

    return betweenness

## 4. Metrics
def centrality(grid:np.array,edges:np.array,metric:str='degree',scale:str='linear',percentile:float=95):
    if metric == 'degree':
        in_degree = [np.argwhere(edges[:,1]==i).shape[0] for i,_ in enumerate(grid)]
        out_degree = [np.argwhere(edges[:,0]==i).shape[0] for i,_ in enumerate(grid)]
        cent_u = np.array(in_degree)+np.array(out_degree)
        cent = cent_u/grid.shape[0]
    if metric == 'closeness':
        cent = closeness(grid,edges,weighted=False)
    if metric == 'closeness_w':
        cent = closeness(grid,edges,weighted=True)
    if metric == 'betweenness':
        cent = betweenness(grid,edges,weighted=False)
    if metric == 'betweenness_w':
        cent = betweenness(grid,edges,weighted=True)

    # eliminate values outside percentile
    up = np.percentile(cent,percentile)
    lp = np.percentile(cent,100-percentile)
    cent = np.clip(cent,lp,up)

    #normalize centrality
    # scale: 'linear', 'log', 'sqrt'
    if scale == 'linear':
        return (cent-cent.min())/(cent.max()-cent.min())
    if scale == 'log':
        cent = cent-(cent.min()-1)
        cent = np.log(cent)
        return (cent-cent.min())/(cent.max()-cent.min())
    if scale == 'sqrt':
        cent = np.sqrt(cent)
        return (cent-cent.min())/(cent.max()-cent.min())
# centrality: eigenvector, harmonic, Katz, Laplacian
# view_depth
# view_integration
#1. Angular Choice
#2. Angular Integration
#3. Connectivity
#4. Intelligibility
#5. Mean Depth
