# Goal is to construct a random d-regular graph
import numpy as np
import scipy.sparse as sp
import networkx as nx
import matplotlib.pyplot as plt

# The following function constructs a random d-regular graph on n vertices
# with the random 2-regular subgraph (lining) obtained by taking only the first two matchings.
def d_regular(n,d):
    if (n)%2 != 0:
        raise ValueError("n must be even")
    matchings = np.empty((n//2,d,2))
    for i in range(d):
        perm = np.random.permutation(n)
        matchings[:, i, 0] = perm[:n//2]     # left endpoints
        matchings[:, i, 1] = perm[n//2:]     # right endpoints
      
    # A edges
    rows = matchings[..., 0].ravel()
    cols = matchings[..., 1].ravel()

    # Make symmetric in COO
    A = sp.coo_matrix(
        (np.ones(rows.size*2, dtype=np.int64),
         (np.concatenate([rows, cols]),
          np.concatenate([cols, rows]))),
        shape=(n,n)
    )
    # Repeat for the lining
    rows = matchings[:,:2, 0].ravel()
    cols = matchings[:,:2, 1].ravel()

    # Make symmetric in COO
    L = sp.coo_matrix(
        (np.ones(rows.size*2, dtype=np.int64),
         (np.concatenate([rows, cols]),
          np.concatenate([cols, rows]))),
        shape=(n,n)
    )
    return A,L

# The following function computes the n-th power of a sparse crs matrix A.
def sparse_matrix_power(A, n):
    if n < 0:
        raise ValueError("Negative powers not supported for sparse matrices")
    if n == 0:
        return sp.identity(A.shape[0], format=A.format)
    elif n == 1:
        return A.copy()
    elif n % 2 == 0:
        half = sparse_matrix_power(A, n // 2)
        return half @ half
    else:  # n % 2 == 1
        half = sparse_matrix_power(A, (n - 1) // 2)
        return A @ half @ half


# The following function samples from d_regular(n,d) and plots a ball of radius r around a randomly chosen vertex.
def plot_lining_ball(r,n,d):
    A,L = d_regular(n,d)
    A = A.tocsr()
    L = L.tocsr() 
    B = A + sp.identity(A.shape[0], format=A.format)
    B_power = sparse_matrix_power(B,r) #compute power
    root = np.random.randint(0,B.shape[0] - 1) #choose random root
    vertices = B_power.indices[np.arange(B_power.indptr[root],B_power.indptr[root + 1])] #get vertices in ball
    adj = A[vertices,:][:,vertices] #induced adjacency matrix
    lin = L[vertices,:][:,vertices] #induced lining matrix
    G = nx.from_scipy_sparse_array(adj)
    pos = nx.spring_layout(G)
    H = nx.from_scipy_sparse_array(lin)
    nx.draw(G,pos,node_color='blue',edge_color='blue',with_labels=True)
    nx.draw(H,pos,node_color='red',edge_color='red',with_labels=True)
    return plt.show()
    
# The following function takes as input a sparse adjacency matrix A and samples a ball of radius r around a random node

def plot_lining(n,d):
    A,L = d_regular(n,d)
    G = nx.from_scipy_sparse_array(A)
    pos = nx.spring_layout(G)
    H = nx.from_scipy_sparse_array(L)
    nx.draw(G,pos,node_color='blue',edge_color='blue',with_labels=True)
    nx.draw(H,pos,node_color='red',edge_color='red',with_labels=True)
    return plt.show()


# Now we construct an approximation for the product of two d-regular trees.
# This is done by gluing, on an n x n grid, random d-regular graphs along rows and columns.
# The linings now give a 4-regular subgraph. How close is it to a tree?
def product_of_trees(n,d):
    A_list = []
    L_list = []
    for _ in range(2*n):
        A, L = d_regular(n, d)
        A_list.append(A.tocoo())
        L_list.append(L.tocoo())

    edge_in_A = d*n
    edge_in_L = edge_in_A//2
    Astep = 2 * edge_in_A #number of edges treated in each iteration
    Lstep = 2 * edge_in_L #number of edges treated in each iteration
    edge_number_A = Astep * n #total number of edges
    edge_number_L = Lstep * n #total number of edges
    Arow = np.zeros(edge_number_A)
    Acol = np.zeros(edge_number_A)
    Lrow = np.zeros(edge_number_L)
    Lcol = np.zeros(edge_number_L)
    for i in range(n):
        Arow[i*Astep : i*Astep + edge_in_A], Lrow[i*Lstep : i*Lstep + edge_in_L] = i*n + A_list[i].row, i*n + L_list[i].row
        Acol[i*Astep : i*Astep + edge_in_A], Lcol[i*Lstep : i*Lstep + edge_in_L] = i*n + A_list[i].col, i*n + L_list[i].col
         
        Arow[i*Astep + edge_in_A : (i+1)*Astep], Lrow[i*Lstep + edge_in_L : (i+1)*Lstep] = A_list[i+1].row*n + i, L_list[i+1].row*n + i
        Acol[i*Astep + edge_in_A : (i+1)*Astep], Lcol[i*Lstep + edge_in_L : (i+1)*Lstep] = A_list[i+1].col*n + i, L_list[i+1].col*n + i
    A = sp.coo_array((np.ones_like(Arow), (Arow, Acol)), shape=(n**2,n**2))
    L = sp.coo_array((np.ones_like(Lrow), (Lrow, Lcol)), shape=(n**2,n**2))
    return A,L 

# Plots a ball of the lining in the product of trees.
def plot_lining_ball_prod(r,n,d):
    A,L = product_of_trees(n,d)
    L = L.tocsr() 
    B = L + sp.identity(L.shape[0], format=L.format)
    B_power = sparse_matrix_power(B,r) #compute power
    root = np.random.randint(0,B.shape[0] - 1) #choose random root
    vertices = B_power.indices[B_power.indptr[root]:B_power.indptr[root + 1]] #get vertices in ball
    adj = L[vertices,:][:,vertices] #induced adjacency matrix
    G = nx.from_scipy_sparse_array(adj)
    pos = nx.spring_layout(G)
    nx.draw(G,pos,node_color='blue',edge_color='blue',with_labels=True)
    return plt.show() 

#The above could be improved by using sparse arrays and avoiding the for loop over n, vectorizing it.

# We are doing Hashimoto for L
def hashimoto(L):
    edges = np.stack((L.row, L.col), axis=1) # Gives edges as (number of edges,2)
    edges = np.unique(edges, axis=0) # Remove duplicates
    degree = np.bincount(edges[:,0])
    num_edges = np.sum(degree * (degree - 1))
    row , col = np.zeros(num_edges), np.zeros(num_edges)
    ptr = np.zeros((L.shape[0]+1), dtype=int)
    for i in range(L.shape[0]):
        ptr[i] = np.min(np.where(edges[:,0]==i)[0])
    ptr[L.shape[0]] = num_edges
    j=0
    for i in range(edges.shape[0]):
        successors = np.where((edges[:,0] == edges[i,1]) & (edges[:,1]!=edges[i,0]))[0]
        step = len(successors)
        row[j:j+step] = i
        col[j:j+step] = successors
        j += step
    H = sp.coo_array((np.ones_like(row), (row, col)), shape=(edges.shape[0], edges.shape[0])).tocsr()
    return H,ptr
# CHECK HERE WHAT HAPPENS WITH THE J AND WHY IT GIVES ERROR BY THE END

def diagonal_pooling(H,vx_ptr):
    n = vx_ptr.shape[0] - 1
    vx_ptr = vx_ptr.astype(int)
    diag = H[np.arange(H.shape[0]), np.arange(H.shape[0])]
    out = np.zeros((n,))
    for i in range(n):
        out[i] = np.sum(diag[vx_ptr[i]:vx_ptr[i+1]])
    return out

def return_prob_tree(n,d,r):
    A = d_regular(n,d)[0]
    A = A.tocsr()
    B = sparse_matrix_power(A/d,r)
    E = np.mean(B.diagonal()) ** (1/r)
    return E






