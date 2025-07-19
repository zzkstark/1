import torch
import faiss
import numpy as np
from collections import defaultdict
from time import perf_counter
import scipy.sparse as sp
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from torch.nn.functional import normalize


def build_affinity_matrix(X, k, device='cuda'):
    X = torch.tensor(X).float().to(device)
    index = faiss.IndexFlatL2(X.shape[1])  
    gpu_res = faiss.StandardGpuResources()  
    gpu_index = faiss.index_cpu_to_gpu(gpu_res, 0, index) 
    gpu_index.add(X.cpu().numpy()) 
    _, ind = gpu_index.search(X.cpu().numpy(), k + 1) 
    dist = np.array([np.linalg.norm(X[i].cpu().numpy() - X[ind[i][1:]].cpu().numpy(), axis=1) for i in range(X.shape[0])])
    dist = torch.tensor(dist, device=device)  
    aff = torch.exp(-dist ** 2 / 2)  
    W = torch.zeros(X.shape[0], X.shape[0], device=device) 
    for i in range(X.shape[0]):  
        neighbors = ind[i][1:k + 1]  
        W[i, neighbors] = aff[i]  
        W[neighbors, i] = aff[i] 
    adj = W.cpu().numpy()
    normalization = 'NormAdj'  
    adj_normalizer = fetch_normalization(normalization) 
    adj = adj_normalizer(adj)  
    adj = sparse_mx_to_torch_sparse_tensor(adj).float().to(device)  
    ind = torch.from_numpy(ind).to(device)  
    return adj 

def graph_fusion(graphs,num_iters=20, tol=1e-6,device='cuda'):
    A = sum(graphs) / len(graphs)  
    V = len(graphs)  
    N = graphs[0].shape[0]  

    for iteration in range(num_iters):
        previous_A = A.clone()
        weights = []
        for v in range(V):
            dist = torch.norm(A.to_dense() - graphs[v], p='fro')
            weight = 1.0 / (dist + 1e-8)
            weights.append(weight)
        weights = torch.tensor(weights, dtype=torch.float32, device=A.device)  
        weights = weights / weights.sum()
        A = torch.zeros((N, N), dtype=torch.float32, device=A.device) 
        for v in range(V):
            A += weights[v] * graphs[v]
        degree = A.to_dense().sum(dim=1)  
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(degree + 1e-8))  
        A = D_inv_sqrt @ A.to_dense() @ D_inv_sqrt  
        if torch.norm(A - previous_A, p='fro') < tol:
            break
    indices = torch.stack(A.nonzero(as_tuple=True), dim=0)  
    values = A[indices[0], indices[1]]  
    A = torch.sparse_coo_tensor(indices, values, size=(N, N), device=A.device)
    return A, weights

def normalized_adjacency(adj):       
   adj = adj                        
   adj = sp.coo_matrix(adj)         
   row_sum = np.array(adj.sum(1))   
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()  
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.          
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)           
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

def aug_normalized_adjacency(adj):      
   adj = adj + sp.eye(adj.shape[0])   
   adj = sp.coo_matrix(adj)            
   row_sum = np.array(adj.sum(1))     
   d_inv_sqrt = np.power(row_sum, -0.5).flatten() 
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.          
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)          
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()  

def fetch_normalization(type):  
   switcher = {
       'AugNormAdj': aug_normalized_adjacency,  
       'NormAdj': normalized_adjacency,  
   }
   func = switcher.get(type, lambda: "Invalid normalization technique.")  
   return func  

def sparse_mx_to_torch_sparse_tensor(sparse_mx):      
    sparse_mx = sparse_mx.tocoo().astype(np.float32)  
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))  
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)      
    return torch.sparse.FloatTensor(indices, values, shape)        