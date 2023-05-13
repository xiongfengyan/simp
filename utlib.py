import numpy as np
import matplotlib.pyplot as plt
import math

import geopandas as gpd
from geopandas import GeoSeries
from shapely.geometry import LineString
import scipy.sparse as sp

from scipy.sparse.linalg.eigen.arpack import eigsh
import networkx as nx


seed = 111
np.random.seed(seed)

def input_data(path, layers, num_supports):
    df_shape = gpd.read_file(path, encode='utf-8')
    lst = df_shape['geometry']
    line_x = []
    line_y = []
    num = lst.size
    min_x_list = []
    max_x_list = []
    min_y_list = []
    max_y_list = []

    features = []
    all_supports, all_marks = [], []

    start_x_list = []
    start_y_list = []
    end_x_list = []
    end_y_list = []

    pad = []
    in_shp = []
    for i in range(num):
        print('---------')
        x_train_1 = []
        y_train_1 = []
        line = list(lst.iloc[i].xy)
        start_x = line[0][0]
        start_y = line[1][0]
        end_x = line[0][-1]
        end_y = line[1][-1]

        start_x_list.append(start_x)
        start_y_list.append(start_y)
        end_x_list.append(end_x)
        end_y_list.append(end_y)

        x = np.array(line[0], dtype='float32')
        y = np.array(line[1], dtype='float32')

        x = np.reshape(x, (x.shape[0], 1))
        y = np.reshape(y, (y.shape[0], 1))

        min_x = x.min()
        min_y = y.min()
        max_x = x.max()
        max_y = y.max()
        min_x_list.append(min_x)
        min_y_list.append(min_y)
        max_x_list.append(max_x)
        max_y_list.append(max_y)
        line_x.append(x)
        line_y.append(y)

        x = (x-min_x)/(max_x-min_x)
        y = (y-min_y)/(max_y-min_y)

        shp_x = x.reshape(-1)
        shp_y = y.reshape(-1)
        line_t = LineString(list(zip(shp_x, shp_y)))
        in_shp.append(line_t)
        f = np.stack((shp_x, shp_y), axis=1)

        features.append(f)
        #print(f, 'f')

        supports, marks, l = [], [], x.shape[0]
        for i in range(layers):
            if i < 1:   # building 3     landuse 1
                l_t = math.ceil(l/4) if i > 0 else l
            else:
                l_t = math.ceil(l/2) if i > 0 else l

            if not l_t < 8:     # building 24
                l = l_t
                marks.append([1, l, 1 if i == 0 else (4 if i < 1 else 2)])    # building 3    landuse 1
            else:
                marks.append([0, l, 1])
            print(l, l_t)

            # p1 = list(range(int(x.shape[0]/(math.pow(2,i)))-1))
            # p2 = list(range(1, int(x.shape[0]/(math.pow(2,i)))))
            p1 = list(range(l-1))
            p2 = list(range(1,l))
            #print(l, p1, p2)
            edge = list(zip(p1, p2))
            G = nx.from_edgelist(edge)
            adj = nx.adjacency_matrix(G).todense()
            support = chebyshev_polynomials(adj, num_supports)
            #print(np.array(support[0]).shape)
            #print(np.array(support[0]))
            supports.append(support)

        all_supports.append(supports)
        all_marks.append(marks)

    #g = GeoSeries(in_shp)
    #g.to_file('../data/contour/in_c_g.shp')

    for j in range(len(line_x)):
        plt.plot(line_x[j], line_y[j], 'y')

    return features, all_supports, all_marks, max_x_list, max_y_list, min_x_list, min_y_list,start_x_list,start_y_list,end_x_list,\
           end_y_list,pad

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1), dtype='float32')
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def chebyshev_polynomials(adj, k):
    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2./ largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)
