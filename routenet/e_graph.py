import networkx as nx
import numpy as np
import pandas as pd


def nodeGraph_to_edgeGraph(graph,support=False):
    num_edges=graph.number_of_edges()
    edges_attr = {}
    idx = 0
    tran_dict = {'core layer': 1, 'converge layer': 2, 'access layer': 3}
    for u, v in graph.edges:
        attr = graph.get_edge_data(u, v)
        edges_attr[idx] = {'u': u, 'v': v, 'layer': tran_dict[attr['layer']], 'bandwidth': attr['bandwidth']}
        idx += 1
    edges_df = pd.DataFrame.from_dict(edges_attr)

    adj = np.zeros([num_edges, num_edges], dtype=np.int)
    for i in range(num_edges - 1):
        for j in range(i + 1, num_edges):
            if edges_attr[i]['u'] == edges_attr[i]['u'] \
                    or edges_attr[i]['v'] == edges_attr[i]['v'] \
                    or edges_attr[i]['u'] == edges_attr[i]['v'] \
                    or edges_attr[i]['v'] == edges_attr[i]['u']:
                adj[i, j] = 1
                adj[j, i] = 1
    sup=adj
    if support:
        edge_graph = nx.from_numpy_matrix(adj)
        sup=nx.normalized_laplacian_matrix(edge_graph).todense()
    f = edges_df.loc['bandwidth']
    f = f.to_numpy()

    return sup, f