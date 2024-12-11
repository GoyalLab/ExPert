import pandas as pd
from arboreto.algo import grnboost2

import networkx as nx
import graph_tool.all as gt
import numpy as np
import logging


G_PROPERTIES = ['degree', 'btwn', 'eigenvalue', 'pagerank', 'closeness', 'clustering']


def get_degrees(G: gt.Graph):
    logging.debug('Computung node degree')
    degrees = G.degree_property_map("total").a
    logging.debug('Finished computung node degree')
    return degrees

def get_betweenness(G: gt.Graph):
    logging.debug('Computing node betweenness')
    # Calculate betweenness for all nodes
    vertex_betweenness, _ = gt.betweenness(G)
    # Convert the vertex_betweenness PropertyMap to a numpy array
    betweenness_values = np.array([vertex_betweenness[v] for v in G.vertices()])
    logging.debug('Finished computing node betweenness')
    return betweenness_values

def is_hub(metric):
    m = np.mean(metric)
    std = np.std(metric)

    t = m + 2 * std
    return metric >= t

def scale(X):
    mean = np.mean(X)
    std = np.std(X)
    X_scaled = (X - mean) / std
    return X_scaled

def summarize_graph(G: gt.Graph) -> pd.DataFrame:
    # extract node information of graph
    node_names = [G.vp.name[v] for v in G.vertices()]
    # extract node degree in network
    degree = get_degrees(G)
    # extract node betweenness in network
    btwn = get_betweenness(G)
    # extract eigenvector
    eigenvalue, eigenvector = gt.eigenvector(G)
    # extract PageRank
    pagerank = gt.pagerank(G)
    # extract closeness
    closeness = gt.closeness(G)
    # extract clustering coefficient
    clustering = gt.local_clustering(G)
    # return collected information as dataframe
    return pd.DataFrame({'degree': degree, 'degree_scaled': scale(degree),
                         'btwn': btwn, 'btwn_scaled': scale(btwn),
                         'eigenvalue': eigenvalue, 'ev_scaled': scale(eigenvalue),
                         'pagerank': pagerank, 'pagerank_scaled': scale(pagerank),
                         'closeness': closeness, 'closeness_scaled': scale(closeness),
                         'clustering': clustering, 'clustering_scaled': scale(clustering)
                          },index=node_names)

def parse_grn(grn_df: pd.DataFrame, source: str ='TF', target: str ='target', edge_attr: str ='importance'):

    # Initialize an empty directed graph
    G = gt.Graph()

    # Add vertex property maps for vertex names and edge property maps for coefficients
    vprop_name = G.new_vertex_property("string")
    eprop = G.new_edge_property("double")

    # Keep track of vertices by names to avoid duplicates
    vertices = {}

    for _, row in grn_df.iterrows():
        src_name = row[source]
        tgt_name = row[target]

        # Add source vertex if it doesn't exist
        if src_name not in vertices:
            src_v = G.add_vertex()
            vprop_name[src_v] = src_name
            vertices[src_name] = src_v
        else:
            src_v = vertices[src_name]

        # Add target vertex if it doesn't exist
        if tgt_name not in vertices:
            tgt_v = G.add_vertex()
            vprop_name[tgt_v] = tgt_name
            vertices[tgt_name] = tgt_v
        else:
            tgt_v = vertices[tgt_name]

        # Add edge with the coefficient as an attribute
        e = G.add_edge(src_v, tgt_v)
        eprop[e] = row[edge_attr]

    # Assign the property maps to the graph
    G.vertex_properties["name"] = vprop_name
    G.edge_properties[edge_attr] = eprop

    return G