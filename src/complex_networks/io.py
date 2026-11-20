from pathlib import Path

import networkx as nx

from src.complex_networks.config import RAW_DATA_DIR


def get_network_path(name: str) -> Path:
    return RAW_DATA_DIR / f"{name}.txt.gz"


def load_edgelist(path, directed=False):
    G = nx.read_edgelist(
        path,
        comments="#",
        nodetype=int,
        data=False,
        create_using=nx.DiGraph if directed else nx.Graph,
    )

    G.remove_edges_from(list(nx.selfloop_edges(G)))
    G.remove_nodes_from(list(nx.isolates(G)))

    return G


def preprocess(G: nx.Graph) -> nx.Graph:
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    return G


