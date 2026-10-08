import networkx as nx

def load_edgelist(path, directed=False):
    return nx.read_edgelist(
        path,
        comments="#",
        nodetype=int,
        data=False,
        create_using=nx.DiGraph if directed else nx.Graph,
    )


def preprocess(G: nx.Graph) -> nx.Graph:
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    return G