import csv
from collections import Counter
from pathlib import Path

import numpy as np
import networkx as nx
import time

from math import ceil, isfinite
from operator import index
from random import Random

from networkit.community import Modularity, PLM
from networkit.graph import GraphFromCoo
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from networkx.algorithms.approximation import diameter as approximate_diameter

def _append_csv(path, row):
    """Acrescenta uma linha; escreve o cabeçalho se o arquivo estiver vazio."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0

    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def get_degree_distribution(G):
    """Retorna {grau: número de vértices}."""
    return Counter(degree for _, degree in G.degree())


def get_clustering_distribution(G):
    """Retorna coeficientes locais dos nós com grau > 1, incluindo zeros."""
    all_cc = nx.clustering(G)
    return [all_cc[v] for v, degree in G.degree() if degree > 1]


def get_component_size_distribution(G):
    """Retorna {tamanho: número de componentes}."""
    return Counter(len(component) for component in nx.connected_components(G))





def get_modularity(G):
    """Retorna Q pelo Louvain paralelo para um grafo simples e sem pesos."""
    m = G.number_of_edges()
    if m == 0:
        return None

    n = G.number_of_nodes()
    node_ids = dict(zip(G, range(n)))
    edges = np.fromiter(
        (node_ids[node] for edge in G.edges for node in edge),
        dtype=np.uint64,
        count=2 * m,
    ).reshape(m, 2)

    graph = GraphFromCoo((edges[:, 0], edges[:, 1]), n=n)
    del node_ids, edges

    partition = PLM(graph, maxIter=8).run().getPartition()
    return float(Modularity().getQuality(partition, graph))




def estimate_average_distance(G, percentage=1, seed=42, *, max_sources=32):
    """Estima a distância média em arestas de um grafo conexo e não direcionado."""
    if not 0 < percentage <= 100:
        raise ValueError("percentage deve estar entre 0 e 100.")

    if max_sources is not None:
        max_sources = index(max_sources)
        if max_sources < 1:
            raise ValueError("max_sources deve ser positivo.")

    if G.is_directed():
        raise nx.NetworkXNotImplemented("Use um grafo não direcionado.")

    n = G.number_of_nodes()
    if n == 0:
        raise nx.NetworkXPointlessConcept("O grafo está vazio.")
    if n == 1:
        return 0.0

    sample_size = ceil(n * percentage / 100)
    if max_sources is not None:
        sample_size = min(sample_size, max_sources)
    sources = Random(seed).sample(range(n), sample_size)

    adjacency = nx.to_scipy_sparse_array(G, dtype=float, weight=None, format="csr")
    # Usa índices de 32 bits quando cabem, compatíveis com versões antigas do SciPy.
    adjacency = csr_matrix(
        (adjacency.data, adjacency.indices, adjacency.indptr), shape=adjacency.shape
    )

    total_distance = 0.0
    for source in sources:
        distances = dijkstra(adjacency, indices=source, unweighted=True)
        distance_sum = float(distances.sum())
        if not isfinite(distance_sum):
            raise ValueError("Use um grafo conexo, como o maior componente.")
        total_distance += distance_sum

    return total_distance / (sample_size * (n - 1))



def calculate_general_metrics(G,name,results_dir,estimate_distance=False,estimate_diameter=False,):
    """Calcula métricas e acrescenta uma linha em general_metrics.csv.

    Recebe um grafo simples, não direcionado, sem laços e não vazio.
    Apenas distância média e diâmetro usam o maior componente conexo.
    A distância média pode ser exata ou estimada; o diâmetro é exato.
    Retorna a linha escrita. Não salva distribuições nem gera figuras.
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()
    if n == 0:
        raise ValueError("O grafo não pode ser vazio.")

    print("->calculando cc")
    filtered_cc = get_clustering_distribution(G)
    eligible_count = len(filtered_cc)
    mean_cc = sum(filtered_cc) / eligible_count if eligible_count else None
    del filtered_cc

    transitivity = nx.transitivity(G)

    print("->calculando modularidade (Louvain)")
    modularity = get_modularity(G)

    print("->calculando componentes conexos")
    n_components = nx.number_connected_components(G)
    GS = G
    if n_components > 1:
        largest_component = max(nx.connected_components(G), key=len)
        GS = G.subgraph(largest_component).copy()

    mode = "estimada" if estimate_distance else "exata"
    print(f"->calculando distância média ({mode})")
    start = time.perf_counter()

    if estimate_distance:
        average_distance = estimate_average_distance(
            GS,
            percentage=1,
            seed=24,
        )
    else:
        average_distance = nx.average_shortest_path_length(GS)

    elapsed_minutes = (time.perf_counter() - start) / 60
    print(f"->concluído em {elapsed_minutes:.2f} minutos")


    mode = "estimado" if estimate_diameter else "exato"
    print(f"->calculando diâmetro ({mode})")
    start = time.perf_counter()

    if estimate_diameter:
        diameter = approximate_diameter(GS, seed=42)
    else:
        diameter = nx.diameter(GS, usebounds=True)

    elapsed_minutes = (time.perf_counter() - start) / 60
    print(f"->concluído em {elapsed_minutes:.2f} minutos")

    row = {
        "network": name,
        "n_nodes": n,
        "n_edges": m,
        "average_degree": 2 * m / n,
        "mean_clustering_degree_gt_1": mean_cc,
        "clustering_eligible_nodes": eligible_count,
        "transitivity": transitivity,
        "average_shortest_path_length": average_distance,
        "diameter": diameter,
        "density": nx.density(G),
        "n_components": n_components,
        "modularity": modularity,
    }
    _append_csv(Path(results_dir) / "general_metrics.csv", row)
    return row

