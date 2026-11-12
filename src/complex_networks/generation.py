from pathlib import Path

import networkx as nx
import numpy as np

def generate_degree_graph(
    distribution,
    n=5_000,
    *,
    kmin=1,
    kmax=100,
    seed=42,
    max_attempts=1_000,
    swaps_per_edge=5,
    **parameters,
):
    """Retorna um nx.Graph e preserva os graus sorteados em 'sampled_degree'.

    O suporte deve satisfazer 1 <= kmin <= kmax < n. Isolados são excluídos.
    Para pular a etapa de trocas, use swaps_per_edge=0.
    """
    if not isinstance(n, int) or n < 2:
        raise ValueError("n deve ser um inteiro >= 2.")
    if not isinstance(kmin, int) or not isinstance(kmax, int):
        raise ValueError("kmin e kmax devem ser inteiros.")
    if not 1 <= kmin <= kmax < n:
        raise ValueError("Use 1 <= kmin <= kmax < n.")
    if not isinstance(max_attempts, int) or max_attempts < 1:
        raise ValueError("max_attempts deve ser um inteiro positivo.")
    if not isinstance(swaps_per_edge, int) or swaps_per_edge < 0:
        raise ValueError("swaps_per_edge deve ser um inteiro >= 0.")

    required = {
        "exponential": {"Lambda"},
        "lognormal": {"mu", "sigma"},
        "stretched_exponential": {"Lambda", "beta"},
        "truncated_power_law": {"alpha", "Lambda"},
        "power_law": {"alpha"},
    }
    if distribution not in required:
        raise ValueError(f"Distribuição desconhecida: {distribution}")
    if set(parameters) != required[distribution]:
        raise ValueError(f"Parâmetros esperados: {sorted(required[distribution])}")
    if not all(np.isfinite(value) for value in parameters.values()):
        raise ValueError("Os parâmetros devem ser finitos.")
    for name, value in parameters.items():
        if name != "mu" and value <= 0:
            raise ValueError(f"{name} deve ser positivo.")

    rng = np.random.default_rng(seed)
    support = np.arange(kmin, kmax + 1)
    k = support.astype(float)
    p = parameters

    # Log-pesos: evita underflow ao normalizar probabilidades pequenas.
    if distribution == "exponential":
        log_weights = -p["Lambda"] * k
    elif distribution == "lognormal":
        log_weights = -np.log(k) - (np.log(k) - p["mu"]) ** 2 / (2 * p["sigma"] ** 2)
    elif distribution == "stretched_exponential":
        log_weights = (p["beta"] - 1) * np.log(k) - (p["Lambda"] * k) ** p["beta"]
    elif distribution == "truncated_power_law":
        log_weights = -p["alpha"] * np.log(k) - p["Lambda"] * k
    else:
        log_weights = -p["alpha"] * np.log(k)

    if not np.all(np.isfinite(log_weights)):
        raise ValueError("Parâmetros extremos: não foi possível calcular as probabilidades.")
    weights = np.exp(log_weights - log_weights.max())
    probabilities = weights / weights.sum()

    for _ in range(max_attempts):
        degrees = rng.choice(support, size=n, p=probabilities).tolist()
        if sum(degrees) % 2 == 0 and nx.is_graphical(degrees, method="eg"):
            break
    else:
        raise RuntimeError("Não foi encontrada uma sequência gráfica. Reveja n e o suporte.")

    G = nx.havel_hakimi_graph(degrees)
    nx.set_node_attributes(G, dict(enumerate(degrees)), "sampled_degree")

    nswap = swaps_per_edge * G.number_of_edges()
    if nswap and n >= 4 and G.number_of_edges() >= 2:
        try:
            nx.double_edge_swap(
                G,
                nswap=nswap,
                max_tries=100 * nswap,
                seed=int(rng.integers(0, 2**32)),
            )
        except nx.NetworkXAlgorithmError as error:
            raise RuntimeError(
                "As trocas não foram concluídas. A sequência pode admitir poucas "
                "trocas; use swaps_per_edge=0 para obter apenas Havel-Hakimi."
            ) from error

    G.graph.update(distribution=distribution, kmin=kmin, kmax=kmax, **parameters)
    return G


def sample_graphs(n=5_000, kmax=100, seed=42, configurations=None):
    """Gera grafos usando as configurações fornecidas ou as padrão."""
    if configurations is None:
        configurations = {
            "exponential": {"Lambda": 0.15},
            "lognormal": {"mu": 1.5, "sigma": 0.8},
            "stretched_exponential": {"Lambda": 0.10, "beta": 0.7},
            "truncated_power_law": {"alpha": 1.8, "Lambda": 0.03},
        }

    graphs = {}

    for i, name in enumerate(configurations):
        parameters = configurations[name]

        G = generate_degree_graph(
            name,
            n=n,
            kmax=kmax,
            seed=seed + i,
            **parameters,
        )

        graphs[name] = G

    return graphs