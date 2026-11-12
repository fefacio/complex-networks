from typing import Any

import networkx as nx
import numpy as np
import powerlaw


MODELS = (
    "power_law",
    "exponential",
    "lognormal",
    "stretched_exponential",
    "truncated_power_law",
)

DISTANCE_METRICS = {
    "ks": "D",
    "kuiper": "V",
    "anderson_darling": "Asquare",
    "kappa": "Kappa",
}


def fit_degree_distributions(G: nx.Graph, xmin: int | None = 1) -> dict[str, Any]:
    """Ajusta cinco modelos e coleta resultados, sem ordenar ou selecionar.

    xmin=1 usa todos os graus positivos; outro inteiro restringe o intervalo.
    xmin=None escolhe um corte comum pela distância KS da lei de potência.
    Requer grafo simples, não direcionado, sem isolados e com graus variados.
    noise_flag pode sinalizar falha numérica ou proximidade dos limites; não
    significa, isoladamente, que o ajuste deve ser descartado.
    """
    if G.is_directed() or G.is_multigraph() or nx.number_of_selfloops(G):
        raise ValueError("O grafo deve ser simples, não direcionado e sem self-loops.")
    if G.number_of_nodes() == 0:
        raise ValueError("O grafo está vazio.")
    if xmin is not None and (not isinstance(xmin, (int, np.integer)) or xmin < 1):
        raise ValueError("xmin deve ser um inteiro positivo ou None.")

    degrees = np.fromiter((d for _, d in G.degree()), dtype=np.int64)
    if np.any(degrees == 0):
        raise ValueError("O grafo possui vértices isolados.")
    if np.unique(degrees).size < 2:
        raise ValueError("Todos os graus são iguais; ajustes podem ser degenerados.")

    fit = powerlaw.Fit(
        degrees, 
        xmin=xmin, 
        discrete=True, 
        estimate_discrete=False,
        fit_method="likelihood", 
        discrete_normalization="round",
        parameter_ranges={"alpha": (1.000001, None)}, 
        verbose=0,
    )
    if fit.data.size < 2 or np.unique(fit.data).size < 2:
        raise ValueError("O intervalo selecionado precisa conter graus distintos.")

    models = []
    for name in MODELS:
        # Fit compartilha os limites; cada modelo é ajustado ao ser acessado.
        # O cutoff admite alpha < 1; a potência pura sem xmax exige alpha > 1.
        fit.parameter_ranges["alpha"] = (
            0.0 if name == "truncated_power_law" else 1.000001, None
        )
        distribution = getattr(fit, name)
        parameters = {k: float(v) for k, v in distribution.parameters.items()}
        log_likelihood = float(distribution.loglikelihood)
        models.append({
            "model": name,
            "parameters": parameters,
            "parameter_ranges": dict(distribution.parameter_ranges),
            "n_parameters": len(parameters),
            "log_likelihood": log_likelihood,
            "aic": 2 * len(parameters) - 2 * log_likelihood,
            **{key: float(getattr(distribution, attr))
               for key, attr in DISTANCE_METRICS.items()},
            "in_range": bool(distribution.in_range()),
            "noise_flag": bool(distribution.noise_flag),
        })

    return {
        "powerlaw_version": powerlaw.__version__,
        "xmin_requested": None if xmin is None else int(xmin),
        "xmin": int(fit.xmin),
        "n_nodes": int(degrees.size),
        "n_fitted": int(fit.data.size),
        "fitted_percentage": 100.0 * fit.data.size / degrees.size,
        "discrete": True,
        "discrete_normalization": "round",
        "fit_method": "likelihood",
        "distance_metrics_source": "powerlaw.compute_distance_metrics",
        "models": models,
    }
