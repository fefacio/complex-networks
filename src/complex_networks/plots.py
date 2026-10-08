from pathlib import Path

import matplotlib.pyplot as plt


def plot_degree_distribution(degree_counts, output_path, *, title="", log_scale=False):
    """Plota a fração de vértices de cada grau; omite grau zero no log-log."""
    n = sum(degree_counts.values())
    if n == 0:
        raise ValueError("A distribuição de graus está vazia.")
    degrees = sorted(k for k in degree_counts if not log_scale or k > 0)
    probabilities = [degree_counts[k] / n for k in degrees]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(degrees, probabilities, s=16, alpha=0.8)
    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlabel("Grau k")
    ax.set_ylabel("P(k)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_clustering_distribution(coefficients, output_path, *, title="", bins=20):
    """Plota a frequência dos coeficientes recebidos, em intervalos de [0, 1]."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(coefficients, bins=bins, range=(0, 1), edgecolor="white")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Coeficiente de clusterização local")
    ax.set_ylabel("Número de vértices")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_component_size_distribution(
    component_counts, output_path, *, title="", log_scale=False
):
    """Plota a fração de componentes de cada tamanho."""
    total = sum(component_counts.values())
    if total == 0:
        raise ValueError("A distribuição de componentes está vazia.")
    sizes = sorted(component_counts)
    probabilities = [component_counts[size] / total for size in sizes]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(sizes, probabilities, s=16, alpha=0.8)
    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlabel("Tamanho da componente (vértices)")
    ax.set_ylabel("Fração das componentes P(s)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
