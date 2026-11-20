"""Guloso de cobertura para o conjunto dominante mínimo sem pesos.

https://networkx.org/documentation/stable/_modules/networkx/algorithms/approximation/dominating_set.html
"""

from __future__ import annotations

from collections.abc import Callable, Hashable
from heapq import heapify, heappop, heappush
from itertools import chain
from math import isfinite
from time import perf_counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from networkx import Graph


class _PrazoExpirado(Exception):
    pass


def _verificar_prazo(prazo: float | None) -> None:
    if prazo is not None and perf_counter() >= prazo:
        raise _PrazoExpirado


def _construir(G: Graph, prazo: float | None) -> set[Hashable]:
    _verificar_prazo(prazo)
    vertices = tuple(G)
    posicoes = {}
    ganhos = []
    fila = []
    for posicao, vertice in enumerate(vertices):
        if posicao % 512 == 0:
            _verificar_prazo(prazo)
        vizinhos = G[vertice]
        if vertice in vizinhos:
            raise ValueError("Remova os laços do grafo antes de executar o Guloso.")
        ganho = len(vizinhos) + 1
        posicoes[vertice] = posicao
        ganhos.append(ganho)
        fila.append((-ganho, posicao))
    heapify(fila)

    descobertos = bytearray(b"\x01") * len(vertices)
    restantes = len(vertices)
    dominantes = set()
    while restantes:
        _verificar_prazo(prazo)
        prioridade, posicao = heappop(fila)
        ganho = ganhos[posicao]
        # As prioridades só diminuem. Reavalie entradas antigas ao removê-las.
        if -prioridade != ganho:
            if ganho:
                heappush(fila, (-ganho, posicao))
            continue

        vertice = vertices[posicao]
        dominantes.add(vertice)
        for novo in chain((vertice,), G[vertice]):
            indice = posicoes[novo]
            if not descobertos[indice]:
                continue
            descobertos[indice] = 0
            restantes -= 1
            ganhos[indice] -= 1
            for vizinho in G[novo]:
                ganhos[posicoes[vizinho]] -= 1

    return dominantes


def mds_solver_greedy(
    G: Graph,
    *,
    seed: int = 1,
    time_limit: float | None = None,
    on_improvement: Callable[[int], None] | None = None,
) -> set[Hashable]:
    """Seleciona o vértice cuja vizinhança fechada cobre mais descobertos.

    G deve ser simples e não direcionado. A ordem dos vértices em G desempata
    ganhos iguais. O parâmetro seed é aceito para a interface do benchmark.

    O relógio inclui a preparação interna. Quando o prazo expira antes da
    solução gulosa, retorna o conjunto dominante trivial com todos os vértices.
    on_improvement recebe a cardinalidade quando a solução está disponível.
    Os conjuntos parciais da construção ainda não são conjuntos dominantes.
    """
    if G.is_directed() or G.is_multigraph():
        raise ValueError("Guloso requer um grafo simples e não direcionado.")

    prazo = None
    if time_limit is not None:
        time_limit = float(time_limit)
        if not isfinite(time_limit) or time_limit < 0:
            raise ValueError("time_limit deve ser finito e não negativo.")
        prazo = perf_counter() + time_limit

    try:
        solucao = _construir(G, prazo)
    except _PrazoExpirado:
        solucao = set(G)
    if on_improvement is not None:
        on_improvement(len(solucao))
    return solucao
