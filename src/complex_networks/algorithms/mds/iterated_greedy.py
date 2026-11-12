"""Iterated Greedy para o conjunto dominante mínimo sem pesos.

Referência: Casado et al. (2023), https://doi.org/10.1016/j.matcom.2022.12.018
Implementação Java: https://github.com/AlejandraCasado/MinimumDominatingSet

Porta a configuração GIP + IG_RandomGreedy + LocalSearchEfficient_1_1.
Os empates usam os IDs internos em ordem crescente, definidos pela ordem
dos vértices em G. A aleatoriedade segue java.util.Random.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Iterator
from heapq import heapify, heappop, heappush
from itertools import chain, count
from math import ceil, isfinite
from operator import index
from struct import Struct
from time import perf_counter
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from networkx import Graph


_FLOAT32 = Struct("!f")
_DEFAULT_TIME_LIMIT_S = 600.0
_MAX_STAGNANT_ROUNDS = 200
_DEADLINE_CHECK_INTERVAL = 512
_Item = TypeVar("_Item")


class _DeadlineExpired(Exception):
    """Encerra a etapa em andamento num ponto de verificação."""


def _check_deadline(deadline: float | None) -> None:
    if deadline is not None and perf_counter() >= deadline:
        raise _DeadlineExpired


def _checked(items: Iterable[_Item], deadline: float | None) -> Iterator[_Item]:
    if deadline is None:
        yield from items
        return
    for position, item in enumerate(items):
        if position % _DEADLINE_CHECK_INTERVAL == 0:
            _check_deadline(deadline)
        yield item


def _float32(value: float) -> float:
    return _FLOAT32.unpack(_FLOAT32.pack(value))[0]


class _JavaRandom:
    """Gerador e embaralhamento usados por Random e Collections.shuffle."""

    _MULTIPLIER = 0x5DEECE66D
    _MASK = (1 << 48) - 1

    def __init__(self, seed: int) -> None:
        self._state = (seed ^ self._MULTIPLIER) & self._MASK

    def _next31(self) -> int:
        self._state = (self._state * self._MULTIPLIER + 11) & self._MASK
        return self._state >> 17

    def _next_int(self, bound: int) -> int:
        if bound & (bound - 1) == 0:
            return (bound * self._next31()) >> 31
        while True:
            bits = self._next31()
            value = bits % bound
            if bits - value + bound - 1 < 1 << 31:
                return value

    def shuffle(self, vertices: list[int], deadline: float | None) -> None:
        for size in _checked(range(len(vertices), 1, -1), deadline):
            other = self._next_int(size)
            vertices[size - 1], vertices[other] = vertices[other], vertices[size - 1]


class _IteratedGreedy:
    def __init__(
        self,
        adjacency: list[tuple[int, ...]],
        seed: int,
        on_improvement: Callable[[int], None] | None = None,
        *,
        deadline: float | None = None,
    ) -> None:
        self._adj = adjacency
        self._rng = _JavaRandom(seed)
        self._deadline = deadline
        size = len(adjacency)
        self._on_improvement = on_improvement
        self._reported_size = size + 1
        self._fixed: set[int] = set()
        leaves: set[int] = set()
        for vertex, neighbors in _checked(enumerate(adjacency), deadline):
            if len(neighbors) == 1:
                neighbor = neighbors[0]
                if neighbor not in leaves:
                    leaves.add(vertex)
                    self._fixed.add(neighbor)
            elif not neighbors:
                self._fixed.add(vertex)

        self._available = bytearray(b"\x01") * size
        for vertex in _checked(leaves | self._fixed, deadline):
            self._available[vertex] = 0

        self._movable: set[int] = set()
        self._uncovered = set(range(size))
        self._coverage = [0] * size
        # XOR dos dominadores identifica o dono quando a cobertura é 1.
        self._owners = [0] * size
        self._private = [0] * size
        self._redundant: set[int] = set()
        for vertex in _checked(sorted(self._fixed), deadline):
            self._add(vertex)

    def _report_improvement(self) -> None:
        if self._on_improvement is None:
            return
        size = len(self._fixed) + len(self._movable)
        if size < self._reported_size:
            self._reported_size = size
            self._on_improvement(size)

    def _add(self, vertex: int) -> list[int]:
        self._available[vertex] = 0
        if vertex not in self._fixed:
            self._movable.add(vertex)
            self._redundant.add(vertex)

        newly_covered = []
        for node in chain((vertex,), self._adj[vertex]):
            count = self._coverage[node]
            if count == 0:
                self._uncovered.remove(node)
                newly_covered.append(node)
                self._private[vertex] += 1
                self._redundant.discard(vertex)
            elif count == 1:
                owner = self._owners[node]
                self._private[owner] -= 1
                if self._private[owner] == 0 and owner in self._movable:
                    self._redundant.add(owner)
            self._coverage[node] = count + 1
            self._owners[node] ^= vertex
        return newly_covered

    def _remove(self, vertex: int) -> None:
        self._movable.remove(vertex)
        self._redundant.discard(vertex)
        self._available[vertex] = 1
        for node in chain((vertex,), self._adj[vertex]):
            self._owners[node] ^= vertex
            count = self._coverage[node] - 1
            self._coverage[node] = count
            if count == 0:
                self._uncovered.add(node)
                self._private[vertex] -= 1
            elif count == 1:
                owner = self._owners[node]
                self._private[owner] += 1
                self._redundant.discard(owner)

    def _construct(self) -> None:
        if not self._uncovered:
            return
        adjacency = self._adj
        available = self._available
        # O GIP do Java pontua vizinhos descobertos na vizinhança aberta.
        gain = [0] * len(adjacency)
        for node in _checked(self._uncovered, self._deadline):
            for neighbor in adjacency[node]:
                if available[neighbor]:
                    gain[neighbor] += 1
        queue = [
            (-gain[vertex], vertex)
            for vertex, free in _checked(enumerate(available), self._deadline)
            if free
        ]
        heapify(queue)
        for _ in _checked(count(), self._deadline):
            if not self._uncovered:
                return
            if not queue:
                raise RuntimeError("A reconstrução ficou sem vértices elegíveis.")
            priority, vertex = heappop(queue)
            if not available[vertex] or priority != -gain[vertex]:
                continue
            changed: set[int] = set()
            for node in self._add(vertex):
                for neighbor in adjacency[node]:
                    if available[neighbor]:
                        gain[neighbor] -= 1
                        changed.add(neighbor)
            for neighbor in changed:
                heappush(queue, (-gain[neighbor], neighbor))

    def _replacement(self, vertex: int) -> int:
        if not self._private[vertex]:
            return -1
        adjacency = self._adj
        critical = [
            node
            for node in chain((vertex,), adjacency[vertex])
            if self._coverage[node] == 1
        ]
        pivot = min(critical, key=lambda node: len(adjacency[node]))
        common = set(adjacency[pivot])
        common.discard(vertex)
        for node in _checked(critical, self._deadline):
            if node != pivot:
                common.intersection_update(adjacency[node])
                if not common:
                    return -1
        return min(common, default=-1)

    def _improve(self) -> None:
        while True:
            _check_deadline(self._deadline)
            order = sorted(self._movable)
            self._rng.shuffle(order, self._deadline)
            for vertex in _checked(order, self._deadline):
                replacement = self._replacement(vertex)
                if replacement == -1:
                    continue

                # A troca preserva a cobertura, inclusive quando não reduz o tamanho.
                self._add(replacement)
                self._remove(vertex)
                removed = False
                # Remover um dominador redundante só pode tornar os outros necessários.
                for redundant in _checked(sorted(self._redundant), self._deadline):
                    if self._private[redundant] == 0:
                        self._remove(redundant)
                        removed = True
                if removed:
                    self._report_improvement()
                    break
            else:
                return

    def search(self, max_iterations: int | None, beta: float) -> set[int]:
        best: set[int] | None = None
        try:
            self._construct()
            self._report_improvement()
            self._improve()
            best = self._movable.copy()
            # O Java calcula esta quantidade uma única vez, após a solução inicial.
            destroy_count = ceil(_float32(_float32(beta) * _float32(len(best))))
            rounds = count() if max_iterations is None else range(max_iterations)
            stagnant_rounds = 0
            for _ in rounds:
                _check_deadline(self._deadline)
                if not self._movable:
                    break
                order = sorted(self._movable)
                self._rng.shuffle(order, self._deadline)
                for vertex in _checked(order[:destroy_count], self._deadline):
                    self._remove(vertex)
                self._construct()
                self._report_improvement()
                self._improve()
                improved = len(self._movable) < len(best)
                if improved:
                    best = self._movable.copy()
                if self._deadline is not None:
                    stagnant_rounds = 0 if improved else stagnant_rounds + 1
                    if stagnant_rounds >= _MAX_STAGNANT_ROUNDS:
                        break
        except _DeadlineExpired:
            # A destruição e a reconstrução podem deixar vértices descobertos.
            # A busca local pode ter melhorado uma solução válida na rodada atual.
            if not self._uncovered and (best is None or len(self._movable) < len(best)):
                best = self._movable
                self._report_improvement()
            if best is None:
                raise
        return self._fixed | best


def mds_solver_iterated_greedy(
    G: Graph,
    *,
    seed: int = 13,
    max_iterations: int | None = None,
    beta: float = 0.2,
    time_limit: float | None = None,
    on_improvement: Callable[[int], None] | None = None,
) -> set[Hashable]:
    """Retorna o melhor conjunto dominante encontrado, com os rótulos de G.

    Recebe um grafo simples e não direcionado, inclusive com vértices isolados
    e componentes desconexas. Empates usam a ordem dos vértices de G.
    A destruição usa até o número de vértices móveis disponíveis na rodada.

    No modo temporal, time_limit mede segundos de tempo decorrido (600
    por padrão), incluindo a preparação interna. A busca checa o prazo
    periodicamente durante cada etapa e também para após 200 rodadas
    consecutivas sem melhoria. Se o prazo expirar antes da primeira
    solução, retorna todos os vértices de G como solução dominante.
    Para validação, max_iterations conta rodadas totais. Com zero,
    retorna a solução inicial após a busca local.

    on_improvement recebe a cardinalidade da primeira solução dominante e
    de cada melhora estrita encontrada depois. A função fornecida pelo
    chamador pode registrar o instante em que recebeu cada valor.
    """
    if G.is_directed() or G.is_multigraph():
        raise ValueError("Iterated Greedy requer um grafo simples e não direcionado.")
    seed = index(seed)
    if not -(1 << 63) <= seed < (1 << 63):
        raise ValueError("A semente deve caber em um long de Java.")
    if max_iterations is not None:
        if time_limit is not None:
            raise ValueError("Escolha max_iterations ou time_limit.")
        max_iterations = index(max_iterations)
        if not 0 <= max_iterations < (1 << 31):
            raise ValueError("max_iterations deve ser um inteiro entre 0 e 2**31 - 1.")
    else:
        time_limit = _DEFAULT_TIME_LIMIT_S if time_limit is None else float(time_limit)
        if not isfinite(time_limit) or time_limit < 0:
            raise ValueError("time_limit deve ser finito e não negativo.")
    beta = float(beta)
    if not 0 <= beta <= 1:
        raise ValueError("beta deve estar entre 0 e 1.")

    deadline = None if time_limit is None else perf_counter() + time_limit
    try:
        _check_deadline(deadline)
        labels = tuple(G)
        positions = {
            vertex: pos for pos, vertex in _checked(enumerate(labels), deadline)
        }
        adjacency = []
        for vertex in _checked(labels, deadline):
            neighbors = G[vertex]
            if vertex in neighbors:
                raise ValueError(
                    "Remova os laços do grafo antes de executar o Iterated Greedy."
                )
            adjacency.append(tuple(sorted(positions[neighbor] for neighbor in neighbors)))
        del positions

        state = _IteratedGreedy(adjacency, seed, on_improvement, deadline=deadline)
        solution = state.search(max_iterations, beta)
        return {labels[vertex] for vertex in solution}
    except _DeadlineExpired:
        fallback = set(G)
        if on_improvement is not None:
            on_improvement(len(fallback))
        return fallback
