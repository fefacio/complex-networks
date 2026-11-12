"""DemDS de Peng et al. (2025) para grafos simples em memória.

Referência da implementação: https://github.com/xinyepeng521/DemDS
Artigo: https://doi.org/10.1007/s11227-025-07724-7
"""

from __future__ import annotations

from collections.abc import Callable, Hashable
from heapq import heapify, heappop, heappush
from itertools import chain
from operator import index
from random import Random
from time import process_time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from networkx import Graph


_DEFAULT_ITERATIONS = 10_000
_CPP_ITERATION_LIMIT = 90_000_000_000
_PROTECTED_UNTIL = 2_147_483_643


def _cpp_random(seed: int) -> Random:
    # std::mt19937 e Random usam o mesmo gerador, com inicializações distintas.
    mask = (1 << 32) - 1
    state = [seed & mask]
    for position in range(1, 624):
        previous = state[-1]
        state.append((1_812_433_253 * (previous ^ (previous >> 30)) + position) & mask)
    rng = Random(0)
    rng.setstate((3, (*state, 624), None))
    return rng


def _remove_indexed(items: list[int], positions: list[int], vertex: int) -> None:
    """Remove em O(1), mantendo a ordem de troca usada pelo C++."""
    position = positions[vertex]
    last = items[-1]
    items[position] = last
    positions[last] = position
    items.pop()
    positions[vertex] = -1


class _DemDS:
    def __init__(
        self,
        adjacency: list[tuple[int, ...]],
        seed: int,
        on_improvement: Callable[[int], None] | None = None,
    ) -> None:
        self.adj = adjacency
        n = len(adjacency)
        self._on_improvement = on_improvement
        self._reported_size = n + 1
        self.selected = bytearray(n)
        self.protected = bytearray(n)
        self.rng = _cpp_random(seed)
        self.solution, self.fixed_count = self._initial_solution()
        self._report_improvement()
        self.solution_position = [-1] * n
        for position, vertex in enumerate(self.solution):
            self.solution_position[vertex] = position

        # Quantidades relativas às vizinhanças abertas. Um vértice escolhido
        # domina a si próprio, mesmo que tenha zero vizinhos escolhidos.
        self.cover = [sum(self.selected[u] for u in row) for row in adjacency]
        self.gain = [0] * n
        self.loss = [0] * n
        self.owner = [-1] * n
        self.uncovered: list[int] = []
        self.uncovered_position = [-1] * n
        self.age = [0] * n
        for vertex in range(n):
            if self.cover[vertex] == 1 and not self.selected[vertex]:
                self._set_owner(vertex, self._unique_neighbor(vertex))

    def _report_improvement(self) -> None:
        if self._on_improvement is None:
            return
        size = len(self.solution)
        if size < self._reported_size:
            self._reported_size = size
            self._on_improvement(size)

    def _initial_solution(self) -> tuple[list[int], int]:
        covered = bytearray(len(self.adj))
        for vertex, neighbors in enumerate(self.adj):
            if covered[vertex]:
                continue
            if not neighbors:
                picked = vertex
            elif len(neighbors) == 1:
                picked = neighbors[0]
                self.protected[vertex] = 1
            elif len(neighbors) == 2:
                first, second = neighbors
                if len(self.adj[first]) == 2 and second in self.adj[first]:
                    picked, other = second, first
                elif len(self.adj[second]) == 2 and first in self.adj[second]:
                    picked, other = first, second
                else:
                    continue
                self.protected[vertex] = self.protected[other] = 1
            else:
                continue
            self.selected[picked] = covered[picked] = 1
            for neighbor in self.adj[picked]:
                covered[neighbor] = 1

        fixed = [v for v, selected in enumerate(self.selected) if selected]
        remaining = len(covered) - sum(covered)
        if not remaining:
            return fixed, len(fixed)

        # O ganho só diminui. Corrigir entradas do heap ao retirá-las
        # preserva o desempate pelo menor ID do guloso do C++.
        gain = [int(not covered[v]) + sum(not covered[u] for u in row)
                for v, row in enumerate(self.adj)]
        queue = [(-score, vertex) for vertex, score in enumerate(gain) if score]
        heapify(queue)
        added = []
        while remaining:
            negative_score, vertex = heappop(queue)
            if -negative_score != gain[vertex]:
                if gain[vertex]:
                    heappush(queue, (-gain[vertex], vertex))
                continue
            self.selected[vertex] = 1
            added.append(vertex)
            for newly_covered in chain((vertex,), self.adj[vertex]):
                if covered[newly_covered]:
                    continue
                covered[newly_covered] = 1
                remaining -= 1
                gain[newly_covered] -= 1
                for neighbor in self.adj[newly_covered]:
                    gain[neighbor] -= 1
        # O C++ inicia o vetor de busca pelos fixos e depois pelos demais IDs.
        added.sort()
        return fixed + added, len(fixed)

    def _unique_neighbor(self, vertex: int) -> int:
        return next(u for u in self.adj[vertex] if self.selected[u])

    def _set_owner(self, vertex: int, owner: int) -> None:
        previous = self.owner[vertex]
        if previous != -1:
            self.loss[previous] -= 1
        self.owner[vertex] = owner
        if owner != -1:
            self.loss[owner] += 1

    def _set_uncovered(self, vertex: int, uncovered: bool) -> None:
        if uncovered:
            self.uncovered_position[vertex] = len(self.uncovered)
            self.uncovered.append(vertex)
        else:
            _remove_indexed(self.uncovered, self.uncovered_position, vertex)
        change = 1 if uncovered else -1
        for neighbor in self.adj[vertex]:
            self.gain[neighbor] += change

    def _insert(self, vertex: int) -> None:
        self.selected[vertex] = 1
        self.solution_position[vertex] = len(self.solution)
        self.solution.append(vertex)
        if self.cover[vertex] == 0:
            self._set_uncovered(vertex, False)
        elif self.cover[vertex] == 1:
            self._set_owner(vertex, -1)
        for neighbor in self.adj[vertex]:
            previous_cover = self.cover[neighbor]
            self.cover[neighbor] += 1
            if self.selected[neighbor]:
                continue
            if previous_cover == 0:
                self._set_uncovered(neighbor, False)
                self._set_owner(neighbor, vertex)
            elif previous_cover == 1:
                self._set_owner(neighbor, -1)

    def _remove(self, vertex: int) -> None:
        self.selected[vertex] = 0
        _remove_indexed(self.solution, self.solution_position, vertex)
        if self.cover[vertex] == 0:
            self._set_uncovered(vertex, True)
        elif self.cover[vertex] == 1:
            self._set_owner(vertex, self._unique_neighbor(vertex))
        for neighbor in self.adj[vertex]:
            self.cover[neighbor] -= 1
            if self.selected[neighbor]:
                continue
            if self.cover[neighbor] == 0:
                self._set_owner(neighbor, -1)
                self._set_uncovered(neighbor, True)
            elif self.cover[neighbor] == 1:
                self._set_owner(neighbor, self._unique_neighbor(neighbor))

    def _prune(self) -> None:
        position = 0
        while position < len(self.solution):
            vertex = self.solution[position]
            if self.loss[vertex] == 0 and self.cover[vertex] != 0:
                self._remove(vertex)
            position += 1

    def _swap_positions(self, left: int, right: int) -> None:
        a, b = self.solution[left], self.solution[right]
        self.solution[left], self.solution[right] = b, a
        self.solution_position[a], self.solution_position[b] = right, left

    def _select_move(self, iteration: int) -> tuple[int, int, int]:
        draw = self.rng.getrandbits
        age = self.age
        fixed = self.fixed_count
        y = self.uncovered[draw(32) % len(self.uncovered)]
        entering, best_gain, ties = -1, -1, 1
        for vertex in chain(self.adj[y], (y,)):
            if self.protected[vertex] and iteration < _PROTECTED_UNTIL:
                continue
            score = self.gain[vertex] + (self.cover[vertex] == 0)
            if score > best_gain:
                entering, best_gain = vertex, score
            elif score == best_gain:
                if age[vertex] < age[entering]:
                    entering, ties = vertex, 1
                elif age[vertex] == age[entering]:
                    ties += 1
                    if draw(32) % ties == 0:
                        entering = vertex

        # Os dois modos avaliam a mesma troca, mas usam listas diferentes
        # de candidatos à remoção. Este mapa conta as perdas recuperadas.
        neighbors = self.adj[entering]
        recovered: dict[int, int] = {}
        for vertex in chain(neighbors, (entering,)):
            owner = self.owner[vertex]
            if owner != -1:
                recovered[owner] = recovered.get(owner, 0) + 1

        local_count = 0
        local_vertex, local_score, local_ties = -1, -len(self.adj), 1
        if draw(32) % 100:
            candidates = []
            for vertex in neighbors:
                if self.solution_position[vertex] >= fixed:
                    candidates.append(vertex)
                    local_count += 1
                owner = self.owner[vertex]
                if owner != -1:
                    local_count += 1
                    if self.solution_position[owner] >= fixed:
                        candidates.append(owner)

            for vertex in candidates:
                score = best_gain + recovered.get(vertex, 0) - self.loss[vertex]
                if score > local_score:
                    local_vertex, local_score = vertex, score
                elif score == local_score:
                    if age[vertex] < age[local_vertex]:
                        local_vertex = vertex
                    elif age[vertex] == age[local_vertex]:
                        local_ties += 1
                        if draw(32) % local_ties == 0:
                            local_vertex = vertex

        window = max(1, local_count)
        general_vertex, general_score = -1, -len(self.adj)
        for position in range(fixed, min(fixed + window + 1, len(self.solution))):
            vertex = self.solution[position]
            score = best_gain + recovered.get(vertex, 0) - self.loss[vertex]
            if score > general_score:
                general_vertex, general_score = vertex, score
            elif score == general_score:
                if age[vertex] < age[general_vertex]:
                    general_vertex, ties = vertex, 1
                elif age[vertex] == age[general_vertex]:
                    ties += 1
                    if draw(32) % ties == 0:
                        general_vertex = vertex

        # MDSP::alpha é 30 no código publicado. O argumento da CLI altera
        # outra variável com o mesmo nome e não chega a este desempate.
        if local_score > general_score or (local_score == general_score and draw(32) % 100 > 30):
            return local_vertex, entering, window
        return general_vertex, entering, window

    def search(self, max_iterations: int, time_limit: int | None, created_at: float) -> tuple[int, ...]:
        self._prune()
        self._report_improvement()
        best = tuple(self.solution)
        fixed = self.fixed_count
        iteration = 0
        sampled_cpu = started_cpu = process_time() if time_limit is not None else 0.0
        last_prune_cpu = created_at

        while iteration < max_iterations and len(self.solution) > fixed:
            self._remove(self.solution[fixed])
            if self.uncovered and len(self.solution) == fixed:
                # Nenhuma troca consegue reduzir a solução mantendo os fixos.
                break
            while self.uncovered and iteration < max_iterations:
                if time_limit is not None and iteration % 1000 == 0:
                    sampled_cpu = process_time()
                    if int(sampled_cpu - started_cpu) > time_limit:
                        return best
                leaving, entering, window = self._select_move(iteration)
                self._insert(entering)
                self.age[entering] = self.age[leaving] = iteration
                self._remove(leaving)
                self._swap_positions(fixed, self.solution_position[entering])
                if not self.uncovered:
                    if time_limit is not None and int(sampled_cpu - last_prune_cpu) > 2:
                        self._report_improvement()
                        self._prune()
                        last_prune_cpu = sampled_cpu
                    break

                # A janela é limitada aos candidatos existentes para evitar
                # os acessos fora do vetor possíveis no original.
                size = len(self.solution)
                for position in range(fixed, min(fixed + window, size)):
                    random_position = fixed + self.rng.getrandbits(32) % (size - fixed)
                    self._swap_positions(position, random_position)
                iteration += 1
            if self.uncovered:
                break
            self._report_improvement()
            best = tuple(self.solution)
        return best


def mds_solver_demds(
    G: Graph,
    *,
    seed: int = 1,
    max_iterations: int | None = None,
    time_limit: int | None = None,
    on_improvement: Callable[[int], None] | None = None,
) -> set[Hashable]:
    """Retorna o melhor conjunto dominante encontrado, com os rótulos de G.

    Aceita um networkx.Graph simples, inclusive desconexo ou com isolados.
    Ignora atributos de peso. A ordem de iteração de G define os IDs internos.

    Escolha um orçamento. Com max_iterations, a busca é determinística para
    o mesmo grafo, ordenamento e seed. O contador segue o C++ e avança nas
    iterações que ainda deixam vértices descobertos. O padrão é 10 mil.

    Com time_limit, replica o limite do C++ em segundos inteiros de CPU da
    busca. O teste ocorre a cada mil iterações e para quando o contador de
    segundos ultrapassa o limite. O tempo total deve ser medido por fora.

    on_improvement recebe a cardinalidade da primeira solução dominante e
    de cada melhora estrita encontrada depois. A função fornecida pelo
    chamador pode registrar o instante em que recebeu cada valor.
    """
    if G.is_directed() or G.is_multigraph():
        raise ValueError("DemDS requer um grafo simples e não direcionado.")
    if max_iterations is not None and time_limit is not None:
        raise ValueError("Escolha max_iterations ou time_limit.")
    seed = index(seed)
    if not -(1 << 31) <= seed < (1 << 31):
        raise ValueError("A seed deve caber no inteiro de 32 bits do DemDS original.")
    if time_limit is not None:
        time_limit = index(time_limit)
        if time_limit < 0:
            raise ValueError("time_limit deve ser não negativo.")
        max_iterations = _CPP_ITERATION_LIMIT
    elif max_iterations is None:
        max_iterations = _DEFAULT_ITERATIONS
    else:
        max_iterations = index(max_iterations)
        if max_iterations < 0:
            raise ValueError("max_iterations deve ser não negativo.")

    created_at = process_time() if time_limit is not None else 0.0
    labels = tuple(G)
    if not labels:
        if on_improvement is not None:
            on_improvement(0)
        return set()
    positions = {vertex: i for i, vertex in enumerate(labels)}
    adjacency = []
    for vertex in labels:
        neighbors = G[vertex]
        if vertex in neighbors:
            raise ValueError("Remova os laços do grafo antes de executar o DemDS.")
        adjacency.append(tuple(sorted(positions[neighbor] for neighbor in neighbors)))
    del positions
    state = _DemDS(adjacency, seed, on_improvement)
    return {labels[i] for i in state.search(max_iterations, time_limit, created_at)}
