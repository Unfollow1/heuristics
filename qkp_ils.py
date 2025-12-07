#!/usr/bin/env python3
"""
qkp_ils.py

Iterated Local Search (ILS) solver for the Quadratic 0/1 Knapsack Problem (QKP).
(versão ajustada para compatibilidade com runner: imprime BEST_OBJ e TOTAL_TIME e NÃO salva JSON)
"""

import sys
import math
import random
import time
import argparse
from typing import List, Dict, Tuple, Set


def parse_instance(path: str):
    with open(path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip() != ""]

    idx = 0

    # 1) Pode existir uma linha de referência antes do n
    try:
        n = int(lines[idx])
        idx += 1
    except:
        # pula a referência
        idx += 1
        n = int(lines[idx])
        idx += 1

    # 2) coeficientes lineares
    values = list(map(float, lines[idx].split()))
    if len(values) != n:
        raise ValueError("Erro ao ler coeficientes lineares")
    idx += 1

    # 3) coeficientes quadráticos (triangular superior)
    synergy = {}

    for i in range(n):
        row = list(map(float, lines[idx].split()))

        # Caso 1: veio só triangular puro
        if len(row) == n - i - 1:
            for offset, val in enumerate(row, start=1):
                j = i + offset
                synergy[(i, j)] = val
            idx += 1
            continue

        # Caso 2: veio com diagonal (n - i valores)
        if len(row) == n - i:
            row = row[1:]  # ignora diagonal c_ii
            for offset, val in enumerate(row, start=1):
                j = i + offset
                synergy[(i, j)] = val
            idx += 1
            continue

        # Caso 3: matriz completa (n valores sempre)
        if len(row) == n:
            for j in range(i + 1, n):
                synergy[(i, j)] = row[j]
            idx += 1
            continue

        raise ValueError(
            f"Erro na linha quadrática {i}: tamanho inesperado {len(row)}"
        )

    # 4) tipo da restrição (0 ou 1)
    constraint_type = int(lines[idx])
    idx += 1

    # 5) capacidade
    W = float(lines[idx])
    idx += 1

    # 6) pesos
    weights = list(map(float, lines[idx].split()))
    if len(weights) != n:
        raise ValueError("Erro ao ler pesos")
    idx += 1

    return n, values, synergy, weights, W



class QKPSolverILS:
    def __init__(self, n: int, values: List[float], synergy: Dict[Tuple[int, int], float],
                 weights: List[float], W: float,
                 seed: int = None,
                 time_limit: float = 60.0,
                 max_no_improve_iters: int = 200,
                 perturb_strength: int = 3,
                 max_iters: int = 10000):
        self.n = n
        self.values = values
        self.synergy = synergy
        self.weights = weights
        self.W = W
        self.time_limit = time_limit
        self.max_no_improve_iters = max_no_improve_iters
        self.perturb_strength = perturb_strength
        self.max_iters = max_iters
        self.best_sol = None
        self.best_obj = -math.inf
        self.best_weight = None
        self.start_time = None
        if seed is not None:
            random.seed(seed)

        self.adj = [[] for _ in range(n)]
        for (i, j), s in synergy.items():
            self.adj[i].append((j, s))
            self.adj[j].append((i, s))

    def _objective_of_set(self, selected: Set[int]) -> float:
        lin = sum(self.values[i] for i in selected)
        quad = 0.0
        sset = set(selected)
        for i in selected:
            for j, s in self.adj[i]:
                if j > i and j in sset:
                    quad += s
        return lin + quad

    def _weight_of_set(self, selected: Set[int]) -> float:
        return sum(self.weights[i] for i in selected)

    def greedy_init(self) -> Set[int]:
        items = list(range(self.n))
        ratio = [(self.values[i] / (self.weights[i] if self.weights[i] > 0 else 1e-9), i) for i in items]
        ratio.sort(reverse=True)
        selected = set()
        cur_w = 0.0
        for r, i in ratio:
            if cur_w + self.weights[i] <= self.W:
                selected.add(i)
                cur_w += self.weights[i]
        return selected

    def compute_synergy_sums(self, selected: Set[int]) -> List[float]:
        ssum = [0.0] * self.n
        sset = set(selected)
        for i in range(self.n):
            s = 0.0
            for j, val in self.adj[i]:
                if j in sset:
                    s += val
            ssum[i] = s
        return ssum

    def local_search(self, selected: Set[int]) -> Tuple[Set[int], float, float]:
        cur_sel = set(selected)
        cur_w = self._weight_of_set(cur_sel)
        cur_obj = self._objective_of_set(cur_sel)
        ssum = self.compute_synergy_sums(cur_sel)

        improved = True
        iter_count = 0
        while improved and (time.time() - self.start_time) < self.time_limit:
            improved = False
            best_move = None
            best_delta = 0.0

            # add
            for i in range(self.n):
                if i in cur_sel:
                    continue
                wi = self.weights[i]
                if cur_w + wi > self.W:
                    continue
                delta = self.values[i] + ssum[i]
                if delta > best_delta:
                    best_delta = delta
                    best_move = ('add', i)

            # remove
            for i in list(cur_sel):
                delta = -(self.values[i] + ssum[i])
                if delta > best_delta:
                    best_delta = delta
                    best_move = ('remove', i)

            # swap (possibly sampled)
            sel_list = list(cur_sel)
            non_sel = [x for x in range(self.n) if x not in cur_sel]
            MAX_NEIGH = 50000
            if len(sel_list) * len(non_sel) > MAX_NEIGH:
                sample_k = min(len(non_sel), 200)
                sample_i = min(len(sel_list), 200)
                cand_non = random.sample(non_sel, sample_k)
                cand_sel = random.sample(sel_list, sample_i)
            else:
                cand_non = non_sel
                cand_sel = sel_list

            for i in cand_sel:
                for k in cand_non:
                    new_w = cur_w - self.weights[i] + self.weights[k]
                    if new_w > self.W:
                        continue
                    delta = self.values[k] - self.values[i] + ssum[k] - ssum[i]
                    if delta > best_delta:
                        best_delta = delta
                        best_move = ('swap', i, k)

            if best_move is not None and best_delta > 1e-9:
                improved = True
                if best_move[0] == 'add':
                    i = best_move[1]
                    cur_sel.add(i)
                    cur_w += self.weights[i]
                    cur_obj += best_delta
                    for j, s in self.adj[i]:
                        ssum[j] += s
                    ssum[i] = 0.0
                elif best_move[0] == 'remove':
                    i = best_move[1]
                    cur_sel.remove(i)
                    cur_w -= self.weights[i]
                    cur_obj += best_delta
                    for j, s in self.adj[i]:
                        ssum[j] -= s
                    ssum[i] = 0.0
                elif best_move[0] == 'swap':
                    i = best_move[1]
                    k = best_move[2]
                    cur_sel.remove(i)
                    cur_sel.add(k)
                    cur_w = cur_w - self.weights[i] + self.weights[k]
                    cur_obj += best_delta
                    for j, s in self.adj[i]:
                        ssum[j] -= s
                    for j, s in self.adj[k]:
                        ssum[j] += s
                    ssum[i] = 0.0
                    ssum[k] = 0.0
                iter_count += 1
            else:
                break

        return cur_sel, cur_obj, cur_w

    def perturb(self, selected: Set[int], strength: int) -> Set[int]:
        sel = set(selected)
        cur_w = self._weight_of_set(sel)
        for _ in range(strength):
            if len(sel) == 0 or (random.random() < 0.5 and len(sel) < self.n):
                candidates = [i for i in range(self.n) if i not in sel]
                if not candidates:
                    continue
                k = random.choice(candidates)
                if cur_w + self.weights[k] <= self.W:
                    sel.add(k)
                    cur_w += self.weights[k]
                else:
                    if sel:
                        i = random.choice(list(sel))
                        if cur_w - self.weights[i] + self.weights[k] <= self.W:
                            sel.remove(i)
                            sel.add(k)
                            cur_w = cur_w - self.weights[i] + self.weights[k]
            else:
                i = random.choice(list(sel))
                sel.remove(i)
                cur_w -= self.weights[i]
        return sel

    def solve(self):
        self.start_time = time.time()
        cur_sel = self.greedy_init()
        cur_obj = self._objective_of_set(cur_sel)
        cur_w = self._weight_of_set(cur_sel)

        self.best_sol = set(cur_sel)
        self.best_obj = cur_obj
        self.best_weight = cur_w

        no_improve = 0
        iters = 0
        while (time.time() - self.start_time) < self.time_limit and no_improve < self.max_no_improve_iters and iters < self.max_iters:
            new_sel, new_obj, new_w = self.local_search(cur_sel)
            iters += 1
            if new_obj > self.best_obj + 1e-9:
                self.best_obj = new_obj
                self.best_sol = set(new_sel)
                self.best_weight = new_w
                no_improve = 0
            else:
                no_improve += 1
            cur_sel = self.perturb(self.best_sol, self.perturb_strength)

        end_time = time.time()
        return {
            'best_obj': self.best_obj,
            'best_weight': self.best_weight,
            'best_sol': sorted(list(self.best_sol)),
            'time': end_time - self.start_time,
            'iters': iters
        }


def main():
    parser = argparse.ArgumentParser(description='Iterated Local Search for Quadratic Knapsack Problem')
    parser.add_argument('instance', type=str, help='path to instance file')
    parser.add_argument('--time-limit', type=float, default=60.0, help='time limit in seconds')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--perturb', type=int, default=3, help='perturbation strength (k flips)')
    parser.add_argument('--max-no-improve', type=int, default=1000, help='stop after this many ILS iterations without improvement')
    parser.add_argument('--max-iters', type=int, default=1500, help='max ILS iterations')
    args = parser.parse_args()

    n, values, synergy, weights, W = parse_instance(args.instance)
    print("DEBUG W =", W)
    print("DEBUG min weight =", min(weights))
    print("DEBUG max weight =", max(weights))

    solver = QKPSolverILS(n, values, synergy, weights, W,
                          seed=args.seed,
                          time_limit=args.time_limit,
                          max_no_improve_iters=args.max_no_improve,
                          perturb_strength=args.perturb,
                          max_iters=args.max_iters)
    t0 = time.time()
    res = solver.solve()
    total_time = time.time() - t0

    # Human-friendly prints
    print('Instance:', args.instance)
    print('n=', n)
    print('Best objective:', res['best_obj'])
    print('Best weight:', res['best_weight'])
    print('Selected items (0-based):', res['best_sol'])
    print('Time (s):', res['time'])
    print('Iters:', res['iters'])

    # Strict lines expected by runner
    print(f"BEST_OBJ={res['best_obj']}", flush=True)
    print(f"TOTAL_TIME={total_time}", flush=True)


if __name__ == '__main__':
    main()