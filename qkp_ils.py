#!/usr/bin/env python3
"""
qkp_ils.py

Iterated Local Search (ILS) solver for the Quadratic 0/1 Knapsack Problem (QKP).

Input: instance file in the format the user provided (see format.pdf):
- first line: n (number of variables)
- second line: n linear coefficients (v_i)
- next n-1 lines: upper-triangular quadratic coefficients (row i has n-i numbers: c_{i,i+1} ... c_{i,n})
- a blank line (optional)
- a line with 0 (<=) or 1 (=) for constraint type (we treat as <=)
- a line with capacity W
- a line with n weights (a_i)

Output: prints best objective, weight, selected set and saves a JSON with details if requested.

Algorithm highlights:
- Efficient incremental updates using `synergy_sum` array: for each item i, synergy_sum[i] = sum_{j in S} s_{ij}
- Neighborhood: add (0->1), remove (1->0), and swap (1<->0) with O(1) delta formulas
    - delta_add(i) = v_i + synergy_sum[i]
    - delta_remove(i) = -(v_i + synergy_sum[i])
    - delta_swap(i_in, k_out) = v_k - v_i + synergy_sum[k] - synergy_sum[i]
      (derived so s_{ik} cancels correctly)
- ILS structure: greedy initial solution (value/weight), local search (best-improvement), perturbation by random k-flips,
  repeated until no improvement for max_no_improve iterations or time limit.

Designed for speed and clarity. Tunable parameters: time_limit, max_iters, perturbation_strength, seed.

"""

import sys
import math
import random
import time
import json
import argparse
from typing import List, Dict, Tuple, Set


def parse_instance(path: str):
    """Parse instance file in the described format.
    Returns: n, values (list), synergy dict {(i,j):s}, weights (list), capacity W
    Indexing: items are 0..n-1; synergy keys always (i,j) with i<j
    """
    with open(path, 'r') as f:
        tokens = []
        # read all non-empty tokens separated by whitespace but preserve line breaks to know triangular block
        lines = [line.strip() for line in f.readlines()]

    # remove empty lines but keep blanks as separators for robustness
    # First meaningful token should be n
    idx = 0
    while idx < len(lines) and lines[idx] == '':
        idx += 1
    if idx >= len(lines):
        raise ValueError('Empty instance file')

    # Read n
    n_line = lines[idx].split()
    n = int(n_line[0])
    idx += 1

    # Read linear coefficients: may span multiple lines; gather until we have n numbers
    vals = []
    while len(vals) < n and idx < len(lines):
        if lines[idx] == '':
            idx += 1
            continue
        parts = lines[idx].split()
        for p in parts:
            if len(vals) < n:
                vals.append(float(p))
        idx += 1
    if len(vals) != n:
        raise ValueError(f'Expected {n} linear coefficients, got {len(vals)}')

    # Read upper-triangular quadratic coefficients: rows with lengths n-1, n-2, ..., 1
    synergy = {}
    for i in range(n-1):
        # skip blank lines
        while idx < len(lines) and lines[idx] == '':
            idx += 1
        if idx >= len(lines):
            raise ValueError('Unexpected EOF while reading quadratic coefficients')
        parts = lines[idx].split()
        expected = n - 1 - i
        if len(parts) < expected:
            # maybe coefficients span multiple lines: gather until we have expected
            vals_row = [float(p) for p in parts]
            idx += 1
            while len(vals_row) < expected and idx < len(lines):
                if lines[idx] == '':
                    idx += 1
                    continue
                more = lines[idx].split()
                for p in more:
                    if len(vals_row) < expected:
                        vals_row.append(float(p))
                idx += 1
            if len(vals_row) != expected:
                raise ValueError(f'Quadratic row {i} expected {expected} values, got {len(vals_row)}')
            parts = vals_row
        else:
            parts = [float(p) for p in parts[:expected]]
            idx += 1
        # map to synergy entries
        for offset, val in enumerate(parts, start=1):
            j = i + offset
            synergy[(i, j)] = float(val)

    # After triangular, skip blank lines
    while idx < len(lines) and lines[idx] == '':
        idx += 1
    if idx >= len(lines):
        raise ValueError('Unexpected EOF after quadratic block')

    # constraint type line (we ignore equality vs <=, treat as <=)
    constraint_type = int(lines[idx].split()[0])
    idx += 1

    # skip blanks
    while idx < len(lines) and lines[idx] == '':
        idx += 1
    if idx >= len(lines):
        raise ValueError('Missing capacity line')
    W = float(lines[idx].split()[0])
    idx += 1

    # skip blanks
    while idx < len(lines) and lines[idx] == '':
        idx += 1
    if idx >= len(lines):
        raise ValueError('Missing weights line')

    weights = [float(p) for p in lines[idx].split()]
    # if weights span multiple lines, gather until n
    idx += 1
    while len(weights) < n and idx < len(lines):
        if lines[idx] == '':
            idx += 1
            continue
        for p in lines[idx].split():
            if len(weights) < n:
                weights.append(float(p))
        idx += 1

    if len(weights) != n:
        raise ValueError(f'Expected {n} weights, got {len(weights)}')

    # convert vals and weights to lists of floats
    values = [float(x) for x in vals]
    weights = [float(x) for x in weights]

    return n, values, synergy, weights, W


class QKPSolverILS:
    def __init__(self, n: int, values: List[float], synergy: Dict[Tuple[int,int], float], weights: List[float], W: float,
                 seed: int = None,
                 time_limit: float = 60.0,
                 max_no_improve_iters: int = 200,
                 perturb_strength: int = 3,
                 max_iters: int = 10000):
        self.n = n
        self.values = values
        self.synergy = synergy  # (i,j) i<j
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

        # Precompute adjacency lists for faster synergy_sum updates
        self.adj = [[] for _ in range(n)]  # list of (neighbor, s)
        for (i, j), s in synergy.items():
            self.adj[i].append((j, s))
            self.adj[j].append((i, s))

    def _objective_of_set(self, selected: Set[int]) -> float:
        # compute full objective (linear + quadratic)
        lin = sum(self.values[i] for i in selected)
        quad = 0.0
        sel_list = sorted(selected)
        sset = set(selected)
        for i in sel_list:
            for j, s in self.adj[i]:
                if j > i and j in sset:
                    quad += s
        return lin + quad

    def _weight_of_set(self, selected: Set[int]) -> float:
        return sum(self.weights[i] for i in selected)

    def greedy_init(self) -> Set[int]:
        # simple greedy by value/weight ratio; alternately try value-only
        items = list(range(self.n))
        # avoid division by zero
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
        for i in range(self.n):
            if i not in selected:
                # compute sum of s_{i,j} for j in selected
                s = 0.0
                for j, val in self.adj[i]:
                    if j in selected:
                        s += val
                ssum[i] = s
            else:
                # for selected items also store their sum with other selected
                s = 0.0
                for j, val in self.adj[i]:
                    if j in selected and j > i:
                        s += val
                    elif j in selected and j < i:
                        s += val
                ssum[i] = s
        return ssum

    def local_search(self, selected: Set[int]) -> Tuple[Set[int], float, float]:
        """
        Perform best-improvement local search with add/remove/swap moves until local optimum.
        Returns improved (selected, objective, weight)
        """
        cur_sel = set(selected)
        cur_w = self._weight_of_set(cur_sel)
        cur_obj = self._objective_of_set(cur_sel)
        # initialize synergy sums
        ssum = self.compute_synergy_sums(cur_sel)

        improved = True
        iter_count = 0
        while improved and (time.time() - self.start_time) < self.time_limit:
            improved = False
            best_move = None
            best_delta = 0.0

            # 1) try best add
            # consider non-selected items that fit
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

            # 2) try best remove (remove if negative contribution)
            for i in list(cur_sel):
                delta = -(self.values[i] + ssum[i])
                if delta > best_delta:
                    best_delta = delta
                    best_move = ('remove', i)

            # 3) try best swap: i in selected, k not selected
            # We use O(|S| * (n-|S|)) iteration; if too big, can sample
            sel_list = list(cur_sel)
            non_sel = [x for x in range(self.n) if x not in cur_sel]
            # if neighborhood too large, sample a subset for performance
            MAX_NEIGH = 50000
            if len(sel_list) * len(non_sel) > MAX_NEIGH:
                # sample candidates
                sample_k = min(len(non_sel), 200)
                sample_i = min(len(sel_list), 200)
                cand_non = random.sample(non_sel, sample_k)
                cand_sel = random.sample(sel_list, sample_i)
            else:
                cand_non = non_sel
                cand_sel = sel_list
            for i in cand_sel:
                for k in cand_non:
                    # check weight feasibility: remove i, add k
                    new_w = cur_w - self.weights[i] + self.weights[k]
                    if new_w > self.W:
                        continue
                    # delta_swap = v_k - v_i + ssum[k] - ssum[i]
                    delta = self.values[k] - self.values[i] + ssum[k] - ssum[i]
                    if delta > best_delta:
                        best_delta = delta
                        best_move = ('swap', i, k)

            if best_move is not None and best_delta > 1e-9:
                improved = True
                if best_move[0] == 'add':
                    i = best_move[1]
                    # apply add
                    cur_sel.add(i)
                    # update weight and objective
                    cur_w += self.weights[i]
                    cur_obj += best_delta
                    # update ssum for all neighbors
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
                    # perform swap: remove i, add k
                    cur_sel.remove(i)
                    cur_sel.add(k)
                    cur_w = cur_w - self.weights[i] + self.weights[k]
                    cur_obj += best_delta
                    # update ssum: for neighbors of i subtract, neighbors of k add
                    for j, s in self.adj[i]:
                        ssum[j] -= s
                    for j, s in self.adj[k]:
                        ssum[j] += s
                    # recompute ssum entries for i and k
                    ssum[i] = 0.0
                    ssum[k] = 0.0
                iter_count += 1
                # continue search
            else:
                break

        return cur_sel, cur_obj, cur_w

    def perturb(self, selected: Set[int], strength: int) -> Set[int]:
        # random k-flips: try to flip `strength` items; if cannot add due to weight, do swaps
        sel = set(selected)
        cur_w = self._weight_of_set(sel)
        for _ in range(strength):
            if len(sel) == 0 or (random.random() < 0.5 and len(sel) < self.n):
                # try random add
                candidates = [i for i in range(self.n) if i not in sel]
                if not candidates:
                    continue
                k = random.choice(candidates)
                if cur_w + self.weights[k] <= self.W:
                    sel.add(k)
                    cur_w += self.weights[k]
                else:
                    # try swap with random selected
                    if sel:
                        i = random.choice(list(sel))
                        if cur_w - self.weights[i] + self.weights[k] <= self.W:
                            sel.remove(i)
                            sel.add(k)
                            cur_w = cur_w - self.weights[i] + self.weights[k]
            else:
                # remove random
                i = random.choice(list(sel))
                sel.remove(i)
                cur_w -= self.weights[i]
        return sel

    def solve(self):
        self.start_time = time.time()
        # initial solution
        cur_sel = self.greedy_init()
        cur_obj = self._objective_of_set(cur_sel)
        cur_w = self._weight_of_set(cur_sel)

        self.best_sol = set(cur_sel)
        self.best_obj = cur_obj
        self.best_weight = cur_w

        no_improve = 0
        iters = 0
        while (time.time() - self.start_time) < self.time_limit and no_improve < self.max_no_improve_iters and iters < self.max_iters:
            # local search
            new_sel, new_obj, new_w = self.local_search(cur_sel)
            iters += 1
            if new_obj > self.best_obj + 1e-9:
                self.best_obj = new_obj
                self.best_sol = set(new_sel)
                self.best_weight = new_w
                no_improve = 0
            else:
                no_improve += 1

            # perturb current solution from best
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
    parser.add_argument('--max-no-improve', type=int, default=200, help='stop after this many ILS iterations without improvement')
    parser.add_argument('--max-iters', type=int, default=20, help='max ILS iterations')
    parser.add_argument('--save', type=str, default=None, help='path to save JSON result')
    args = parser.parse_args()

    n, values, synergy, weights, W = parse_instance(args.instance)
    solver = QKPSolverILS(n, values, synergy, weights, W,
                          seed=args.seed,
                          time_limit=args.time_limit,
                          max_no_improve_iters=args.max_no_improve,
                          perturb_strength=args.perturb,
                          max_iters=args.max_iters)
    res = solver.solve()

    print('Instance:', args.instance)
    print('n=', n)
    print('Best objective:', res['best_obj'])
    print('Best weight:', res['best_weight'])
    print('Selected items (0-based):', res['best_sol'])
    print('Time (s):', res['time'])
    print('Iters:', res['iters'])

    if args.save:
        out = {
            'instance': args.instance,
            'n': n,
            'best_obj': res['best_obj'],
            'best_weight': res['best_weight'],
            'best_sol': res['best_sol'],
            'time': res['time'],
            'iters': res['iters']
        }
        with open(args.save, 'w') as f:
            json.dump(out, f, indent=2)
        print('Saved result to', args.save)


if __name__ == '__main__':
    main()