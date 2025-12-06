#!/usr/bin/env python3
"""
qkp_tabu.py

Busca Tabu para o Quadratic 0/1 Knapsack Problem (QKP).

Usa lista tabu para evitar ciclos e explorar o espaço de soluções de forma inteligente.
"""

import sys
import math
import random
import time
import json
import argparse
from typing import List, Dict, Tuple, Set


def parse_instance(path: str):
    """Parse instance file supporting multiple QKP formats.
    Returns: n, values (list), synergy dict {(i,j):s}, weights (list), capacity W
    Indexing: items are 0..n-1; synergy keys always (i,j) with i<j
    """
    with open(path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]

    # remove empty lines
    idx = 0
    while idx < len(lines) and lines[idx] == '':
        idx += 1
    if idx >= len(lines):
        raise ValueError('Empty instance file')

    # Read n
    n = int(lines[idx].split()[0])
    idx += 1

    # Read linear coefficients (values)
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
        raise ValueError(f'Expected {n} values, got {len(vals)}')

    # Read synergy matrix (n lines)
    synergy = {}
    for i in range(n):
        while idx < len(lines) and lines[idx] == '':
            idx += 1
        if idx >= len(lines):
            raise ValueError('Unexpected EOF in synergy matrix')
        parts = [float(p) for p in lines[idx].split()]
        idx += 1
        
        # Store only upper triangle (j > i)
        for j in range(i + 1, n):
            if j - i - 1 < len(parts):
                synergy[(i, j)] = parts[j - i - 1]

    # Skip blank lines
    while idx < len(lines) and lines[idx] == '':
        idx += 1
    if idx >= len(lines):
        raise ValueError('Unexpected EOF after synergy matrix')

    # Check if next line is constraint type (0) or capacity
    next_val = float(lines[idx].split()[0])
    
    if next_val == 0:
        # Format: 0, W, weights
        idx += 1
        while idx < len(lines) and lines[idx] == '':
            idx += 1
        W = float(lines[idx].split()[0])
        idx += 1
    else:
        # Format: W (or weights first)
        # Try to detect: if next_val is very large, it's W, else it's first weight
        if len(lines[idx].split()) == 1 and next_val > 1000:
            W = next_val
            idx += 1
        else:
            # weights come first, read all weights then W
            weights = []
            while len(weights) < n and idx < len(lines):
                if lines[idx] == '':
                    idx += 1
                    continue
                for p in lines[idx].split():
                    if len(weights) < n:
                        weights.append(float(p))
                idx += 1
            
            while idx < len(lines) and lines[idx] == '':
                idx += 1
            W = float(lines[idx].split()[0])
            
            values = [float(x) for x in vals]
            weights = [float(x) for x in weights]
            return n, values, synergy, weights, W

    # Read weights
    while idx < len(lines) and lines[idx] == '':
        idx += 1
    
    weights = []
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

    values = [float(x) for x in vals]
    weights = [float(x) for x in weights]

    return n, values, synergy, weights, W


class QKPSolverTabu:

    def __init__(self, n: int, values: List[float], synergy: Dict[Tuple[int,int], float], 
                 weights: List[float], W: float,
                 seed: int = None,
                 time_limit: float = 60.0,
                 tabu_tenure: int = 10,
                 max_iters: int = 1000):
        """
        Inicializa o solver de Busca Tabu.
        
        Parâmetros:
        - tabu_tenure: quantas iterações um movimento fica proibido
        - max_iters: máximo de iterações
        """
        self.n = n
        self.values = values
        self.synergy = synergy
        self.weights = weights
        self.W = W
        self.time_limit = time_limit
        self.tabu_tenure = tabu_tenure
        self.max_iters = max_iters
        
        self.best_sol = None
        self.best_obj = -math.inf
        self.best_weight = None
        self.start_time = None
        
        if seed is not None:
            random.seed(seed)
        
        # Pré-computar adjacências
        self.adj = [[] for _ in range(n)]
        for (i, j), s in synergy.items():
            self.adj[i].append((j, s))
            self.adj[j].append((i, s))
    
    def _objective_of_set(self, selected: Set[int]) -> float:
        """Calcula objetivo completo (linear + quadrático)."""
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
        """Calcula peso total da solução."""
        return sum(self.weights[i] for i in selected)

    def compute_synergy_sums(self, selected: Set[int]) -> List[float]:
        """
        Calcula synergy_sum para cada item.
        synergy_sum[i] = soma das sinergias de i com todos os itens selecionados.
        """
        ssum = [0.0] * self.n
        for i in range(self.n):
            s = 0.0
            for j, val in self.adj[i]:
                if j in selected:
                    s += val
            ssum[i] = s
        return ssum

    def greedy_init(self) -> Set[int]:
        """Solução inicial gulosa por valor/peso."""
        items = list(range(self.n))
        ratio = [(self.values[i] / (self.weights[i] if self.weights[i] > 0 else 1e-9), i) 
                for i in items]
        ratio.sort(reverse=True)
        selected = set()
        cur_w = 0.0
        for r, i in ratio:
            if cur_w + self.weights[i] <= self.W:
                selected.add(i)
                cur_w += self.weights[i]
        return selected

    def local_search(self, selected: Set[int]) -> Tuple[Set[int], float, float]:
        """
        Busca local best-improvement (EXATAMENTE igual ao ILS).
        """
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

            # 1) ADD
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

            # 2) REMOVE
            for i in list(cur_sel):
                delta = -(self.values[i] + ssum[i])
                if delta > best_delta:
                    best_delta = delta
                    best_move = ('remove', i)

            # 3) SWAP
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

            # Aplicar melhor movimento
            if best_move is not None and best_delta > 1e-9:  # ← MUDANÇA AQUI
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
        """Perturba a solução com k-flips aleatórios."""
        sel = set(selected)
        cur_w = self._weight_of_set(sel)
        
        for _ in range(strength):
            if len(sel) == 0 or (random.random() < 0.5 and len(sel) < self.n):
                # Tentar adicionar
                candidates = [i for i in range(self.n) if i not in sel]
                if not candidates:
                    continue
                k = random.choice(candidates)
                if cur_w + self.weights[k] <= self.W:
                    sel.add(k)
                    cur_w += self.weights[k]
                else:
                    # Swap se não couber
                    if sel:
                        i = random.choice(list(sel))
                        if cur_w - self.weights[i] + self.weights[k] <= self.W:
                            sel.remove(i)
                            sel.add(k)
                            cur_w = cur_w - self.weights[i] + self.weights[k]
            else:
                # Remover
                i = random.choice(list(sel))
                sel.remove(i)
                cur_w -= self.weights[i]
        
        return sel

    def solve(self):
        """Executa a Busca Tabu com melhorias: tenure adaptativo, restart, busca local e intensificação."""
        self.start_time = time.time()
        
        # Solução inicial com busca local
        initial = self.greedy_init()
        cur_sel, cur_obj, cur_w = self.local_search(initial)
        
        # Melhor solução global
        self.best_sol = set(cur_sel)
        self.best_obj = cur_obj
        self.best_weight = cur_w
        
        # Lista tabu: armazena (item, iteração_que_expira)
        tabu_list = {}
        
        # Controle de diversificação
        iters_sem_melhora = 0
        restart_threshold = 50  # Reinicia após 50 iters sem melhora
        
        # Tenure adaptativo
        current_tenure = self.tabu_tenure
        min_tenure = max(3, self.tabu_tenure // 3)
        max_tenure = self.tabu_tenure * 2
        
        iters = 0
        restarts = 0
        
        while (time.time() - self.start_time) < self.time_limit and iters < self.max_iters:
            # Calcular synergy_sums para a solução atual
            ssum = self.compute_synergy_sums(cur_sel)
            
            # Encontrar melhor movimento (não-tabu ou com aspiração)
            best_move = None
            best_delta = -math.inf
            best_new_obj = -math.inf
            
            # 1) Tentar ADD (adicionar item não selecionado)
            for i in range(self.n):
                if i in cur_sel:
                    continue
                if cur_w + self.weights[i] > self.W:
                    continue
                
                delta = self.values[i] + ssum[i]
                new_obj = cur_obj + delta
                
                # Verificar se movimento é tabu
                is_tabu = (i in tabu_list and tabu_list[i] > iters)
                
                # Critério de aspiração: aceita se melhor que best global
                aspiration = new_obj > self.best_obj
                
                if (not is_tabu or aspiration) and delta > best_delta:
                    best_delta = delta
                    best_move = ('add', i)
                    best_new_obj = new_obj
            
            # 2) Tentar REMOVE (remover item selecionado)
            for i in list(cur_sel):
                delta = -(self.values[i] + ssum[i])
                new_obj = cur_obj + delta
                
                is_tabu = (i in tabu_list and tabu_list[i] > iters)
                aspiration = new_obj > self.best_obj
                
                if (not is_tabu or aspiration) and delta > best_delta:
                    best_delta = delta
                    best_move = ('remove', i)
                    best_new_obj = new_obj
            
            # 3) Tentar SWAP (trocar item selecionado por não selecionado)
            sel_list = list(cur_sel)
            non_sel = [x for x in range(self.n) if x not in cur_sel]
            
            # Limitar tamanho da vizinhança para performance
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
                    new_obj = cur_obj + delta
                    
                    # Movimento swap é tabu se qualquer um dos itens estiver tabu
                    is_tabu = ((i in tabu_list and tabu_list[i] > iters) or 
                              (k in tabu_list and tabu_list[k] > iters))
                    aspiration = new_obj > self.best_obj
                    
                    if (not is_tabu or aspiration) and delta > best_delta:
                        best_delta = delta
                        best_move = ('swap', i, k)
                        best_new_obj = new_obj
            
            # Aplicar melhor movimento encontrado
            if best_move is None:
                # Nenhum movimento viável - forçar restart
                iters_sem_melhora = restart_threshold
            else:
                if best_move[0] == 'add':
                    i = best_move[1]
                    cur_sel.add(i)
                    cur_w += self.weights[i]
                    cur_obj = best_new_obj
                    tabu_list[i] = iters + current_tenure
                    
                elif best_move[0] == 'remove':
                    i = best_move[1]
                    cur_sel.remove(i)
                    cur_w -= self.weights[i]
                    cur_obj = best_new_obj
                    tabu_list[i] = iters + current_tenure
                    
                elif best_move[0] == 'swap':
                    i = best_move[1]  # sai
                    k = best_move[2]  # entra
                    cur_sel.remove(i)
                    cur_sel.add(k)
                    cur_w = cur_w - self.weights[i] + self.weights[k]
                    cur_obj = best_new_obj
                    tabu_list[i] = iters + current_tenure
                    tabu_list[k] = iters + current_tenure
                
                # Atualizar melhor solução global
                if cur_obj > self.best_obj + 1e-9:
                    self.best_obj = cur_obj
                    self.best_sol = set(cur_sel)
                    self.best_weight = cur_w
                    iters_sem_melhora = 0
                    # Reduzir tenure quando encontra melhora (intensificação)
                    current_tenure = max(min_tenure, current_tenure - 1)
                else:
                    iters_sem_melhora += 1
                    # Aumentar tenure quando não encontra melhora (diversificação)
                    if iters_sem_melhora % 10 == 0:
                        current_tenure = min(max_tenure, current_tenure + 1)
            
            # RESTART: Se ficou preso, reiniciar de uma solução perturbada
            if iters_sem_melhora >= restart_threshold:
                restarts += 1
                # Perturbar a melhor solução encontrada
                perturbed = self.perturb(self.best_sol, strength=max(10, self.n // 50))
                # Busca local na solução perturbada
                cur_sel, cur_obj, cur_w = self.local_search(perturbed)
                # Limpar lista tabu
                tabu_list = {}
                # Resetar tenure
                current_tenure = self.tabu_tenure
                iters_sem_melhora = 0
            
            iters += 1
        
        return {
            'best_obj': self.best_obj,
            'best_weight': self.best_weight,
            'best_sol': sorted(list(self.best_sol)) if self.best_sol else [],
            'time': time.time() - self.start_time,
            'iters': iters,
            'restarts': restarts
        }


def main():
    parser = argparse.ArgumentParser(description='Busca Tabu para Quadratic Knapsack Problem')
    parser.add_argument('instance', type=str, help='caminho para arquivo de instância')
    parser.add_argument('--time-limit', type=float, default=60.0, help='limite de tempo em segundos')
    parser.add_argument('--seed', type=int, default=None, help='semente aleatória')
    parser.add_argument('--tabu-tenure', type=int, default=10, help='tamanho da lista tabu')
    parser.add_argument('--max-iters', type=int, default=1000, help='máximo de iterações')
    parser.add_argument('--save', type=str, default=None, help='caminho para salvar resultado JSON')
    args = parser.parse_args()

    n, values, synergy, weights, W = parse_instance(args.instance)
    solver = QKPSolverTabu(n, values, synergy, weights, W,
                           seed=args.seed,
                           time_limit=args.time_limit,
                           tabu_tenure=args.tabu_tenure,
                           max_iters=args.max_iters)
    res = solver.solve()

    print('Instance:', args.instance)
    print('n=', n)
    print('Best objective:', res['best_obj'])
    print('Best weight:', res['best_weight'])
    print('Selected items (0-based):', res['best_sol'])
    print('Time (s):', res['time'])
    print('Iters:', res['iters'])
    print('Restarts:', res['restarts'])

    if args.save:
        out = {
            'instance': args.instance,
            'n': n,
            'best_obj': res['best_obj'],
            'best_weight': res['best_weight'],
            'best_sol': res['best_sol'],
            'time': res['time'],
            'iters': res['iters'],
            'restarts': res['restarts']
        }
        with open(args.save, 'w') as f:
            json.dump(out, f, indent=2)
        print('Saved result to', args.save)


if __name__ == '__main__':
    main()