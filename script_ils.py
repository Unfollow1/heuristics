#!/usr/bin/env python3
"""
Runner automatizado para executar a ILS do QKP em múltiplas instâncias
 e salvar resultados em um arquivo TXT.

USO:
python run_ils_batch.py --input-dir ./instancias \
                        --solver ./qkp_ils.py \
                        --time-limit 120 \
                        --perturb 4 \
                        --restarts 10 \
                        --out results.txt

REQUISITOS:
- O script qkp_ils.py deve aceitar os parâmetros:
    python qkp_ils.py instancia.txt --time-limit T --perturb P --seed S
  e imprimir ao final
    BEST_OBJ=<valor>
    TOTAL_TIME=<tempo>

Esse runner é robusto: captura stdout, parseia o melhor valor e grava logs.
"""

import argparse
import os
import subprocess
import time
import random
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Diretório com arquivos de instância")
    parser.add_argument("--solver", type=str, required=True,
                        help="Caminho para o script qkp_ils.py")
    parser.add_argument("--time-limit", type=int, default=120)
    parser.add_argument("--perturb", type=int, default=4)
    parser.add_argument("--restarts", type=int, default=10,
                        help="Número de execuções por instância")
    parser.add_argument("--out", type=str, default="results.txt")
    parser.add_argument("--seed-base", type=int, default=1234)
    parser.add_argument("--extensions", type=str, default=".txt",
                        help="Extensões de instância, ex: .txt,.dat")
    return parser.parse_args()


def extract_best_obj(output: str):
    """Extrai BEST_OBJ da saída do solver."""
    best = None
    total_time = None
    for line in output.splitlines():
        if "BEST_OBJ=" in line:
            try:
                best = float(line.split("=")[1])
            except:
                pass
        if "TOTAL_TIME=" in line:
            try:
                total_time = float(line.split("=")[1])
            except:
                pass
    return best, total_time


def main():
    args = parse_args()

    exts = tuple(e.strip() for e in args.extensions.split(","))

    instances = []
    for root, dirs, files in os.walk(args.input_dir):
        for f in files:
            if f.endswith(exts):
                instances.append(os.path.join(root, f))
    instances = sorted(instances)

    if not instances:
        print("Nenhuma instância encontrada.")
        return

    print(f"Encontradas {len(instances)} instâncias.")

    with open(args.out, "w", encoding="utf-8") as fout:
        fout.write("# QKP ILS Batch Results\n")
        fout.write(f"# Data: {datetime.now()}\n")
        fout.write(f"# Solver: {args.solver}\n")
        fout.write(f"# TimeLimit: {args.time_limit}\n")
        fout.write(f"# Perturb: {args.perturb}\n")
        fout.write(f"# Restarts: {args.restarts}\n")
        fout.write("# instance,best_obj,best_time,avg_obj,avg_time,seed_base\n")
        fout.flush()

        for idx, inst in enumerate(instances):
            print(f"[{idx+1}/{len(instances)}] Executando: {os.path.basename(inst)}")

            best_global = float("-inf")
            times = []
            values = []

            for r in range(args.restarts):
                seed = args.seed_base + r

                cmd = [
                    "python3",
                    args.solver,
                    inst,
                    "--time-limit", str(args.time_limit),
                    "--perturb", str(args.perturb),
                    "--seed", str(seed),
                ]

                t0 = time.time()
                proc = subprocess.run(cmd, capture_output=True, text=True)
                t1 = time.time()

                out = proc.stdout
                err = proc.stderr

                best, t_solver = extract_best_obj(out)

                if best is None:
                    print("ERRO: não foi possível extrair BEST_OBJ")
                    print(out)
                    print(err)
                    continue

                values.append(best)
                times.append(t_solver if t_solver is not None else (t1 - t0))
                best_global = max(best_global, best)

            avg_val = sum(values) / len(values)
            avg_time = sum(times) / len(times)

            fout.write(f"{os.path.basename(inst)},{best_global:.6f},{min(times):.4f},{avg_val:.6f},{avg_time:.4f},{args.seed_base}\n")
            fout.flush()

            print(f"  Melhor = {best_global:.4f} | Média = {avg_val:.4f}")

    print("Execução finalizada. Resultados salvos em:", args.out)


if __name__ == "__main__":
    main()