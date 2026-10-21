#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Лабораторная работа № 2. Вариант 14.
Задача: Максимальное покрытие графа (размещение k наблюдательных пунктов).
Представление: бинарная строка (1 - пункт есть, 0 - нет).

Реализовано:
- Генерация графа (N=25 вершин, связный).
- Оценка покрытия.
- Два способа работы с ограничениями: штраф (Penalty) и ремонт (Repair).
- Два оператора мутации: Bit-flip и Swap.
- Сравнение с жадным алгоритмом и случайным поиском.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Set

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------- Параметры задачи -----------------------------

N_VERTICES = 25
EDGE_PROB = 0.2
K_LIMIT = 6  # Максимальное число наблюдательных пунктов
PENALTY_WEIGHT = 10.0  # Коэффициент штрафа за превышение K


# ----------------------------- Генерация данных -----------------------------

def generate_graph(n: int, p: float, seed: int) -> np.ndarray:
    """Генерирует связный неориентированный граф (матрицу смежности)."""
    rng = random.Random(seed)
    adj = np.zeros((n, n), dtype=int)

    # Гарантируем связность (создаем остовное дерево)
    nodes = list(range(n))
    rng.shuffle(nodes)
    for i in range(n - 1):
        u, v = nodes[i], nodes[i + 1]
        adj[u, v] = adj[v, u] = 1

    # Добавляем случайные ребра
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] == 0 and rng.random() < p:
                adj[i, j] = adj[j, i] = 1

    return adj


def get_coverage(adj: np.ndarray, x: np.ndarray) -> int:
    """Считает количество покрытых вершин (сам пункт + его соседи)."""
    covered = np.zeros(len(x), dtype=bool)
    for i in range(len(x)):
        if x[i] == 1:
            covered[i] = True
            covered |= (adj[i] == 1)
    return int(np.sum(covered))


# ----------------------------- Генетические операторы -----------------------------

def repair(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Ремонт: приводит число единиц к ровно k."""
    y = x.copy()
    ones = np.where(y == 1)[0]
    zeros = np.where(y == 0)[0]

    if len(ones) > k:
        # Убираем лишние
        to_remove = rng.choice(ones, size=len(ones) - k, replace=False)
        y[to_remove] = 0
    elif len(ones) < k:
        # Добавляем недостающие
        to_add = rng.choice(zeros, size=k - len(ones), replace=False)
        y[to_add] = 1
    return y


def penalty_fitness(adj: np.ndarray, x: np.ndarray, k: int, w: float) -> float:
    """Фитнес со штрафом за превышение K."""
    cov = get_coverage(adj, x)
    excess = max(0, int(np.sum(x)) - k)
    return float(cov - w * excess)


def mutate_bitflip(x: np.ndarray, prob: float, rng: np.random.Generator) -> np.ndarray:
    """Битовая мутация: инвертирует случайный бит."""
    y = x.copy()
    mask = rng.random(len(y)) < prob
    y[mask] = 1 - y[mask]
    return y


def mutate_swap(x: np.ndarray, prob: float, rng: np.random.Generator) -> np.ndarray:
    """Мутация обмена: меняет местами 1 и 0 (сохраняет число единиц)."""
    y = x.copy()
    if rng.random() < prob:
        ones = np.where(y == 1)[0]
        zeros = np.where(y == 0)[0]
        if len(ones) > 0 and len(zeros) > 0:
            u = rng.choice(ones)
            v = rng.choice(zeros)
            y[u], y[v] = 0, 1
    return y


def crossover_uniform(p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Равномерный кроссовер."""
    mask = rng.random(len(p1)) < 0.5
    child = np.where(mask, p1, p2)
    return child


# ----------------------------- Конфигурация ГА -----------------------------

@dataclass
class GAConfig:
    name: str
    population_size: int
    mutation_prob: float
    crossover_prob: float
    constraint_mode: str  # "penalty" или "repair"
    mutation_mode: str  # "bitflip" или "swap"


def run_ga(cfg: GAConfig, adj: np.ndarray, seed: int, budget: int) -> Tuple[float, np.ndarray, int]:
    rng = np.random.default_rng(seed)
    n = len(adj)

    # Инициализация (случайные бинарные строки)
    pop = rng.integers(0, 2, size=(cfg.population_size, n))

    def evaluate(x):
        if cfg.constraint_mode == "penalty":
            return penalty_fitness(adj, x, K_LIMIT, PENALTY_WEIGHT)
        else:
            x_rep = repair(x, K_LIMIT, rng)
            return float(get_coverage(adj, x_rep))

    fit = np.array([evaluate(ind) for ind in pop])
    evals = cfg.population_size

    best_idx = int(np.argmax(fit))  # ТЕПЕРЬ МАКСИМИЗИРУЕМ!
    best_fit = float(fit[best_idx])
    best_x = pop[best_idx].copy()

    generation = 0
    while evals < budget:
        # Элитизм
        elite_idx = int(np.argmax(fit))
        elite = pop[elite_idx].copy()
        elite_fit = float(fit[elite_idx])

        n_children = min(cfg.population_size - 1, budget - evals)
        children = []
        child_fits = []

        for _ in range(n_children):
            # Турнирная селекция (на максимум)
            idx1 = rng.integers(0, len(pop), 3)
            p1 = pop[idx1[np.argmax(fit[idx1])]].copy()
            idx2 = rng.integers(0, len(pop), 3)
            p2 = pop[idx2[np.argmax(fit[idx2])]].copy()

            # Кроссовер
            if rng.random() < cfg.crossover_prob:
                child = crossover_uniform(p1, p2, rng)
            else:
                child = p1.copy()

            # Мутация
            if cfg.mutation_mode == "bitflip":
                child = mutate_bitflip(child, cfg.mutation_prob, rng)
            else:
                child = mutate_swap(child, cfg.mutation_prob, rng)

            # Если Repair, то применяем его для оценки
            if cfg.constraint_mode == "repair":
                child_eval = repair(child, K_LIMIT, rng)
                child_fit = evaluate(child_eval)
                children.append(child_eval)  # Сохраняем уже отремонтированного
            else:
                child_fit = evaluate(child)
                children.append(child)

            child_fits.append(child_fit)

        evals += n_children
        generation += 1

        pop = np.vstack([elite] + children)
        fit = np.array([elite_fit] + child_fits, dtype=float)

        idx = int(np.argmax(fit))
        if fit[idx] > best_fit:
            best_fit = float(fit[idx])
            best_x = pop[idx].copy()

    return best_fit, best_x, evals


# ----------------------------- Эвристики для сравнения -----------------------------

def greedy_coverage(adj: np.ndarray, k: int) -> Tuple[int, np.ndarray]:
    """Жадный алгоритм: на каждом шаге выбираем вершину с максимальным приростом покрытия."""
    n = len(adj)
    x = np.zeros(n, dtype=int)
    covered = np.zeros(n, dtype=bool)

    for _ in range(k):
        best_v = -1
        best_gain = -1
        for v in range(n):
            if x[v] == 1:
                continue
            # Прирост покрытия
            new_cov = covered.copy()
            new_cov[v] = True
            new_cov |= (adj[v] == 1)
            gain = np.sum(new_cov) - np.sum(covered)

            if gain > best_gain:
                best_gain = gain
                best_v = v

        x[best_v] = 1
        covered[best_v] = True
        covered |= (adj[best_v] == 1)

    return get_coverage(adj, x), x


def random_valid_search(adj: np.ndarray, seed: int, budget: int) -> Tuple[int, np.ndarray]:
    """Случайный поиск допустимых решений (ровно K единиц)."""
    rng = np.random.default_rng(seed)
    n = len(adj)
    best_cov = -1
    best_x = None

    for _ in range(budget):
        x = np.zeros(n, dtype=int)
        ones = rng.choice(n, size=K_LIMIT, replace=False)
        x[ones] = 1
        cov = get_coverage(adj, x)
        if cov > best_cov:
            best_cov = cov
            best_x = x.copy()

    return best_cov, best_x


# ----------------------------- Main -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--budget", type=int, default=5000)
    parser.add_argument("--outdir", type=str, default="results_lab2")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1. Генерация графа
    adj = generate_graph(N_VERTICES, EDGE_PROB, seed=args.seed)
    np.savetxt(outdir / "graph_adj.csv", adj, fmt='%d', delimiter=',')

    # 2. Конфигурации
    configs = [
        GAConfig("GA_Penalty_Bitflip", population_size=50, mutation_prob=0.05, crossover_prob=0.8,
                 constraint_mode="penalty", mutation_mode="bitflip"),
        GAConfig("GA_Repair_Bitflip", population_size=50, mutation_prob=0.05, crossover_prob=0.8,
                 constraint_mode="repair", mutation_mode="bitflip"),
        GAConfig("GA_Repair_Swap", population_size=50, mutation_prob=0.2, crossover_prob=0.8, constraint_mode="repair",
                 mutation_mode="swap")
    ]

    rows = []

    # --- Запуск ГА ---
    for ci, cfg in enumerate(configs):
        for run in range(args.runs):
            t0 = time.time()
            best_fit, best_x, evals = run_ga(cfg, adj, seed=args.seed + ci * 100 + run, budget=args.budget)
            elapsed = time.time() - t0
            rows.append({
                "config": cfg.name, "run": run + 1, "best_fitness": float(best_fit),
                "best_x": json.dumps(best_x.tolist()), "time_sec": round(elapsed, 6)
            })

    # --- Запуск случайного поиска ---
    for run in range(args.runs):
        best_cov, best_x = random_valid_search(adj, seed=args.seed + 500 + run, budget=args.budget)
        rows.append({
            "config": "Random_Search", "run": run + 1, "best_fitness": float(best_cov),
            "best_x": json.dumps(best_x.tolist()), "time_sec": 0.0
        })

    # --- Жадный алгоритм (однократный) ---
    greedy_cov, greedy_x = greedy_coverage(adj, K_LIMIT)
    print(f"Жадный алгоритм: покрытие = {greedy_cov}")

    # 3. Сохранение и статистика
    with (outdir / "runs.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["config", "run", "best_fitness", "best_x", "time_sec"])
        writer.writeheader()
        writer.writerows(rows)

    # Подсчет статистики
    print("\n--- Сводка результатов ---")
    configs_names = sorted(set(r["config"] for r in rows))
    for name in configs_names:
        vals = [r["best_fitness"] for r in rows if r["config"] == name]
        print(f"{name}: Best={np.max(vals):.1f}, Mean={np.mean(vals):.2f}, Worst={np.min(vals):.1f}")


if __name__ == "__main__":
    main()