from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt


# ----------------------------- Параметры задачи -----------------------------

D = 7
LOWER = np.full(D, -5.0)
UPPER = np.full(D, 5.0)


def fitness(x: np.ndarray) -> float:
    """f(x) = 0.5 * sum(x^4 - 16x^2 + 5x)."""
    x = np.asarray(x, dtype=float)
    return float(0.5 * np.sum(x**4 - 16.0 * x**2 + 5.0 * x))


# ----------------------------- Эволюционные операторы -----------------------------

def repair(x: np.ndarray) -> np.ndarray:
    """Отражение координат от границ области поиска"""
    span = UPPER - LOWER
    t = np.mod(x - LOWER, 2.0 * span)
    return LOWER + np.where(t <= span, t, 2.0 * span - t)


def tournament_select(population: np.ndarray, fit: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Турнирный отбор: из k случайных особей выбирается лучшая"""
    idx = rng.integers(0, len(population), size=k)
    best_local = idx[np.argmin(fit[idx])]
    return population[best_local].copy()


def blx_crossover(p1: np.ndarray, p2: np.ndarray, alpha: float, rng: np.random.Generator) -> np.ndarray:
    """BLX-alpha кроссовер для вещественных векторов"""
    cmin = np.minimum(p1, p2)
    cmax = np.maximum(p1, p2)
    delta = alpha * (cmax - cmin)
    return rng.uniform(cmin - delta, cmax + delta)


def mutate(x: np.ndarray, prob: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Гауссова мутация: каждая координата мутирует с вероятностью prob"""
    mask = rng.random(x.shape) < prob
    y = x.copy()
    n = int(mask.sum())
    if n > 0:
        y[mask] += rng.normal(0.0, sigma, size=n)
    return y


# ----------------------------- Конфигурация ГА -----------------------------

@dataclass
class GAConfig:
    name: str
    population_size: int
    mutation_prob: float
    crossover_prob: float = 0.9
    sigma: float = 1.0
    blx_alpha: float = 0.3
    tournament_size: int = 3


def run_ga(cfg: GAConfig, seed: int, budget: int) -> Tuple[float, np.ndarray, int, int, List[Tuple[int, int, float]]]:
    """Один запуск ГА. Возвращает лучший fitness, лучшую особь, число вычислений, поколения, траекторию."""
    rng = np.random.default_rng(seed)

    pop = rng.uniform(LOWER, UPPER, size=(cfg.population_size, D))
    fit = np.array([fitness(ind) for ind in pop])
    evals = cfg.population_size
    generation = 0

    best_idx = int(np.argmin(fit))
    best_fit = float(fit[best_idx])
    best_x = pop[best_idx].copy()

    traj = [(generation, evals, best_fit)]

    while evals < budget:
        #выбираем крутейшего, который пройдёт в следующее поколение
        elite_idx = int(np.argmin(fit))
        elite = pop[elite_idx].copy()
        elite_fit = float(fit[elite_idx])

        n_children = min(cfg.population_size - 1, budget - evals)#одно место ушло на элиту
        children = []
        child_fits = []

        for _ in range(n_children):
            p1 = tournament_select(pop, fit, cfg.tournament_size, rng)
            p2 = tournament_select(pop, fit, cfg.tournament_size, rng)

            if rng.random() < cfg.crossover_prob:
                child = blx_crossover(p1, p2, cfg.blx_alpha, rng)
            else:
                child = p1.copy()

            child = mutate(child, cfg.mutation_prob, cfg.sigma, rng)
            child = repair(child)

            children.append(child)
            child_fits.append(fitness(child))

        evals += n_children
        generation += 1

        pop = np.vstack([elite] + children)
        fit = np.array([elite_fit] + child_fits, dtype=float)

        idx = int(np.argmin(fit))
        if fit[idx] < best_fit:
            best_fit = float(fit[idx])
            best_x = pop[idx].copy()

        traj.append((generation, evals, best_fit))

    return best_fit, best_x, evals, generation, traj


def run_random_search(seed: int, budget: int) -> Tuple[float, np.ndarray, int]:
    rng = np.random.default_rng(seed)
    best_fit = float("inf")
    best_x = None

    #генерируем точки и считаем фитнес
    for _ in range(budget):
        x = rng.uniform(LOWER, UPPER, size=D)

        f = fitness(x)
        if f < best_fit:
            best_fit = f
            best_x = x.copy()

    return best_fit, best_x, budget


# ----------------------------- Сохранение результатов -----------------------------

def write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize(rows: List[dict]) -> List[dict]:
    result = []
    names = sorted(set(r["config"] for r in rows))
    for name in names:
        vals = np.array([float(r["best_fitness"]) for r in rows if r["config"] == name], dtype=float)
        result.append({
            "config": name,
            "runs": len(vals),
            "best": float(np.min(vals)),
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "worst": float(np.max(vals)),
        })
    return result


def plot_config(name: str, traj_list: List[np.ndarray], outdir: Path) -> None:
    """Строит график min/mean/max по поколениям."""
    arr = np.array([t[:, 2] for t in traj_list], dtype=float)
    gens = np.array(traj_list[0][:, 0], dtype=int)

    mn = arr.min(axis=0)
    mx = arr.max(axis=0)
    mean = arr.mean(axis=0)

    plt.figure(figsize=(8, 4.5))
    plt.fill_between(gens, mn, mx, alpha=0.2, label="min–max")
    plt.plot(gens, mean, label="mean")
    plt.plot(gens, mn, label="min")
    plt.plot(gens, mx, label="max")
    plt.xlabel("Поколение")
    plt.ylabel("Лучшее значение f")
    plt.title(f"Лучшее значение по поколениям: {name}")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"traj_{name}.png", dpi=150)
    plt.close()


def plot_compare(traj_by_config: dict, outdir: Path) -> None:
    """Строит график сравнения средних траекторий."""
    plt.figure(figsize=(8, 4.5))
    for name, traj_list in traj_by_config.items():
        arr = np.array([t[:, 2] for t in traj_list], dtype=float)
        gens = np.array(traj_list[0][:, 0], dtype=int)
        mean = arr.mean(axis=0)
        plt.plot(gens, mean, label=name)
    plt.xlabel("Поколение")
    plt.ylabel("Среднее лучшее значение f")
    plt.title("Сравнение конфигураций ГА")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "compare_mean.png", dpi=150)
    plt.close()


# ----------------------------- main -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=20, help="число независимых запусков")
    parser.add_argument("--budget", type=int, default=30000, help="бюджет вычислений fitness-функции")
    parser.add_argument("--outdir", type=str, default="results", help="каталог результатов")
    parser.add_argument("--seed", type=int, default=2026, help="базовый seed")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)


    configs = [
        GAConfig("base_pop50_mut0.1", population_size=50, mutation_prob=0.1),
        GAConfig("pop100_mut0.1", population_size=100, mutation_prob=0.1),
        GAConfig("base_pop50_mut0.3", population_size=50, mutation_prob=0.3),
    ]

    rows = []
    traj_by_config = {}

    for ci, cfg in enumerate(configs):
        traj_list = []
        for run in range(args.runs):
            t0 = time.time()
            best_fit, best_x, evals, gens, traj = run_ga(
                cfg, seed=args.seed + ci * 1000 + run, budget=args.budget
            )
            elapsed = time.time() - t0

            rows.append({
                "config": cfg.name,
                "run": run + 1,
                "best_fitness": float(best_fit),
                "best_x": json.dumps(best_x.tolist()),
                "evals": int(evals),
                "generations": int(gens),
                "time_sec": round(elapsed, 6),
            })
            traj_list.append(np.array(traj))

        traj_by_config[cfg.name] = traj_list
        plot_config(cfg.name, traj_list, outdir)

    # Случайный поиск при том же бюджете
    for run in range(args.runs):
        t0 = time.time()
        best_fit, best_x, evals = run_random_search(seed=args.seed + 9000 + run, budget=args.budget)#сдвиг по сиду, так как надо убрать зависимость от него.
        elapsed = time.time() - t0

        rows.append({
            "config": "random_search",
            "run": run + 1,
            "best_fitness": float(best_fit),
            "best_x": json.dumps(best_x.tolist()),
            "evals": int(evals),
            "generations": 0,
            "time_sec": round(elapsed, 6),
        })

    plot_compare(traj_by_config, outdir)

    write_csv(
        outdir / "runs.csv",
        rows,
        ["config", "run", "best_fitness", "best_x", "evals", "generations", "time_sec"],
    )

    summary = summarize(rows)
    write_csv(outdir / "summary.csv", summary, ["config", "runs", "best", "mean", "median", "std", "worst"])

    traj_rows = []
    for name, traj_list in traj_by_config.items():
        for run, t in enumerate(traj_list, start=1):
            for generation, evals, best_fit in t:
                traj_rows.append({
                    "config": name,
                    "run": run,
                    "generation": int(generation),
                    "evals": int(evals),
                    "best_fitness": float(best_fit),
                })
    write_csv(outdir / "trajectories.csv", traj_rows, ["config", "run", "generation", "evals", "best_fitness"])

    print(f"Готово. Результаты сохранены в {outdir.resolve()}")
    for s in summary:
        print(
            f"{s['config']}: best={s['best']:.6g}, mean={s['mean']:.6g}, "
            f"median={s['median']:.6g}, std={s['std']:.6g}, worst={s['worst']:.6g}"
        )


if __name__ == "__main__":
    main()