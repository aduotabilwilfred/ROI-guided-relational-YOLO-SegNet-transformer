from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Param:
    """
    One searchable hyperparameter with bounds and an optional integer/log form.
    """

    name: str
    low: float
    high: float
    is_int: bool = False
    log_scale: bool = False

    def decode(self, unit: float) -> float:
        """
        Map a value in [0, 1] to the parameter's range.
        """
        unit = float(np.clip(unit, 0.0, 1.0))
        if self.log_scale:
            lo, hi = np.log10(self.low), np.log10(self.high)
            val = 10 ** (lo + unit * (hi - lo))
        else:
            val = self.low + unit * (self.high - self.low)
        if self.is_int:
            val = round(val)
        return val


@dataclass
class SearchSpace:
    params: list[Param]

    def decode(self, vector: np.ndarray) -> dict[str, float]:
        return {p.name: p.decode(vector[i]) for i, p in enumerate(self.params)}

    @property
    def dim(self) -> int:
        return len(self.params)


def default_search_space() -> SearchSpace:
    """
    The paper's stated FHEO search ranges (Eqs. text): lr 1e-5..1e-2,
    batch 8..32, dropout 0.1..0.5, embedding dim 128..512, plus reduction.
    """
    return SearchSpace(
        [
            Param("lr", 1e-5, 1e-2, log_scale=True),
            Param("batch_size", 8, 32, is_int=True),
            Param("dropout", 0.1, 0.5),
            Param("embed_dim", 128, 512, is_int=True),
            Param("reduction", 1, 8, is_int=True),
        ]
    )


@dataclass
class FHEOConfig:
    population: int = 20
    iterations: int = 50
    fire_hawk_fraction: float = 0.5
    seed: int = 42


@dataclass
class FHEOResult:
    best_vector: np.ndarray
    best_params: dict[str, float]
    best_fitness: float
    history: list[float] = field(default_factory=list)


class FHEO:
    """
    Fire Hawk + Election Optimizer over a unit hypercube [0,1]^dim.

    fitness(params) -> score to MAXIMISE (e.g. validation Dice). The optimiser
    works in normalised [0,1] space and decodes to real hyperparameters via the
    search space, so all parameters are handled uniformly regardless of scale.
    """

    def __init__(
        self,
        space: SearchSpace,
        fitness: Callable[[dict], float],
        cfg: FHEOConfig = None,
    ):
        self.space = space
        self.fitness = fitness
        self.cfg = cfg or FHEOConfig()
        self.rng = np.random.default_rng(self.cfg.seed)

    def _evaluate(self, vec: np.ndarray) -> float:
        return self.fitness(self.space.decode(vec))

    def optimise(self) -> FHEOResult:
        dim = self.space.dim
        n = self.cfg.population
        pop = self.rng.random((n, dim))
        fitness = np.array([self._evaluate(v) for v in pop])

        best_idx = int(fitness.argmax())
        best_vec = pop[best_idx].copy()
        best_fit = float(fitness[best_idx])
        history = [best_fit]

        n_fh = max(1, int(n * self.cfg.fire_hawk_fraction))

        for t in range(self.cfg.iterations):
            order = fitness.argsort()[::-1]  # best first
            fire_hawks = order[:n_fh]
            prey = order[n_fh:]
            global_best = pop[order[0]].copy()

            # Fire Hawk phase
            for i in fire_hawks:
                other = pop[self.rng.choice(fire_hawks)]
                r1, r2 = self.rng.random(), self.rng.random()
                pop[i] = pop[i] + r1 * (global_best - pop[i]) + r2 * (other - pop[i])
                pop[i] = np.clip(pop[i], 0.0, 1.0)
            for j in prey:
                # prey move toward a safe location: the mean of the fire hawks
                safe = pop[fire_hawks].mean(axis=0)
                r = self.rng.random()
                pop[j] = np.clip(pop[j] + r * (safe - pop[j]), 0.0, 1.0)

            # Election Optimizer refinement
            beta = 1.0 - (t / self.cfg.iterations) ** 2
            mean_pos = pop.mean(axis=0)
            for i in range(n):
                opponent = pop[self.rng.integers(n)]
                alpha = self.rng.random() * 2.0
                # opponent influence + mean influence, scaled by beta
                pop[i] = (
                    pop[i]
                    + self.rng.random() * (opponent - pop[i])
                    + alpha * (global_best - pop[i])
                    + beta * self.rng.random() * (mean_pos - pop[i])
                )
                pop[i] = np.clip(pop[i], 0.0, 1.0)

            fitness = np.array([self._evaluate(v) for v in pop])
            it_best = int(fitness.argmax())
            if fitness[it_best] > best_fit:
                best_fit = float(fitness[it_best])
                best_vec = pop[it_best].copy()
            history.append(best_fit)

        return FHEOResult(
            best_vector=best_vec,
            best_params=self.space.decode(best_vec),
            best_fitness=best_fit,
            history=history,
        )


def tune_relational_head(
    fold_dir,
    weights,
    base_cfg,
    device: str,
    eval_epochs: int = 10,
    cfg: FHEOConfig = None,
    space: SearchSpace = None,
) -> FHEOResult:
    """
    Run FHEO with training-based fitness on one fold.

    The fitness of a candidate is: build the head with the candidate's
    hyperparameters, train it for eval_epochs on this fold, return the fold's
    validation Dice. Reduced epochs keep each evaluation affordable; the winner
    is then trained fully by the normal training loop.

    This wraps the real training, so it needs a GPU and materialised fold data.
    The optimiser engine itself (FHEO class) is tested independently against
    benchmark functions.
    """
    from dataclasses import replace

    from train import accumulate_fold_dice, train_one_fold

    space = space or default_search_space()

    def fitness(params: dict) -> float:
        cfg_candidate = replace(
            base_cfg,
            embed_dim=int(params["embed_dim"]),
            reduction=int(params["reduction"]),
        )
        head, detector = train_one_fold(
            fold_dir,
            weights,
            cfg_candidate,
            device,
            epochs=eval_epochs,
            batch_size=int(params["batch_size"]),
            lr=float(params["lr"]),
        )
        inter, pred, gt = accumulate_fold_dice(
            head,
            detector,
            fold_dir,
            cfg_candidate,
            device,
            batch_size=int(params["batch_size"]),
        )
        return (2 * inter + 1e-7) / (pred + gt + 1e-7)

    return FHEO(space, fitness, cfg).optimise()
