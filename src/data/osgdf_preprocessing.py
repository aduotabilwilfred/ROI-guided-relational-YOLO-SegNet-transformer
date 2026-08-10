from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np
from scipy.signal import savgol_coeffs

try:  # optional, only needed for the blind fitness guard term
    from skimage.metrics import structural_similarity as _ssim

    _HAVE_SSIM = True
except ImportError:  # pragma: no cover
    _HAVE_SSIM = False


# 1. SAVITZKY-GOLAY FILTERING

# Kernels are tiny and there are at most 27 of them, so cache aggressively.
_KERNEL_CACHE: dict[tuple[int, int, str], np.ndarray] = {}


def validate_sg_params(window_size: float, poly_order: float) -> tuple[int, int]:
    """Coerce (window_size, poly_order) into a legal Savitzky-Golay pair.

    Rules enforced:
      * window_size is an odd integer >= 3
      * poly_order >= 0
      * poly_order <= window_size - 1  (otherwise the fit is under-determined)
    """
    window_size = round(window_size)
    poly_order = round(poly_order)

    if window_size % 2 == 0:
        window_size += 1
    window_size = max(3, window_size)
    poly_order = max(0, poly_order)

    poly_order = min(poly_order, window_size - 1)

    return window_size, poly_order


def sg2d_kernel(window_size: int, poly_order: int) -> np.ndarray:
    """Build a true 2D Savitzky-Golay smoothing kernel.

    A bivariate polynomial of total degree `poly_order` is fitted by least
    squares to every `window_size` x `window_size` neighbourhood, and the
    smoothed output is the value of that polynomial at the window centre.
    Because the fit is linear in the pixel values, the whole operation reduces
    to a single fixed convolution kernel, computed once here.

    At the centre (u, v) = (0, 0) every monomial u^i v^j vanishes except the
    constant term, so the smoothed centre value is exactly the constant
    coefficient - i.e. the first row of the pseudo-inverse of the design
    matrix.

    Returns
    np.ndarray of shape (window_size, window_size), summing to 1.0.
    """
    window_size, poly_order = validate_sg_params(window_size, poly_order)
    key = (window_size, poly_order, "2d")
    cached = _KERNEL_CACHE.get(key)
    if cached is not None:
        return cached

    half = window_size // 2
    coords = np.arange(-half, half + 1, dtype=np.float64)
    u, v = np.meshgrid(coords, coords, indexing="ij")
    u = u.ravel()
    v = v.ravel()

    # Design matrix: all monomials u^i * v^j with i + j <= poly_order.
    columns = []
    for total_degree in range(poly_order + 1):
        for i in range(total_degree + 1):
            j = total_degree - i
            columns.append((u**i) * (v**j))
    design = np.stack(columns, axis=1)

    # Row 0 of the pseudo-inverse maps pixel values -> constant coefficient.
    kernel = np.linalg.pinv(design)[0].reshape(window_size, window_size)

    # Guard against tiny numerical drift; a smoothing kernel must preserve DC.
    kernel = kernel / kernel.sum()

    _KERNEL_CACHE[key] = kernel
    return kernel


def canonical_poly_order(poly_order: int) -> int:
    """Map a polynomial order to its equivalence-class representative.

    Savitzky-Golay smoothing with degree 2m and degree 2m+1 gives identical
    kernels, so 3 -> 2, 5 -> 4, and so on. Used for caching and for honest
    reporting of which orders the optimiser could actually distinguish.
    """
    poly_order = int(poly_order)
    return poly_order - 1 if poly_order % 2 == 1 else poly_order


def equivalent_poly_order(
    poly_order: int, valid_range: tuple[int, int] = (2, 4)
) -> int | None:
    """Return the other polynomial order giving an identical kernel, if any.

    Only returns a partner that actually lies inside `valid_range`, so callers
    are not told about an "equivalent order 5" when the search space stops at
    4 - true but useless.
    """
    canonical = canonical_poly_order(poly_order)
    partner = canonical if canonical != poly_order else poly_order + 1
    low, high = valid_range
    return partner if low <= partner <= high else None


def format_poly_order(poly_order: int, valid_range: tuple[int, int] = (2, 4)) -> str:
    """Human-readable order that makes the equivalence explicit when relevant."""
    partner = equivalent_poly_order(poly_order, valid_range)
    if partner is None:
        return str(poly_order)
    return f"{poly_order} (identical to order {partner})"


def assert_order_equivalence(
    window_sizes: Iterable[int] = (5, 9, 15, 21), tol: float = 1e-12
) -> bool:
    """Numerically verify the order-2/order-3 kernel equivalence.

    Returns True if every tested pair of equivalent orders produces identical
    kernels. Handy to run once and cite in the write-up.
    """
    for window in window_sizes:
        if np.abs(sg2d_kernel(window, 2) - sg2d_kernel(window, 3)).max() > tol:
            return False
    return True


def sg_separable_kernel(window_size: int, poly_order: int) -> np.ndarray:
    """Build the separable (row-then-column) Savitzky-Golay kernel.

    This is the outer product of the classic 1D SG smoothing coefficients with
    themselves. It is what you get by applying `scipy.signal.savgol_filter`
    along rows and then along columns. It is not identical to the true 2D fit
    (it implicitly includes cross terms up to degree 2*poly_order), but it is
    the variant most papers actually use when they say "Savitzky-Golay applied
    to images", so it is provided for comparison.
    """
    window_size, poly_order = validate_sg_params(window_size, poly_order)
    key = (window_size, poly_order, "separable")
    cached = _KERNEL_CACHE.get(key)
    if cached is not None:
        return cached

    coeffs_1d = savgol_coeffs(window_size, poly_order, deriv=0, use="conv")
    kernel = np.outer(coeffs_1d, coeffs_1d)
    kernel = kernel / kernel.sum()

    _KERNEL_CACHE[key] = kernel
    return kernel


def apply_sg_filter(
    image: np.ndarray,
    window_size: int,
    poly_order: int,
    mode: str = "2d",
    clip: bool = True,
) -> np.ndarray:
    """Apply the Savitzky-Golay filter to a single-channel image.

    Parameters
    ----------
    image : 2D float array, expected in [0, 1].
    window_size : odd int in [5, 21] for this project.
    poly_order : int in [2, 4] for this project.
    mode : "2d" for the true bivariate fit, "separable" for row-then-column.
    clip : clamp the result back into [0, 1]. Savitzky-Golay kernels have
        negative lobes, so mild overshoot at strong edges is expected and
        normal - clipping simply keeps the output in valid image range.

    Border handling uses BORDER_REFLECT_101, which avoids the dark halo that
    zero padding would introduce around the image frame.
    """
    if image.ndim != 2:
        raise ValueError(f"expected a 2D single-channel image, got shape {image.shape}")

    kernel = (
        sg2d_kernel(window_size, poly_order)
        if mode == "2d"
        else sg_separable_kernel(window_size, poly_order)
    )

    # float64 throughout: at the window sizes used here it costs essentially
    # nothing versus float32, and it keeps the polynomial reproduction exact
    # to machine precision rather than ~1e-7.
    src = np.ascontiguousarray(image, dtype=np.float64)
    out = cv2.filter2D(
        src, ddepth=cv2.CV_64F, kernel=kernel, borderType=cv2.BORDER_REFLECT_101
    )

    if clip:
        out = np.clip(out, 0.0, 1.0)
    return out


# 2. NOISE AND SNR METRICS


def estimate_noise_sigma(image: np.ndarray) -> float:
    """Immerkaer's fast no-reference noise standard deviation estimator.

    Convolves with a Laplacian-like mask that annihilates locally linear and
    quadratic image structure, so the response is dominated by noise. See
    Immerkaer, "Fast Noise Variance Estimation", CVIU 64(2), 1996.

    Returns sigma in the same units as the input (so [0, 1] for normalised
    images).
    """
    h, w = image.shape[:2]
    if h < 3 or w < 3:
        return 0.0

    mask = np.array(
        [[1.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 1.0]], dtype=np.float32
    )

    src = np.ascontiguousarray(image, dtype=np.float32)
    response = cv2.filter2D(src, cv2.CV_32F, mask, borderType=cv2.BORDER_REFLECT_101)
    # Discard the 1-pixel border, where the response is contaminated.
    response = response[1:-1, 1:-1]

    sigma = (
        np.sum(np.abs(response)) * math.sqrt(0.5 * math.pi) / (6.0 * (w - 2) * (h - 2))
    )
    return float(sigma)


def snr_blind(image: np.ndarray, eps: float = 1e-12) -> float:
    """No-reference SNR in dB: 10*log10(mean_signal^2 / sigma_noise^2).

    This is the metric to use for the before/after table in the report (the
    "21.4 dB -> 29.6 dB" style result). It must NOT be used on its own as an
    optimisation objective - see the module docstring.
    """
    sigma = estimate_noise_sigma(image)
    signal = float(np.mean(image))
    return 10.0 * math.log10((signal**2 + eps) / (sigma**2 + eps))


def snr_reference(reference: np.ndarray, test: np.ndarray, eps: float = 1e-12) -> float:
    """Reference-based SNR in dB: 10*log10(sum(ref^2) / sum((ref - test)^2))."""
    reference = np.asarray(reference, dtype=np.float64)
    test = np.asarray(test, dtype=np.float64)
    signal_power = float(np.sum(reference**2))
    noise_power = float(np.sum((reference - test) ** 2))
    return 10.0 * math.log10((signal_power + eps) / (noise_power + eps))


def psnr(
    reference: np.ndarray, test: np.ndarray, data_range: float = 1.0, eps: float = 1e-12
) -> float:
    """Peak SNR in dB, for images already scaled to [0, data_range]."""
    mse = float(
        np.mean((np.asarray(reference, np.float64) - np.asarray(test, np.float64)) ** 2)
    )
    return 10.0 * math.log10((data_range**2) / (mse + eps))


def ssim(reference: np.ndarray, test: np.ndarray, data_range: float = 1.0) -> float:
    """Structural similarity index; requires scikit-image."""
    if not _HAVE_SSIM:
        raise ImportError(
            "scikit-image is required for SSIM (pip install scikit-image)"
        )
    return float(_ssim(reference, test, data_range=data_range))


def _gradient_magnitude(image: np.ndarray) -> np.ndarray:
    """Sobel gradient magnitude, used by the edge preservation metric."""
    src = np.float32(image)
    gx = cv2.Sobel(src, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(src, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def edge_preservation_index(
    reference: np.ndarray, test: np.ndarray, eps: float = 1e-12
) -> float:
    """Edge preservation index: correlation of Sobel gradient magnitudes.

    Values near 1.0 mean edges survived filtering; values near 0 mean the
    filter destroyed structural detail. A useful sanity check alongside SNR,
    since SNR alone rewards over-smoothing.

    A Sobel (first-derivative) operator is used deliberately. An earlier
    Laplacian-based version of this metric was effectively unusable here: the
    second derivative amplifies high frequencies, so on a noisy X-ray the
    reference Laplacian is dominated by noise rather than anatomy, and any
    successful denoiser scores near zero. On the test data the Laplacian
    variant reported 0.025 where the Sobel variant reported 0.64 for the same
    pair of images.

    CAVEAT when interpreting the reported numbers: `reference` is the raw,
    still-noisy input image, so part of the "edge energy" being compared
    against is noise that the filter is *supposed* to remove. Treat the value
    as a conservative lower bound on structure retention. Scored against a
    denoised pseudo-ground-truth instead, the same filter measured 0.81.
    """
    grad_ref = _gradient_magnitude(reference)
    grad_test = _gradient_magnitude(test)
    a = grad_ref - grad_ref.mean()
    b = grad_test - grad_test.mean()
    denom = math.sqrt(float(np.sum(a * a)) * float(np.sum(b * b))) + eps
    return float(np.sum(a * b) / denom)


# 3. TUNICATE SWARM OPTIMIZATION (TSA / TSO)


@dataclass
class TSOHistory:
    """Per-iteration record of the optimisation run."""

    iteration: list[int] = field(default_factory=list)
    best_fitness: list[float] = field(default_factory=list)
    mean_fitness: list[float] = field(default_factory=list)
    best_position: list[list[float]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class TunicateSwarmOptimizer:
    """Tunicate Swarm Algorithm (Kaur et al., 2020).

    The algorithm models two tunicate behaviours: jet propulsion (avoiding
    conflicts between agents, moving toward the best neighbour, converging on
    the food source) and swarm behaviour (averaging with the previous
    position).

    Governing equations, as published:

        F     = 2 * c1                        (water flow advection)
        G     = c2 + c3 - F                   (gravity force)
        M     = Pmin + c1 * (Pmax - Pmin)     (social force, Pmin=1, Pmax=4)
        A     = G / M                         (conflict avoidance vector)
        PD    = |FS - r * P(x)|               (distance to food source)
        P(x)  = FS + A * PD   if rand >= 0.5
              = FS - A * PD   if rand <  0.5  (converge on food source)
        P(x+1) = (P(x) + P(x+1)) / (2 + c1)   (swarm behaviour)

    where FS is the best position found so far (the "food source") and
    c1, c2, c3, r ~ U(0, 1).

    KNOWN QUIRK: the published swarm-behaviour step divides by (2 + c1) > 2,
    which contracts positions toward the coordinate origin rather than toward
    the swarm. On a search space that does not contain the origin this biases
    the search toward the lower bounds. Set `normalize=True` (default) to run
    the search in a unit hypercube and map to real bounds, which is the usual
    workaround; set it to False to reproduce the raw published behaviour.

    Parameters
    ----------
    objective : callable position -> float. MAXIMISED.
    bounds : sequence of (low, high) per dimension.
    n_agents : population size (roadmap: 20-30).
    n_iterations : iteration budget (roadmap: 50-100).
    """

    def __init__(
        self,
        objective: Callable[[np.ndarray], float],
        bounds: Sequence[tuple[float, float]],
        n_agents: int = 25,
        n_iterations: int = 60,
        p_min: float = 1.0,
        p_max: float = 4.0,
        normalize: bool = True,
        seed: int | None = None,
        verbose: bool = True,
        callback: Callable[[int, np.ndarray, float], None] | None = None,
    ):
        self.objective = objective
        self.bounds = np.asarray(bounds, dtype=np.float64)
        if self.bounds.ndim != 2 or self.bounds.shape[1] != 2:
            raise ValueError("bounds must have shape (n_dims, 2)")
        self.n_dims = self.bounds.shape[0]
        self.n_agents = int(n_agents)
        self.n_iterations = int(n_iterations)
        self.p_min = float(p_min)
        self.p_max = float(p_max)
        self.normalize = bool(normalize)
        self.rng = np.random.default_rng(seed)
        self.verbose = verbose
        self.callback = callback

        self.history = TSOHistory()
        self.best_position: np.ndarray | None = None
        self.best_fitness: float = -np.inf
        self.n_evaluations: int = 0

    # coordinate mapping

    def _to_real(self, pos: np.ndarray) -> np.ndarray:
        if not self.normalize:
            return pos
        low, high = self.bounds[:, 0], self.bounds[:, 1]
        return low + pos * (high - low)

    def _search_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        if self.normalize:
            return np.zeros(self.n_dims), np.ones(self.n_dims)
        return self.bounds[:, 0].copy(), self.bounds[:, 1].copy()

    # main loop
    def optimize(self) -> tuple[np.ndarray, float]:
        low, high = self._search_bounds()

        positions = self.rng.uniform(low, high, size=(self.n_agents, self.n_dims))
        fitness = np.empty(self.n_agents, dtype=np.float64)

        for i in range(self.n_agents):
            fitness[i] = self._evaluate(positions[i])

        best_idx = int(np.argmax(fitness))
        food_source = positions[best_idx].copy()
        self.best_fitness = float(fitness[best_idx])
        self.best_position = self._to_real(food_source).copy()

        for iteration in range(1, self.n_iterations + 1):
            for i in range(self.n_agents):
                c1 = self.rng.random()
                c2 = self.rng.random(self.n_dims)
                c3 = self.rng.random(self.n_dims)

                water_flow = 2.0 * c1  # F
                gravity = c2 + c3 - water_flow  # G
                social = self.p_min + c1 * (self.p_max - self.p_min)  # M
                avoidance = gravity / social  # A

                r = self.rng.random(self.n_dims)
                distance = np.abs(food_source - r * positions[i])  # PD

                if self.rng.random() >= 0.5:
                    candidate = food_source + avoidance * distance
                else:
                    candidate = food_source - avoidance * distance

                # Swarm behaviour: average with the agent's previous position.
                new_position = (positions[i] + candidate) / (2.0 + c1)
                new_position = np.clip(new_position, low, high)

                new_fitness = self._evaluate(new_position)

                # Greedy selection keeps the run monotone; without it the
                # published update can wander off good solutions entirely.
                if new_fitness >= fitness[i]:
                    positions[i] = new_position
                    fitness[i] = new_fitness

            best_idx = int(np.argmax(fitness))
            if fitness[best_idx] > self.best_fitness:
                self.best_fitness = float(fitness[best_idx])
                food_source = positions[best_idx].copy()
                self.best_position = self._to_real(food_source).copy()

            self.history.iteration.append(iteration)
            self.history.best_fitness.append(self.best_fitness)
            self.history.mean_fitness.append(float(np.mean(fitness)))
            self.history.best_position.append([float(v) for v in self.best_position])

            if self.callback is not None:
                self.callback(iteration, self.best_position, self.best_fitness)

            if self.verbose and (
                iteration % 5 == 0 or iteration == 1 or iteration == self.n_iterations
            ):
                pos_str = ", ".join(f"{v:.3f}" for v in self.best_position)
                print(
                    f"  [TSO] iter {iteration:3d}/{self.n_iterations}  "
                    f"best={self.best_fitness:.4f}  pos=({pos_str})  "
                    f"evals={self.n_evaluations}"
                )

        return self.best_position, self.best_fitness

    def _evaluate(self, position: np.ndarray) -> float:
        self.n_evaluations += 1
        return float(self.objective(self._to_real(position)))


# 4. OSGDF OBJECTIVE FUNCTIONS

FITNESS_PSEUDO_GT = "pseudo_gt"
FITNESS_SYNTHETIC = "synthetic"
FITNESS_BLIND = "blind"


def build_pseudo_ground_truth(
    image: np.ndarray, noise_sigma: float | None = None, strength: float = 1.1
) -> np.ndarray:
    """Denoise an image with non-local means to serve as a tuning target.

    NLM is far stronger than a Savitzky-Golay filter and preserves edges well,
    so it is a reasonable stand-in for the unobtainable clean image. The
    filter strength `h` is derived from the measured noise level so the target
    is neither under- nor over-denoised.

    This is a proxy, not truth - see the module docstring.
    """
    if noise_sigma is None:
        noise_sigma = estimate_noise_sigma(image)

    u8 = np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    h = float(np.clip(255.0 * noise_sigma * strength, 3.0, 30.0))
    denoised = cv2.fastNlMeansDenoising(
        u8, None, h=h, templateWindowSize=7, searchWindowSize=21
    )
    return denoised.astype(np.float64) / 255.0


WINDOW_BOUNDS = (5.0, 21.0)
POLY_BOUNDS = (2.0, 4.0)
OSGDF_BOUNDS = [WINDOW_BOUNDS, POLY_BOUNDS]


@dataclass
class OSGDFObjective:
    """Fitness function over (window_size, poly_order) for SG denoising.

    Results are memoised per distinct filter, so the optimiser can request
    thousands of evaluations while doing at most 18 real ones.

    Parameters
    ----------
    images : list of 2D float arrays in [0, 1] - the calibration subset.
    fitness : FITNESS_PSEUDO_GT, FITNESS_SYNTHETIC or FITNESS_BLIND.
    noise_sigma : measured noise level in [0, 1] units, used to calibrate the
        pseudo-GT denoiser strength or the synthetic noise level. If None, it
        is estimated from the calibration images themselves.
    ssim_weight : weight of the SSIM guard term in FITNESS_BLIND mode.
    filter_mode : "2d" or "separable".
    """

    images: list[np.ndarray]
    fitness: str = FITNESS_PSEUDO_GT
    noise_sigma: float | None = None
    ssim_weight: float = 20.0
    filter_mode: str = "2d"
    seed: int = 42

    def __post_init__(self):
        if not self.images:
            raise ValueError("OSGDFObjective needs at least one calibration image")

        self.cache: dict[tuple[int, int], float] = {}
        self.real_evaluations = 0
        self.references: list[np.ndarray] | None = None
        self.noisy_images: list[np.ndarray] | None = None

        if self.fitness == FITNESS_PSEUDO_GT:
            if self.noise_sigma is None:
                sigmas = [estimate_noise_sigma(im) for im in self.images]
                self.noise_sigma = max(float(np.median(sigmas)), 1e-3)
            self.references = [
                build_pseudo_ground_truth(im, self.noise_sigma) for im in self.images
            ]
        elif self.fitness == FITNESS_SYNTHETIC:
            if self.noise_sigma is None:
                sigmas = [estimate_noise_sigma(im) for im in self.images]
                self.noise_sigma = float(np.median(sigmas))
                # Guard against a degenerate estimate on very clean images.
                self.noise_sigma = max(self.noise_sigma, 1e-3)

            # Fixed noise realisation -> deterministic, comparable fitness.
            rng = np.random.default_rng(self.seed)
            self.noisy_images = [
                np.clip(im + rng.normal(0.0, self.noise_sigma, im.shape), 0.0, 1.0)
                for im in self.images
            ]
        elif self.fitness == FITNESS_BLIND:
            if self.ssim_weight > 0 and not _HAVE_SSIM:
                raise ImportError(
                    "FITNESS_BLIND with ssim_weight > 0 requires scikit-image"
                )
            self.noisy_images = None
        else:
            raise ValueError(f"unknown fitness mode: {self.fitness!r}")

    # public API
    def __call__(self, position: Sequence[float]) -> float:
        window, order = validate_sg_params(position[0], position[1])
        window = int(np.clip(window, WINDOW_BOUNDS[0], WINDOW_BOUNDS[1]))
        if window % 2 == 0:
            window -= 1
        order = int(np.clip(order, POLY_BOUNDS[0], POLY_BOUNDS[1]))

        # Orders 2 and 3 are the same filter, so cache on the canonical order.
        key = (window, canonical_poly_order(order))
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        score = self._score(window, order)
        self.cache[key] = score
        self.real_evaluations += 1
        return score

    def _score(self, window: int, order: int) -> float:
        if self.fitness == FITNESS_PSEUDO_GT:
            # Filter the REAL noisy image; score against the NLM pseudo-truth.
            scores = [
                snr_reference(
                    reference,
                    apply_sg_filter(image, window, order, mode=self.filter_mode),
                )
                for image, reference in zip(self.images, self.references)
            ]
            return float(np.mean(scores))

        if self.fitness == FITNESS_SYNTHETIC:
            scores = [
                snr_reference(
                    clean, apply_sg_filter(noisy, window, order, mode=self.filter_mode)
                )
                for clean, noisy in zip(self.images, self.noisy_images)
            ]
            return float(np.mean(scores))

        # FITNESS_BLIND
        scores = []
        for image in self.images:
            filtered = apply_sg_filter(image, window, order, mode=self.filter_mode)
            value = snr_blind(filtered)
            if self.ssim_weight > 0:
                value += self.ssim_weight * ssim(image, filtered)
            scores.append(value)
        return float(np.mean(scores))


def grid_search(
    objective: OSGDFObjective,
    window_values: Iterable[int] | None = None,
    poly_values: Iterable[int] | None = None,
    verbose: bool = True,
) -> tuple[tuple[int, int], float, list[dict]]:
    """Exhaustively evaluate the discrete search space.

    Use this to verify that TSO actually reached the global optimum. Since the
    objective is memoised, running the grid after TSO is nearly free.

    Returns ((best_window, best_order), best_fitness, all_results).
    """
    if window_values is None:
        window_values = range(int(WINDOW_BOUNDS[0]), int(WINDOW_BOUNDS[1]) + 1, 2)
    if poly_values is None:
        poly_values = range(int(POLY_BOUNDS[0]), int(POLY_BOUNDS[1]) + 1)

    results: list[dict] = []
    seen: set = set()
    best_key: tuple[int, int] = (0, 0)
    best_value = -np.inf

    for window in window_values:
        for order in poly_values:
            value = objective([window, order])
            seen.add((int(window), canonical_poly_order(order)))
            results.append(
                {
                    "window_size": int(window),
                    "poly_order": int(order),
                    "canonical_poly_order": canonical_poly_order(order),
                    "fitness": float(value),
                }
            )
            if value > best_value:
                best_value = float(value)
                best_key = (int(window), int(order))

    if verbose:
        print(
            f"  [grid] {len(results)} parameter combinations = "
            f"{len(seen)} distinct filters (orders 2 and 3 are identical)"
        )
        print(
            f"  [grid] best = window {best_key[0]}, order {best_key[1]} "
            f"(fitness {best_value:.4f})"
        )

    return best_key, best_value, results


def optimize_sg_parameters(
    images: list[np.ndarray],
    n_agents: int = 25,
    n_iterations: int = 60,
    fitness: str = FITNESS_PSEUDO_GT,
    noise_sigma: float | None = None,
    ssim_weight: float = 20.0,
    filter_mode: str = "2d",
    seed: int | None = 42,
    verbose: bool = True,
    verify_with_grid: bool = True,
) -> dict:
    """Run TSO to find the best (window_size, poly_order) for a set of images.

    Returns a dictionary with the winning parameters, the fitness value, the
    convergence history, and - if requested - the exhaustive grid results used
    to verify optimality.
    """
    objective = OSGDFObjective(
        images=images,
        fitness=fitness,
        noise_sigma=noise_sigma,
        ssim_weight=ssim_weight,
        filter_mode=filter_mode,
        seed=42 if seed is None else int(seed),
    )

    optimizer = TunicateSwarmOptimizer(
        objective=objective,
        bounds=OSGDF_BOUNDS,
        n_agents=n_agents,
        n_iterations=n_iterations,
        seed=seed,
        verbose=verbose,
    )
    best_position, best_fitness = optimizer.optimize()
    best_window, best_order = validate_sg_params(best_position[0], best_position[1])

    result = {
        "window_size": int(best_window),
        "poly_order": int(best_order),
        "fitness": float(best_fitness),
        "fitness_mode": fitness,
        "filter_mode": filter_mode,
        "noise_sigma_used": float(objective.noise_sigma)
        if getattr(objective, "noise_sigma", None) is not None
        else None,
        "n_calibration_images": len(images),
        "n_agents": int(n_agents),
        "n_iterations": int(n_iterations),
        "tso_evaluations": int(optimizer.n_evaluations),
        "unique_configs_evaluated": int(objective.real_evaluations),
        "canonical_poly_order": canonical_poly_order(best_order),
        "poly_order_caveat": (
            (
                f"Order {best_order} is mathematically identical to order "
                f"{equivalent_poly_order(best_order)}: Savitzky-Golay smoothing "
                "kernels for degree 2m and 2m+1 coincide exactly. The optimiser "
                "cannot distinguish these two orders, and neither can any "
                "downstream metric. Report the pair, not a single value."
            )
            if equivalent_poly_order(best_order) is not None
            else (
                f"Order {best_order} has no equivalent partner inside the "
                "searched range, so it is uniquely identified."
            )
        ),
        "history": optimizer.history.to_dict(),
    }

    if verify_with_grid:
        (grid_window, grid_order), grid_fitness, grid_results = grid_search(
            objective, verbose=verbose
        )
        result["grid_search"] = {
            "best_window_size": grid_window,
            "best_poly_order": grid_order,
            "best_fitness": grid_fitness,
            "results": grid_results,
        }
        result["tso_found_global_optimum"] = bool(
            (best_window, best_order) == (grid_window, grid_order)
        )
        if not result["tso_found_global_optimum"]:
            # Trust the exhaustive result - it is provably correct.
            result["window_size"] = grid_window
            result["poly_order"] = grid_order
            result["fitness"] = grid_fitness
            result["note"] = (
                "TSO did not reach the global optimum; the "
                "exhaustive grid result was used instead."
            )

    return result


# 5. IMAGE I/O HELPERS


def load_grayscale(path: str) -> np.ndarray:
    """Load an image as a float64 grayscale array in [0, 1]."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise OSError(f"could not read image: {path}")
    return image.astype(np.float64) / 255.0


def save_grayscale(path: str, image: np.ndarray) -> None:
    """Save a float image in [0, 1] as 8-bit.

    Prefer a lossless container (PNG) for the OSGDF output: re-encoding a
    denoised X-ray as JPEG would inject fresh compression artefacts and
    partially undo the filtering you just did.
    """
    out = np.clip(np.asarray(image, dtype=np.float64), 0.0, 1.0)
    out = np.rint(out * 255.0).astype(np.uint8)
    if not cv2.imwrite(str(path), out):
        raise OSError(f"could not write image: {path}")


def image_metrics(original: np.ndarray, filtered: np.ndarray) -> dict:
    """Full before/after metric bundle for one image."""
    metrics = {
        "snr_before_db": snr_blind(original),
        "snr_after_db": snr_blind(filtered),
        "noise_sigma_before": estimate_noise_sigma(original),
        "noise_sigma_after": estimate_noise_sigma(filtered),
        "psnr_vs_original_db": psnr(original, filtered),
        "edge_preservation": edge_preservation_index(original, filtered),
    }
    metrics["snr_gain_db"] = metrics["snr_after_db"] - metrics["snr_before_db"]
    if _HAVE_SSIM:
        metrics["ssim_vs_original"] = ssim(original, filtered)
    return metrics


__all__ = [
    "FITNESS_BLIND",
    "FITNESS_PSEUDO_GT",
    "FITNESS_SYNTHETIC",
    "OSGDF_BOUNDS",
    "OSGDFObjective",
    "TSOHistory",
    "TunicateSwarmOptimizer",
    "apply_sg_filter",
    "assert_order_equivalence",
    "build_pseudo_ground_truth",
    "canonical_poly_order",
    "edge_preservation_index",
    "equivalent_poly_order",
    "estimate_noise_sigma",
    "format_poly_order",
    "grid_search",
    "image_metrics",
    "load_grayscale",
    "optimize_sg_parameters",
    "psnr",
    "save_grayscale",
    "sg2d_kernel",
    "sg_separable_kernel",
    "snr_blind",
    "snr_reference",
    "ssim",
    "validate_sg_params",
]
