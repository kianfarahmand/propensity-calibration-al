"""Dataset loading and pool construction.

PREREGISTRATION.md Section 4.1 fixes this procedure completely, in order:

    1. parse every formula with pymatgen.core.Composition
    2. featurize with matminer ElementProperty.from_preset("magpie"), 132-d,
       ignore_errors=False, rejecting any all-NaN feature row
    3. collapse duplicate reduced formulas, keeping the median target
    4. drop rows with missing or non-finite targets
    5. cap at 3000 by subsampling without replacement under default_rng(20261010)
    6. standardize features, then PCA to 95% of variance capped at 20 components

The order is the pre-registered one and is not rearranged for speed. Steps 5
and 6 use no labels, so neither leaks.

The trap this module exists to avoid is documented in docs/data_access_report.md
Section 4: ElementProperty.featurize_many accepts raw formula *strings* without
complaint when ignore_errors=True and returns an all-NaN matrix, which would not
crash the campaign - it would make every surrogate prediction identical. Hence
Composition objects everywhere and ignore_errors=False at the single call site.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import load_datasets


@dataclass
class Pool:
    """A fixed, fully-constructed candidate pool.

    Attributes
    ----------
    X : (n, d) PCA-reduced, standardized descriptors.
    y : (n,) targets, standardized and oriented so that larger is better.
    y_raw : (n,) targets in original magnitude, same orientation as y.
    y_mean, y_std : the standardization constants, for back-transforming.
    """

    name: str
    X: np.ndarray
    y: np.ndarray
    y_raw: np.ndarray
    y_mean: float
    y_std: float
    formulas: list[str]
    direction: str
    units: str
    kind: str
    n_raw_rows: int
    n_after_dedup: int
    n_components: int
    explained_variance: float
    build_seconds: float = 0.0
    provenance: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return self.X.shape[0]

    @property
    def d(self) -> int:
        return self.X.shape[1]

    def to_original_units(self, standardized_values: np.ndarray) -> np.ndarray:
        """Back-transform model-scale values to original target magnitude.

        Orientation is preserved (larger is better), so a difference computed
        here is a difference in the original units of the target, which is what
        PREREGISTRATION.md Section 11 asks regret to be reported in.
        """
        return np.asarray(standardized_values) * self.y_std + self.y_mean

    def summary(self) -> dict:
        return {
            "dataset": self.name,
            "n_raw_rows": self.n_raw_rows,
            "n_after_dedup": self.n_after_dedup,
            "n_pool": self.n,
            "n_components": self.n_components,
            "explained_variance": round(float(self.explained_variance), 6),
            "direction": self.direction,
            "kind": self.kind,
            "units": self.units,
            "y_mean": float(self.y_mean),
            "y_std": float(self.y_std),
            "build_seconds": round(self.build_seconds, 2),
        }


def _cache_path(cache_dir: Path, dataset: str, formulas: list[str]) -> Path:
    digest = hashlib.sha256("|".join(formulas).encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"{dataset}.magpie.{digest}.npz"


def featurize(formulas: list[str], cache_dir: Path | None = None,
              dataset: str = "pool") -> np.ndarray:
    """Magpie features for a list of formula strings.

    Composition objects are built here and only here. ignore_errors is passed
    explicitly as False so that a parse failure is an exception rather than a
    silent NaN row (validity test 7).
    """
    from matminer.featurizers.composition import ElementProperty
    from pymatgen.core import Composition

    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached = _cache_path(cache_dir, dataset, formulas)
        if cached.exists():
            return np.load(cached)["F"]

    compositions = [Composition(f) for f in formulas]
    if any(not isinstance(c, Composition) for c in compositions):
        raise TypeError("featurize() must be given Composition objects, never strings")

    featurizer = ElementProperty.from_preset("magpie")
    F = np.asarray(featurizer.featurize_many(compositions, ignore_errors=False, pbar=False),
                   dtype=float)

    all_nan_rows = int(np.isnan(F).all(axis=1).sum())
    if all_nan_rows:
        raise ValueError(
            f"{all_nan_rows} of {F.shape[0]} feature rows are entirely NaN. "
            "This is the failure mode of passing formula strings to the featurizer; "
            "see docs/data_access_report.md Section 4."
        )
    if cache_dir is not None:
        np.savez_compressed(_cache_path(cache_dir, dataset, formulas), F=F)
    return F


def load_frame(dataset: str, spec: dict):
    """Fetch the raw table. No dataset is ever committed to the repository."""
    from matminer.datasets import load_dataset

    if spec["loader"] != "matminer":
        raise ValueError(f"unknown loader {spec['loader']!r} for {dataset}")
    df = load_dataset(dataset)
    for column in (spec["formula_column"], spec["target_column"]):
        if column not in df.columns:
            raise KeyError(
                f"{dataset}: expected column {column!r}, got {list(df.columns)[:8]}. "
                "configs/datasets.yaml is authoritative; a loader that disagrees "
                "with it is an error, not something to adapt to."
            )
    return df


def build_pool(dataset: str, datasets_cfg: dict | None = None,
               cache_dir: str | Path | None = ".cache/pcal",
               verbose: bool = False) -> Pool:
    """Construct the pool for one dataset, exactly as pre-registered."""
    started = time.time()
    cfg = datasets_cfg or load_datasets()
    pool_cfg = cfg["pool"]
    if pool_cfg["ignore_errors"]:
        raise ValueError("configs/datasets.yaml sets pool.ignore_errors true; "
                         "validity test 7 forbids it")
    spec = cfg["datasets"][dataset]

    df = load_frame(dataset, spec)
    n_raw_rows = len(df)
    formulas = [str(v) for v in df[spec["formula_column"]].tolist()]
    targets = np.asarray(df[spec["target_column"]].to_numpy(), dtype=float)

    # 1-2. parse and featurize (Composition objects, ignore_errors=False)
    F = featurize(formulas, cache_dir=Path(cache_dir) if cache_dir else None, dataset=dataset)
    if F.shape[0] != n_raw_rows:
        raise ValueError("featurizer returned the wrong number of rows")
    if F.shape[1] != pool_cfg["n_features_raw"]:
        raise ValueError(f"expected {pool_cfg['n_features_raw']} Magpie features, "
                         f"got {F.shape[1]}")

    # 3. collapse duplicate reduced formulas, keeping the median target
    from pymatgen.core import Composition

    reduced = [Composition(f).reduced_formula for f in formulas]
    order: list[str] = []
    groups: dict[str, list[int]] = {}
    for i, key in enumerate(reduced):
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(i)

    keep_rows, collapsed_targets, kept_formulas = [], [], []
    for key in order:
        rows = groups[key]
        keep_rows.append(rows[0])
        collapsed_targets.append(float(np.median(targets[rows])))
        kept_formulas.append(key)
    Xf = F[np.asarray(keep_rows)]
    t = np.asarray(collapsed_targets, dtype=float)
    n_after_dedup = len(t)

    # 4. drop missing or non-finite targets
    finite = np.isfinite(t)
    Xf, t = Xf[finite], t[finite]
    kept_formulas = [f for f, ok in zip(kept_formulas, finite, strict=True) if ok]
    if np.isnan(Xf).all(axis=1).any():
        raise ValueError("an all-NaN feature row survived pool construction")

    # 5. cap at 3000 under the fixed subsampling seed, shared by every arm and seed
    cap = int(pool_cfg["max_pool"])
    if len(t) > cap:
        rng = np.random.default_rng(int(pool_cfg["subsample_seed"]))
        picked = np.sort(rng.choice(len(t), size=cap, replace=False))
        Xf, t = Xf[picked], t[picked]
        kept_formulas = [kept_formulas[i] for i in picked]

    # orientation: everything downstream maximizes
    direction = spec["direction"]
    if direction not in ("max", "min"):
        raise ValueError(f"direction must be max or min, got {direction!r}")
    y_raw = -t if direction == "min" else t

    # 6. standardize features, then PCA to 95% of variance capped at 20
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    Z = StandardScaler().fit_transform(Xf)
    constant = ~np.isfinite(Z).all(axis=0)
    if constant.any():
        Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    pca = PCA(n_components=None, svd_solver="full", random_state=0).fit(Z)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    k = int(np.searchsorted(cumulative, float(pool_cfg["pca_variance"])) + 1)
    k = min(k, int(pool_cfg["pca_max_components"]), Z.shape[1])
    X = pca.transform(Z)[:, :k]

    y_mean, y_std = float(np.mean(y_raw)), float(np.std(y_raw))
    if not np.isfinite(y_std) or y_std <= 0:
        raise ValueError(f"{dataset}: target has zero variance after pool construction")
    y = (y_raw - y_mean) / y_std

    pool = Pool(
        name=dataset, X=np.ascontiguousarray(X), y=y, y_raw=y_raw,
        y_mean=y_mean, y_std=y_std, formulas=kept_formulas,
        direction=direction, units=spec.get("units", ""), kind=spec["kind"],
        n_raw_rows=n_raw_rows, n_after_dedup=n_after_dedup,
        n_components=k, explained_variance=float(cumulative[k - 1]),
        build_seconds=time.time() - started,
        provenance={"expect_n": spec.get("expect_n"),
                    "subsample_seed": int(pool_cfg["subsample_seed"]),
                    "formula_column": spec["formula_column"],
                    "target_column": spec["target_column"]},
    )
    if verbose:
        print(json.dumps(pool.summary(), indent=2))
    return pool


def check_pool(dataset: str, **kwargs) -> dict:
    """Build a pool and assert the pre-registered invariants hold on real data.

    This is procedure step 3 of sessions/S04-implementation.md: the row counts
    come out as the data-access report measured them, no feature row is all-NaN,
    and the pool is capped where the plan says it is capped.
    """
    pool = build_pool(dataset, **kwargs)
    expect_n = pool.provenance.get("expect_n")
    problems = []
    if expect_n is not None and pool.n_raw_rows != expect_n:
        problems.append(f"raw rows {pool.n_raw_rows} != expected {expect_n}")
    if pool.n > 3000:
        problems.append(f"pool of {pool.n} exceeds the pre-registered cap of 3000")
    if pool.n_components > 20:
        problems.append(f"{pool.n_components} PCA components exceeds the cap of 20")
    if not np.isfinite(pool.X).all():
        problems.append("non-finite values in the feature matrix")
    if not np.isfinite(pool.y).all():
        problems.append("non-finite values in the targets")
    out = pool.summary()
    out["problems"] = problems
    out["ok"] = not problems
    return out
