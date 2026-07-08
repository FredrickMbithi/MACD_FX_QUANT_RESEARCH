from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np
import pandas as pd


def _require_hmmlearn():
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError as exc:
        raise ImportError(
            "H002 requires hmmlearn for Gaussian HMM estimation. "
            "Install it with: python -m pip install hmmlearn"
        ) from exc
    return GaussianHMM


def logsumexp(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    vmax = np.max(values, axis=axis, keepdims=True)
    stable = np.log(np.sum(np.exp(values - vmax), axis=axis, keepdims=True)) + vmax
    if axis is None:
        return stable.squeeze()
    return np.squeeze(stable, axis=axis)


@dataclass
class HMMFitResult:
    model: object
    n_states: int
    covariance_type: str
    seed: int
    log_likelihood: float
    aic: float
    bic: float
    transition_matrix: np.ndarray

    @property
    def persistence(self) -> np.ndarray:
        return np.diag(self.transition_matrix)

    @property
    def expected_durations(self) -> np.ndarray:
        return 1.0 / np.maximum(1.0 - self.persistence, 1e-12)


def parameter_count(n_states: int, n_features: int, covariance_type: str) -> int:
    start_probs = n_states - 1
    transitions = n_states * (n_states - 1)
    means = n_states * n_features
    if covariance_type == "diag":
        covars = n_states * n_features
    elif covariance_type == "full":
        covars = n_states * n_features * (n_features + 1) // 2
    elif covariance_type == "spherical":
        covars = n_states
    elif covariance_type == "tied":
        covars = n_features * (n_features + 1) // 2
    else:
        raise ValueError(f"Unsupported covariance_type: {covariance_type}")
    return start_probs + transitions + means + covars


def fit_gaussian_hmm(
    x_train: pd.DataFrame | np.ndarray,
    n_states: int,
    covariance_type: str = "diag",
    n_init: int = 20,
    max_iter: int = 500,
    tol: float = 1e-4,
    random_seed: int = 42,
) -> HMMFitResult:
    GaussianHMM = _require_hmmlearn()
    x = np.asarray(x_train, dtype=float)
    best = None

    for offset in range(n_init):
        seed = random_seed + offset
        model = GaussianHMM(
            n_components=n_states,
            covariance_type=covariance_type,
            n_iter=max_iter,
            tol=tol,
            random_state=seed,
            implementation="log",
        )
        model.fit(x)
        log_likelihood = float(model.score(x))
        n_params = parameter_count(n_states, x.shape[1], covariance_type)
        aic = 2 * n_params - 2 * log_likelihood
        bic = np.log(len(x)) * n_params - 2 * log_likelihood
        result = HMMFitResult(
            model=model,
            n_states=n_states,
            covariance_type=covariance_type,
            seed=seed,
            log_likelihood=log_likelihood,
            aic=float(aic),
            bic=float(bic),
            transition_matrix=np.asarray(model.transmat_, dtype=float),
        )
        if best is None or result.log_likelihood > best.log_likelihood:
            best = result

    return best


def model_selection_grid(x_train: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rows = []
    hmm_cfg = cfg["hmm"]
    for n_states in hmm_cfg["state_counts"]:
        for covariance_type in hmm_cfg["covariance_types"]:
            result = fit_gaussian_hmm(
                x_train,
                n_states=n_states,
                covariance_type=covariance_type,
                n_init=hmm_cfg["n_init"],
                max_iter=hmm_cfg["max_iter"],
                tol=hmm_cfg["tol"],
                random_seed=hmm_cfg["random_seed"],
            )
            rows.append(
                {
                    "n_states": result.n_states,
                    "covariance_type": result.covariance_type,
                    "seed": result.seed,
                    "log_likelihood": result.log_likelihood,
                    "aic": result.aic,
                    "bic": result.bic,
                    "min_persistence": float(result.persistence.min()),
                    "mean_persistence": float(result.persistence.mean()),
                    "min_expected_duration": float(result.expected_durations.min()),
                    "fit": result,
                }
            )
    return pd.DataFrame(rows).sort_values(["bic", "aic"]).reset_index(drop=True)


def select_model(selection: pd.DataFrame, min_persistence: float = 0.70) -> HMMFitResult:
    persistent = selection[selection["min_persistence"] >= min_persistence]
    if not persistent.empty:
        return persistent.iloc[0]["fit"]
    return selection.iloc[0]["fit"]


def filtered_state_probabilities(model, x: pd.DataFrame | np.ndarray) -> np.ndarray:
    observations = np.asarray(x, dtype=float)
    log_likelihood = model._compute_log_likelihood(observations)
    log_startprob = np.log(np.maximum(model.startprob_, 1e-300))
    log_transmat = np.log(np.maximum(model.transmat_, 1e-300))

    log_alpha = np.empty_like(log_likelihood)
    log_alpha[0] = log_startprob + log_likelihood[0]
    log_alpha[0] -= logsumexp(log_alpha[0])

    for t in range(1, len(observations)):
        predicted = logsumexp(log_alpha[t - 1][:, None] + log_transmat, axis=0)
        log_alpha[t] = predicted + log_likelihood[t]
        log_alpha[t] -= logsumexp(log_alpha[t])

    return np.exp(log_alpha)


def save_fit(result: HMMFitResult, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(result, handle)


def load_fit(path: str | Path) -> HMMFitResult:
    with open(path, "rb") as handle:
        return pickle.load(handle)
