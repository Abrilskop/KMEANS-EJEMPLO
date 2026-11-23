from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, kmeans_plusplus
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS_2D = ("Annual Income (k$)", "Spending Score (1-100)")
FEATURE_COLUMNS_3D = ("Age", "Annual Income (k$)", "Spending Score (1-100)")
FEATURE_COLUMNS = FEATURE_COLUMNS_3D
FEATURE_LABELS = {
    "Age": "Edad",
    "Annual Income (k$)": "Ingreso anual (k$)",
    "Spending Score (1-100)": "Puntaje de gasto (1-100)",
}


@dataclass
class IterationSnapshot:
    iteration: int
    centers: np.ndarray
    labels: np.ndarray
    inertia: float
    max_center_shift: float
    changed_points: int
    is_converged: bool


def load_customers(
    source: Any = Path("Mall_Customers.csv"),
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
) -> pd.DataFrame:
    raw = pd.read_csv(source)
    missing = [column for column in feature_columns if column not in raw.columns]
    if missing:
        raise ValueError(f"Faltan columnas en el CSV: {', '.join(missing)}")
    data = raw[list(feature_columns)].dropna().copy()
    if data.empty:
        raise ValueError("El CSV no contiene filas válidas.")
    return data


def scale_features(
    data: pd.DataFrame,
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
) -> tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    matrix = scaler.fit_transform(data[list(feature_columns)])
    return matrix, scaler


def elbow_curve(
    matrix: np.ndarray,
    k_values: range = range(1, 11),
    random_state: int = 42,
) -> dict[int, float]:
    return {
        int(k): float(
            KMeans(
                n_clusters=int(k),
                init="k-means++",
                n_init=10,
                random_state=random_state,
            )
            .fit(matrix)
            .inertia_
        )
        for k in k_values
    }


def silhouette_curve(
    matrix: np.ndarray,
    k_values: range = range(2, 11),
    random_state: int = 42,
) -> dict[int, float]:
    """Silhouette promedio para cada K (requiere K >= 2: con un solo grupo no está definida)."""
    scores: dict[int, float] = {}
    for k in k_values:
        if k < 2:
            continue
        labels = KMeans(
            n_clusters=int(k),
            init="k-means++",
            n_init=10,
            random_state=random_state,
        ).fit_predict(matrix)
        scores[int(k)] = float(silhouette_score(matrix, labels))
    return scores


def choose_k_by_silhouette(scores: dict[int, float]) -> int:
    return max(scores, key=lambda k: scores[k])


def choose_k_by_elbow(inertias: dict[int, float]) -> int:
    values = sorted(inertias)
    scores = np.array([inertias[k] for k in values], dtype=float)
    if len(values) < 3:
        return values[-1]

    direction = np.array(
        [values[-1] - values[0], scores[-1] - scores[0]],
        dtype=float,
    )
    direction_length = np.linalg.norm(direction)
    if direction_length == 0:
        return values[1]

    from_origin = np.column_stack(
        [np.asarray(values, dtype=float) - values[0], scores - scores[0]]
    )
    distances = (
        np.abs(direction[0] * from_origin[:, 1] - direction[1] * from_origin[:, 0])
        / direction_length
    )
    interior = distances[1:-1]
    return values[1:-1][int(np.argmax(interior))]


def run_iterations(
    matrix: np.ndarray,
    k: int,
    random_state: int = 42,
    tolerance: float = 1e-6,
    max_iterations: int = 100,
) -> tuple[list[IterationSnapshot], np.ndarray, KMeans]:
    initial_centers, _ = kmeans_plusplus(
        matrix,
        n_clusters=k,
        random_state=random_state,
    )
    current_centers = initial_centers.copy()
    previous_labels: np.ndarray | None = None
    previous_inertia: float | None = None
    snapshots: list[IterationSnapshot] = []

    for iteration in range(1, max_iterations + 1):
        step_model = KMeans(
            n_clusters=k,
            init=current_centers,
            n_init=1,
            max_iter=1,
            random_state=random_state,
        ).fit(matrix)

        next_centers = step_model.cluster_centers_.copy()
        shift = float(np.linalg.norm(next_centers - current_centers, axis=1).max())
        changed = (
            len(matrix)
            if previous_labels is None
            else int(np.sum(step_model.labels_ != previous_labels))
        )
        converged = iteration > 1 and shift <= tolerance and changed == 0

        snapshots.append(
            IterationSnapshot(
                iteration=iteration,
                centers=next_centers,
                labels=step_model.labels_.copy(),
                inertia=float(step_model.inertia_),
                max_center_shift=shift,
                changed_points=changed,
                is_converged=converged,
            )
        )

        current_centers = next_centers
        previous_labels = step_model.labels_.copy()
        previous_inertia = float(step_model.inertia_)
        if converged:
            break

    final_model = KMeans(
        n_clusters=k,
        init=initial_centers,
        n_init=1,
        max_iter=max_iterations,
        random_state=random_state,
    ).fit(matrix)
    assert previous_inertia is not None
    return snapshots, initial_centers, final_model


def silhouette(matrix: np.ndarray, labels: np.ndarray) -> float:
    return float(silhouette_score(matrix, labels))


def _level(value: float, mean: float, std: float, band: float = 0.5) -> str:
    if value > mean + band * std:
        return "alto"
    if value < mean - band * std:
        return "bajo"
    return "medio"


def build_cluster_table(
    data: pd.DataFrame,
    matrix: np.ndarray,
    labels: np.ndarray,
    centers_scaled: np.ndarray,
    scaler: StandardScaler,
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
) -> pd.DataFrame:
    centers_original = scaler.inverse_transform(centers_scaled)
    global_means = {column: float(data[column].mean()) for column in feature_columns}
    global_stds = {column: float(data[column].std(ddof=0)) for column in feature_columns}
    rows: list[dict[str, Any]] = []

    for cluster in sorted(np.unique(labels)):
        mask = labels == cluster
        group = data.iloc[mask]
        group_points = matrix[mask]
        row: dict[str, Any] = {
            "Cluster": int(cluster),
            "Clientes": int(mask.sum()),
        }
        for index, column in enumerate(feature_columns):
            label = FEATURE_LABELS.get(column, column)
            row[f"{label} medio"] = round(float(group[column].mean()), 1)
            row[f"Centroide {label}"] = round(float(centers_original[cluster, index]), 1)

        income_label = FEATURE_LABELS["Annual Income (k$)"]
        spending_label = FEATURE_LABELS["Spending Score (1-100)"]
        if f"{income_label} medio" in row and f"{spending_label} medio" in row:
            income_profile = _level(
                row[f"{income_label} medio"],
                global_means["Annual Income (k$)"],
                global_stds["Annual Income (k$)"],
            )
            spending_profile = _level(
                row[f"{spending_label} medio"],
                global_means["Spending Score (1-100)"],
                global_stds["Spending Score (1-100)"],
            )
            row["Perfil"] = f"ingreso {income_profile} / gasto {spending_profile}"

        row["Inercia del grupo"] = round(
            float(np.sum((group_points - centers_scaled[cluster]) ** 2)), 2
        )
        rows.append(row)

    return pd.DataFrame(rows)
