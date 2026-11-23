from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

from kmeans_engine import (
    FEATURE_COLUMNS_2D,
    FEATURE_COLUMNS_3D,
    build_cluster_table,
    choose_k_by_elbow,
    elbow_curve,
    load_customers,
    run_iterations,
    scale_features,
    silhouette,
)

DATA_PATH = Path(__file__).resolve().parents[1] / "Mall_Customers.csv"


def test_load_customers():
    data = load_customers(DATA_PATH)
    assert list(data.columns) == list(FEATURE_COLUMNS_3D)
    assert len(data) == 200
    assert not data.isna().any().any()

    data_2d = load_customers(DATA_PATH, FEATURE_COLUMNS_2D)
    assert list(data_2d.columns) == list(FEATURE_COLUMNS_2D)
    matrix_2d, _ = scale_features(data_2d, FEATURE_COLUMNS_2D)
    assert matrix_2d.shape == (200, 2)


def test_elbow_selection_is_deterministic():
    matrix, _ = scale_features(load_customers(DATA_PATH))
    inertias = elbow_curve(matrix, range(1, 8), random_state=42)
    selected = choose_k_by_elbow(inertias)
    assert selected == choose_k_by_elbow(inertias)
    assert 2 <= selected <= 7
    assert inertias[1] > inertias[7]


def test_replay_matches_sklearn_final_model():
    matrix, _ = scale_features(load_customers(DATA_PATH))
    snapshots, initial_centers, final_model = run_iterations(matrix, k=5, random_state=42)
    expected = KMeans(
        n_clusters=5,
        init=initial_centers,
        n_init=1,
        random_state=42,
    ).fit(matrix)

    np.testing.assert_allclose(final_model.cluster_centers_, expected.cluster_centers_)
    np.testing.assert_array_equal(final_model.labels_, expected.labels_)
    np.testing.assert_array_equal(final_model.labels_, snapshots[-1].labels)
    assert final_model.n_iter_ == len(snapshots)
    assert snapshots[-1].max_center_shift < 1e-6
    assert snapshots[-1].is_converged


def test_centroids_are_group_means():
    matrix, _ = scale_features(load_customers(DATA_PATH))
    _, _, final_model = run_iterations(matrix, k=5, random_state=42)
    for cluster in range(5):
        members = matrix[final_model.labels_ == cluster]
        np.testing.assert_allclose(final_model.cluster_centers_[cluster], members.mean(axis=0))


def test_cluster_table_and_silhouette():
    data = load_customers(DATA_PATH)
    matrix, scaler = scale_features(data)
    _, _, final_model = run_iterations(matrix, k=5, random_state=42)
    table = build_cluster_table(
        data,
        matrix,
        final_model.labels_,
        final_model.cluster_centers_,
        scaler,
    )

    assert len(table) == 5
    assert table["Clientes"].sum() == 200
    assert "Centroide Edad" in table.columns
    assert "Centroide Ingreso anual (k$)" in table.columns
    assert "Centroide Puntaje de gasto (1-100)" in table.columns
    assert 0 < silhouette(matrix, final_model.labels_) <= 1


def test_silhouette_curve():
    from kmeans_engine import choose_k_by_silhouette, silhouette_curve

    data = load_customers(DATA_PATH, FEATURE_COLUMNS_2D)
    matrix, _ = scale_features(data, FEATURE_COLUMNS_2D)
    scores = silhouette_curve(matrix, range(1, 8), random_state=42)
    assert sorted(scores) == [2, 3, 4, 5, 6, 7]
    assert all(-1 <= value <= 1 for value in scores.values())
    assert choose_k_by_silhouette(scores) == 5
