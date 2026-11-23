from __future__ import annotations

import hashlib
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import pairwise_distances_argmin

from kmeans_engine import (
    FEATURE_COLUMNS_2D,
    FEATURE_COLUMNS_3D,
    FEATURE_LABELS,
    build_cluster_table,
    choose_k_by_elbow,
    choose_k_by_silhouette,
    elbow_curve,
    load_customers,
    run_iterations,
    scale_features,
    silhouette,
    silhouette_curve,
)

st.set_page_config(page_title="Simulador K-means", layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.4rem;}
    .stMetric {background: #f5f7fa; border-radius: 0.5rem; padding: 0.5rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def plot_elbow(inertias: dict[int, float], suggested: int, chosen: int):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    values = sorted(inertias)
    ax.plot(
        values,
        [inertias[k] for k in values],
        marker="o",
        linewidth=2.2,
        color="#1f77b4",
    )
    ax.axvline(suggested, color="#d62728", linestyle="--", linewidth=1.8, label=f"Codo: K={suggested}")
    ax.axvline(chosen, color="#2ca02c", linestyle=":", linewidth=2.0, label=f"Usado: K={chosen}")
    ax.set_xticks(values)
    ax.set_xlabel("K (número de grupos)")
    ax.set_ylabel("WCSS / inercia")
    ax.set_title("Método del codo: error según el número de grupos")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_silhouette(scores: dict[int, float], best: int, chosen: int):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    values = sorted(scores)
    ax.plot(
        values,
        [scores[k] for k in values],
        marker="o",
        linewidth=2.2,
        color="#9467bd",
    )
    ax.axvline(best, color="#ff7f0e", linestyle="--", linewidth=1.8, label=f"Mejor silhouette: K={best}")
    ax.axvline(chosen, color="#2ca02c", linestyle=":", linewidth=2.0, label=f"Usado: K={chosen}")
    ax.annotate(
        f"{scores[best]:.3f}",
        (best, scores[best]),
        textcoords="offset points",
        xytext=(0, 8),
        ha="center",
        fontsize=9,
    )
    ax.set_xticks(values)
    ax.set_xlabel("K (número de grupos)")
    ax.set_ylabel("Silhouette promedio")
    ax.set_title("Método de la silhouette: separación según el número de grupos")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_snapshot(
    data: pd.DataFrame,
    matrix: np.ndarray,
    snapshot,
    scaler,
    initial_centers: np.ndarray,
    feature_columns: tuple[str, ...],
):
    fig, ax = plt.subplots(figsize=(9, 6))
    x_min, x_max = matrix[:, 0].min() - 0.6, matrix[:, 0].max() + 0.6
    y_min, y_max = matrix[:, 1].min() - 0.6, matrix[:, 1].max() + 0.6
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 110), np.linspace(y_min, y_max, 110))
    grid_scaled = np.column_stack([xx.ravel(), yy.ravel()])
    grid_labels = pairwise_distances_argmin(grid_scaled, snapshot.centers)
    grid_original = scaler.inverse_transform(grid_scaled)

    ax.pcolormesh(
        grid_original[:, 0].reshape(xx.shape),
        grid_original[:, 1].reshape(yy.shape),
        grid_labels.reshape(xx.shape),
        shading="auto",
        cmap="tab20",
        alpha=0.12,
    )

    points = data[list(feature_columns)].to_numpy()
    ax.scatter(
        points[:, 0],
        points[:, 1],
        c=snapshot.labels,
        cmap="tab10",
        s=26,
        alpha=0.9,
        edgecolors="none",
    )

    centers_original = scaler.inverse_transform(snapshot.centers)
    initial_original = scaler.inverse_transform(initial_centers)
    step = max(1, len(points) // 45)
    for index in range(0, len(points), step):
        assigned = snapshot.labels[index]
        ax.plot(
            [points[index, 0], centers_original[assigned, 0]],
            [points[index, 1], centers_original[assigned, 1]],
            color="0.3",
            alpha=0.18,
            linewidth=0.7,
        )

    ax.scatter(
        initial_original[:, 0],
        initial_original[:, 1],
        marker="^",
        s=95,
        color="0.5",
        alpha=0.8,
        label="Inicio (k-means++)",
    )
    ax.scatter(
        centers_original[:, 0],
        centers_original[:, 1],
        marker="X",
        s=230,
        c="black",
        edgecolors="white",
        linewidths=1.5,
        label="Centroides actuales",
    )
    ax.set_xlabel(FEATURE_LABELS[feature_columns[0]])
    ax.set_ylabel(FEATURE_LABELS[feature_columns[1]])
    ax.set_title(f"Agrupaciones en la iteración {snapshot.iteration}")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    return fig


def plot_snapshot_3d(data, snapshot, scaler, initial_centers, feature_columns):
    points = data[list(feature_columns)].to_numpy()
    centers_original = scaler.inverse_transform(snapshot.centers)
    initial_original = scaler.inverse_transform(initial_centers)
    palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    fig = go.Figure()

    for cluster in sorted(np.unique(snapshot.labels)):
        mask = snapshot.labels == cluster
        fig.add_trace(
            go.Scatter3d(
                x=points[mask, 0],
                y=points[mask, 1],
                z=points[mask, 2],
                mode="markers",
                name=f"Cluster {cluster}",
                marker=dict(
                    size=4,
                    color=palette[cluster % len(palette)],
                    opacity=0.85,
                    line=dict(width=0),
                ),
                text=[f"Cliente {index + 1}" for index in np.flatnonzero(mask)],
                hovertemplate=(
                    "%{text}<br>"
                    f"{FEATURE_LABELS[feature_columns[0]]}: %{{x}}<br>"
                    f"{FEATURE_LABELS[feature_columns[1]]}: %{{y}}<br>"
                    f"{FEATURE_LABELS[feature_columns[2]]}: %{{z}}<br>"
                    f"<extra>Cluster {cluster}</extra>"
                ),
            )
        )

    sample_step = max(1, len(points) // 30)
    for index in range(0, len(points), sample_step):
        assigned = snapshot.labels[index]
        fig.add_trace(
            go.Scatter3d(
                x=[points[index, 0], centers_original[assigned, 0]],
                y=[points[index, 1], centers_original[assigned, 1]],
                z=[points[index, 2], centers_original[assigned, 2]],
                mode="lines",
                line=dict(color="#6c757d", width=2),
                opacity=0.35,
                showlegend=False,
                hoverinfo="skip",
            )
        )

    fig.add_trace(
        go.Scatter3d(
            x=initial_original[:, 0],
            y=initial_original[:, 1],
            z=initial_original[:, 2],
            mode="markers",
            name="Inicio (k-means++)",
            marker=dict(symbol="diamond", size=7, color="#6c757d"),
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=centers_original[:, 0],
            y=centers_original[:, 1],
            z=centers_original[:, 2],
            mode="markers",
            name="Centroides actuales",
            marker=dict(symbol="x", size=12, color="black", line=dict(width=4, color="black")),
        )
    )
    fig.update_layout(
        title=f"Agrupaciones 3D en la iteración {snapshot.iteration}",
        scene=dict(
            xaxis_title=FEATURE_LABELS[feature_columns[0]],
            yaxis_title=FEATURE_LABELS[feature_columns[1]],
            zaxis_title=FEATURE_LABELS[feature_columns[2]],
        ),
        height=620,
        margin=dict(l=0, r=0, b=0, t=35),
        legend=dict(orientation="h", y=1.03),
    )
    return fig


if "step_index" not in st.session_state:
    st.session_state.step_index = 0
if "playing" not in st.session_state:
    st.session_state.playing = False

st.title("Simulador visual de K-means con scikit-learn")
st.caption("Codo, centroides, agrupaciones e iteraciones explicados dentro de la misma app.")

with st.sidebar:
    st.header("Controles")
    view_mode = st.radio("Vista del simulador", ["3D", "2D"], horizontal=True)
    uploaded = st.file_uploader("CSV propio (opcional)", type=["csv"])
    seed = int(st.number_input("Semilla aleatoria", min_value=0, max_value=999, value=42, step=1))
    k_max = int(st.slider("K máximo para el codo", min_value=5, max_value=10, value=10))
    auto_k = st.checkbox("Elegir K automáticamente con el codo", value=True)

feature_columns = FEATURE_COLUMNS_3D if view_mode == "3D" else FEATURE_COLUMNS_2D
source = uploaded if uploaded is not None else Path("Mall_Customers.csv")
data = load_customers(source, feature_columns)
data_signature = (
    feature_columns,
    len(data),
    hashlib.md5(
        data.head(3).to_numpy().tobytes() + data.tail(3).to_numpy().tobytes()
    ).hexdigest(),
)

elbow_signature = (k_max, seed, data_signature)
if st.session_state.get("elbow_signature") != elbow_signature:
    matrix, scaler = scale_features(data, feature_columns)
    inertias = elbow_curve(matrix, range(1, k_max + 1), seed)
    suggested_k = choose_k_by_elbow(inertias)
    silhouettes = silhouette_curve(matrix, range(2, k_max + 1), seed)
    st.session_state.elbow = {
        "matrix": matrix,
        "scaler": scaler,
        "inertias": inertias,
        "suggested_k": suggested_k,
        "silhouettes": silhouettes,
        "silhouette_k": choose_k_by_silhouette(silhouettes),
    }
    st.session_state.elbow_signature = elbow_signature

elbow = st.session_state.elbow
matrix = elbow["matrix"]
scaler = elbow["scaler"]
inertias = elbow["inertias"]
suggested_k = elbow["suggested_k"]
silhouettes = elbow["silhouettes"]
silhouette_k = elbow["silhouette_k"]

if auto_k:
    k = suggested_k
    with st.sidebar:
        st.info(f"K elegido por el codo: {k}")
else:
    with st.sidebar:
        k = int(st.slider("K deseado", min_value=2, max_value=k_max, value=suggested_k))

run_signature = (k, seed, data_signature)
if st.session_state.get("run_signature") != run_signature:
    snapshots, initial_centers, final_model = run_iterations(matrix, k, seed)
    cluster_table = build_cluster_table(
        data,
        matrix,
        final_model.labels_,
        final_model.cluster_centers_,
        scaler,
        feature_columns,
    )
    st.session_state.run = {
        "snapshots": snapshots,
        "initial_centers": initial_centers,
        "final_model": final_model,
        "cluster_table": cluster_table,
        "silhouette": silhouette(matrix, final_model.labels_),
    }
    st.session_state.run_signature = run_signature
    st.session_state.step_index = 0
    st.session_state.playing = False

run = st.session_state.run
snapshots = run["snapshots"]
initial_centers = run["initial_centers"]
final_model = run["final_model"]
cluster_table = run["cluster_table"]

with st.sidebar:
    st.divider()
    st.subheader("Animación")
    speed = st.slider("Velocidad", min_value=1, max_value=5, value=2)
    control_columns = st.columns(4)
    if control_columns[0].button("Reproducir", width="stretch"):
        st.session_state.playing = True
        if st.session_state.step_index >= len(snapshots) - 1:
            st.session_state.step_index = 0
    if control_columns[1].button("Pausa", width="stretch"):
        st.session_state.playing = False
    if control_columns[2].button("Paso", width="stretch"):
        st.session_state.playing = False
        st.session_state.step_index = min(st.session_state.step_index + 1, len(snapshots) - 1)
    if control_columns[3].button("Reiniciar", width="stretch"):
        st.session_state.playing = False
        st.session_state.step_index = 0

if st.session_state.playing:
    time.sleep(1.0 / speed)
    if st.session_state.step_index < len(snapshots) - 1:
        st.session_state.step_index += 1
    else:
        st.session_state.playing = False
    st.rerun()

tab_codo, tab_simulacion, tab_resultado, tab_mapa = st.tabs(
    ["1. Hallar K", "2. Simulación", "3. Resultado", "4. Mapa del modelo"]
)

with tab_codo:
    st.header("¿Cómo se halló el valor de K?")
    chart_column, detail_column = st.columns([2, 1])
    with chart_column:
        figure = plot_elbow(inertias, suggested_k, k)
        st.pyplot(figure)
        plt.close(figure)
    with detail_column:
        st.metric("K sugerido por el codo", suggested_k)
        st.metric("K usado en la simulación", k)
        st.metric("WCSS con K=1", f"{inertias[1]:.1f}")
        st.metric(f"WCSS con K={k}", f"{inertias[k]:.1f}")
    st.markdown(
        f"""
        **Qué hizo scikit-learn:** entrenó un `KMeans` distinto para cada K entre 1 y {k_max},
        con 10 reinicios (`n_init=10`) para evitar malos puntos de partida. Cada punto de la
        gráfica es `inertia_`, es decir, la suma de distancias al cuadrado de cada cliente a su
        centroide asignado (WCSS).

        **Por qué el codo:** al aumentar K el error siempre baja, incluso con agrupaciones sin
        sentido. Se busca el punto donde la mejora deja de ser notable. Aquí se mide la distancia
        de cada punto de la curva al segmento que une el primero y el último; el más alejado es
        K={suggested_k}. Es una heurística visual, no una verdad absoluta: por eso puedes cambiar K
        en la barra lateral.
        """
    )

    st.divider()
    st.subheader("Segundo método: silhouette por K")
    sil_chart_column, sil_detail_column = st.columns([2, 1])
    with sil_chart_column:
        figure = plot_silhouette(silhouettes, silhouette_k, k)
        st.pyplot(figure)
        plt.close(figure)
    with sil_detail_column:
        st.metric("K con mejor silhouette", silhouette_k)
        st.metric(f"Silhouette con K={silhouette_k}", f"{silhouettes[silhouette_k]:.3f}")
        if k in silhouettes:
            st.metric(f"Silhouette con K={k} (usado)", f"{silhouettes[k]:.3f}")
    agreement = (
        f"**Ambos métodos coinciden en K={k}**, lo que refuerza la elección."
        if suggested_k == silhouette_k
        else (
            f"**Los métodos no coinciden:** el codo sugiere K={suggested_k} y la silhouette "
            f"K={silhouette_k}. La diferencia de silhouette es de "
            f"{silhouettes[silhouette_k] - silhouettes[suggested_k]:.3f}; si es pequeña, ambos K "
            "son razonables y se prefiere el más simple de interpretar."
        )
    )
    st.markdown(
        f"""
        **Qué mide:** para cada cliente compara *a* = distancia media a los de su propio grupo y
        *b* = distancia media al grupo vecino más cercano: $s = \\frac{{b - a}}{{\\max(a, b)}}$.
        Va de -1 a 1; cerca de 1 significa grupos compactos y bien separados, cerca de 0 grupos
        que se solapan. Cada punto de la gráfica es el promedio de *s* de los clientes con ese K
        (`silhouette_score` de scikit-learn, mismos `KMeans` con `n_init=10`).

        **Diferencia con el codo:** el WCSS siempre baja al subir K, por eso hay que buscar un
        quiebre; la silhouette no, así que se elige directamente el K con el valor **más alto**.
        Empieza en K=2 porque con un solo grupo no existe "grupo vecino".

        {agreement}
        """
    )

with tab_simulacion:
    st.header("¿Cómo se agrupan los clientes y cómo se mueven los centroides?")
    st.slider("Iteración", min_value=0, max_value=len(snapshots) - 1, key="step_index")
    snapshot = snapshots[st.session_state.step_index]
    metric_columns = st.columns(5)
    metric_columns[0].metric("K (grupos)", k)
    metric_columns[1].metric("Iteración", f"{snapshot.iteration}/{len(snapshots)}")
    metric_columns[2].metric("Puntos que cambiaron", snapshot.changed_points)
    metric_columns[3].metric("Movimiento máximo", f"{snapshot.max_center_shift:.4f}")
    metric_columns[4].metric("WCSS", f"{snapshot.inertia:.2f}")
    st.caption(f"K={k} significa {k} centroides: las {k} marcas X negras del gráfico.")

    if snapshot.iteration > 1:
        previous_inertia = snapshots[snapshot.iteration - 2].inertia
        st.caption(f"El WCSS bajó {previous_inertia - snapshot.inertia:.2f} respecto a la iteración anterior.")

    if view_mode == "3D":
        figure = plot_snapshot_3d(data, snapshot, scaler, initial_centers, feature_columns)
        st.plotly_chart(figure, width="stretch")
    else:
        figure = plot_snapshot(data, matrix, snapshot, scaler, initial_centers, feature_columns)
        st.pyplot(figure)
        plt.close(figure)

    dimensions = "tres dimensiones" if view_mode == "3D" else "dos dimensiones"
    if snapshot.iteration == 1:
        step_explanation = (
            f"Con la semilla {seed}, `kmeans_plusplus` colocó {k} centroides iniciales lo más "
            f"separados posible en {dimensions}. En esta primera iteración cada cliente se asignó "
            "al centroide más cercano usando distancia euclidiana sobre los datos estandarizados; "
            "después, cada centroide se movió a la media de los clientes que le tocaron."
        )
    else:
        region_text = (
            "Las zonas de color muestran qué centroide ganaría en esa región; "
            if view_mode == "2D"
            else "Puedes girar la escena 3D para ver la separación en Edad, Ingreso y Gasto; "
        )
        step_explanation = (
            "Se repite el mismo par de operaciones: asignar cada punto al centroide más cercano "
            "y recolocar cada centroide en la media de su grupo. "
            + region_text
            + "las líneas grises unen clientes de muestra con su centroide asignado."
        )
    if snapshot.is_converged:
        step_explanation += (
            " **Convergencia:** los centroides dejaron de moverse y las etiquetas ya no cambian, "
            "así que el algoritmo se detuvo."
        )
    st.markdown(step_explanation)
    st.info(
        f"El modelo terminó en {final_model.n_iter_} iteraciones de scikit-learn; la simulación "
        f"reproduce esas mismas {len(snapshots)} iteraciones una por una."
    )

with tab_resultado:
    st.header("Resultado final")
    result_metrics = st.columns(4)
    result_metrics[0].metric("Grupos (K)", k)
    result_metrics[1].metric("Iteraciones", final_model.n_iter_)
    result_metrics[2].metric("WCSS final", f"{final_model.inertia_:.2f}")
    result_metrics[3].metric("Silhouette", f"{run['silhouette']:.3f}")
    st.dataframe(cluster_table, width="stretch", hide_index=True)
    centroid_columns = ", ".join(
        f"`Centroide {FEATURE_LABELS[column]}`" for column in feature_columns
    )
    st.markdown(
        f"""
        Cada **centroide** es el promedio de las coordenadas estandarizadas de su grupo; al
        convertirlo a la escala original obtenemos {centroid_columns}, que coinciden con el
        promedio real de ese clúster. La inercia del grupo suma las distancias al cuadrado de sus
        clientes respecto a ese centroide; la suma de todas es el WCSS final.

        En vista 3D la distancia usa **Edad, Ingreso y Gasto** a la vez; en 2D solo Ingreso y
        Gasto. **Silhouette** mide qué tan separados quedaron los grupos (cerca de 1 es mejor).
        """
    )

with tab_mapa:
    st.header("De dónde sale cada cosa y qué archivo la produce")
    st.code(
        """Mall_Customers.csv
  └─> kmeans_engine.load_customers()      lee Edad, Ingreso y Gasto (o solo Ingreso y Gasto en 2D)
        └─> scale_features()              StandardScaler
              ├─> elbow_curve()           WCSS para K=1..10 con sklearn KMeans
              │     └─> choose_k_by_elbow()  → K sugerido
              └─> run_iterations()        kmeans_plusplus + KMeans(max_iter=1) por paso
                    ├─> IterationSnapshot  → pestaña 2 (gráfica 3D/2D y texto por iteración)
                    └─> final_model        → pestaña 3 (K, iteraciones, inercia, centroides)

app.py construye las pestañas, los controles, Plotly 3D y las gráficas a partir de esos objetos.
flowers/ queda intacto: no se usa en este simulador.
""",
        language="text",
    )
    st.markdown(
        """
        **Distancia de asignación:** $d(x, \\mu_c) = \\|x - \\mu_c\\|^2$. Cada cliente entra al
        grupo cuyo centroide esté más cerca.

        **Centroide:** $\\mu_c = \\frac{1}{|C_c|}\\sum_{x \\in C_c} x$. Es la media de los puntos
        asignados; por eso la gráfica lo dibuja como una `X`.

        **WCSS / inercia:** $\\sum_c \\sum_{x \\in C_c} \\|x - \\mu_c\\|^2$. Es el número que baja
        en cada iteración y el que se grafica para elegir K.

        **Iteración:** una asignación completa seguida de una actualización completa de todos los
        centroides. `n_iter_` de scikit-learn cuenta cuántas veces se repitió ese ciclo; la
        simulación guarda una instantánea por cada una.

        **Archivos relacionados:** `Mall_Customers.csv` aporta los datos; `kmeans_engine.py` hace
        todo el cálculo con scikit-learn; `app.py` solo muestra y explica; `tests/` comprueba que
        la simulación coincide con el modelo final; `requirements.txt` fija las librerías.
        """
    )
