import math

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader


REPRESENTATION_ORDER = (
    "H",
    "Z",
    "LHZ",
    "HHZ",
)


def cluster_acc(
    y_true,
    y_pred,
):
    y_true = np.asarray(
        y_true,
        dtype=np.int64,
    ).reshape(-1)

    y_pred = np.asarray(
        y_pred,
        dtype=np.int64,
    ).reshape(-1)

    if y_true.size != y_pred.size:
        raise ValueError(
            "y_true and y_pred must have the same size"
        )

    D = int(
        max(
            y_true.max(),
            y_pred.max(),
        )
        + 1
    )

    w = np.zeros(
        (D, D),
        dtype=np.int64,
    )

    for pred, true in zip(
        y_pred,
        y_true,
    ):
        w[pred, true] += 1

    row_ind, col_ind = (
        linear_sum_assignment(
            w.max() - w
        )
    )

    return float(
        w[row_ind, col_ind].sum()
    ) / float(y_true.size)


def purity(
    y_true,
    y_pred,
):
    y_true = np.asarray(
        y_true,
    ).reshape(-1)

    y_pred = np.asarray(
        y_pred,
    ).reshape(-1)

    classes = np.unique(
        y_true
    )

    mapping = {
        value: index
        for index, value
        in enumerate(
            classes.tolist()
        )
    }

    remapped = np.asarray(
        [
            mapping[value]
            for value in y_true
        ],
        dtype=np.int64,
    )

    voted = np.zeros_like(
        remapped
    )

    for cluster in np.unique(
        y_pred
    ):
        mask = y_pred == cluster

        counts = np.bincount(
            remapped[mask],
            minlength=len(classes),
        )

        voted[mask] = int(
            np.argmax(counts)
        )

    return accuracy_score(
        remapped,
        voted,
    )


def evaluate(
    label,
    pred,
):
    label = np.asarray(
        label
    ).reshape(-1)

    pred = np.asarray(
        pred
    ).reshape(-1)

    nmi = normalized_mutual_info_score(
        label,
        pred,
    )

    ari = adjusted_rand_score(
        label,
        pred,
    )

    acc = cluster_acc(
        label,
        pred,
    )

    pur = purity(
        label,
        pred,
    )

    return (
        nmi,
        ari,
        acc,
        pur,
    )


def _safe_numpy(
    values,
):
    values = np.asarray(
        values,
        dtype=np.float32,
    )

    return np.nan_to_num(
        values,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def _prepare_view(
    values,
    latent_scaling="none",
):
    values = _safe_numpy(
        values
    )

    if latent_scaling == "none":
        # Conservative V4 default: restore the earlier raw-latent KMeans protocol.
        return values

    if latent_scaling == "standard":
        if values.shape[0] <= 1:
            return values

        return (
            StandardScaler()
            .fit_transform(values)
            .astype(np.float32)
        )

    if latent_scaling == "standard_balanced":
        if values.shape[0] <= 1:
            return values

        values = (
            StandardScaler()
            .fit_transform(values)
            .astype(np.float32)
        )

        return values / math.sqrt(
            max(
                1,
                values.shape[1],
            )
        )

    raise ValueError(
        "latent_scaling must be one of: "
        "none, standard, standard_balanced"
    )


def _cluster(
    features,
    class_num,
    random_state=42,
    n_init=100,
):
    features = _safe_numpy(
        features
    )

    n_samples = features.shape[0]

    if n_samples > 10000:
        estimator = MiniBatchKMeans(
            n_clusters=int(
                class_num
            ),
            batch_size=min(
                4096,
                n_samples,
            ),
            n_init=max(
                20,
                int(n_init // 2),
            ),
            random_state=int(
                random_state
            ),
        )

    else:
        estimator = KMeans(
            n_clusters=int(
                class_num
            ),
            n_init=int(
                n_init
            ),
            random_state=int(
                random_state
            ),
        )

    return estimator.fit_predict(
        features
    )


def _metric_dict(
    labels,
    pred,
):
    nmi, ari, acc, pur = evaluate(
        labels,
        pred,
    )

    return {
        "ACC": float(acc),
        "NMI": float(nmi),
        "ARI": float(ari),
        "PUR": float(pur),
    }


def _record(
    metrics,
):
    return [
        float(metrics["ACC"]),
        float(metrics["NMI"]),
        float(metrics["ARI"]),
        float(metrics["PUR"]),
    ]


def inference(
    loader,
    model,
    device,
    view,
    data_size,
):
    model.eval()

    Zs = [
        []
        for _ in range(view)
    ]
    Hs = [
        []
        for _ in range(view)
    ]
    LHZs = [
        []
        for _ in range(view)
    ]
    HHZs = [
        []
        for _ in range(view)
    ]

    labels_vector = []
    quality_batches = []

    for xs, y, _ in loader:
        xs = [
            x.to(
                device,
                non_blocking=True,
            )
            for x in xs
        ]

        with torch.no_grad():
            (
                zs,
                lHZs,
                hHZs,
                _,
                _,
                _,
                hs,
                _,
                _,
            ) = model(xs)

        for v in range(view):
            Zs[v].append(
                zs[v]
                .detach()
                .cpu()
                .numpy()
            )

            Hs[v].append(
                hs[v]
                .detach()
                .cpu()
                .numpy()
            )

            LHZs[v].append(
                lHZs[v]
                .detach()
                .cpu()
                .numpy()
            )

            HHZs[v].append(
                hHZs[v]
                .detach()
                .cpu()
                .numpy()
            )

        labels_vector.append(
            np.asarray(
                y
            ).reshape(-1)
        )

        quality = getattr(
            model,
            "last_view_quality",
            None,
        )

        if quality is not None:
            quality_batches.append(
                quality
                .detach()
                .cpu()
                .numpy()
            )

    labels_vector = np.concatenate(
        labels_vector,
        axis=0,
    )

    if labels_vector.size != data_size:
        raise ValueError(
            f"Expected {data_size} labels, "
            f"got {labels_vector.size}"
        )

    def finish(
        nested,
    ):
        return [
            np.concatenate(
                values,
                axis=0,
            )
            for values in nested
        ]

    avg_view_quality = None

    if quality_batches:
        all_quality = np.concatenate(
            quality_batches,
            axis=0,
        )

        avg_view_quality = (
            all_quality
            .mean(axis=0)
            .tolist()
        )

    return {
        "Z": finish(Zs),
        "H": finish(Hs),
        "LHZ": finish(LHZs),
        "HHZ": finish(HHZs),
        "labels": labels_vector,
        "view_quality": avg_view_quality,
    }


def _evaluate_representation(
    representation_name,
    view_features,
    labels,
    class_num,
    seed,
    n_init,
    latent_scaling,
):
    prepared_views = [
        _prepare_view(
            values,
            latent_scaling=latent_scaling,
        )
        for values in view_features
    ]

    multi_features = np.concatenate(
        prepared_views,
        axis=1,
    )

    per_view = {}

    for v, features in enumerate(
        prepared_views
    ):
        pred = _cluster(
            features,
            class_num,
            random_state=(
                int(seed)
                + 1000
                + 31 * v
            ),
            n_init=n_init,
        )

        per_view[
            f"view_{v}"
        ] = _metric_dict(
            labels,
            pred,
        )

    multi_pred = _cluster(
        multi_features,
        class_num,
        random_state=int(seed),
        n_init=n_init,
    )

    multi_metrics = _metric_dict(
        labels,
        multi_pred,
    )

    best_view_name = max(
        per_view,
        key=lambda name: (
            per_view[name]["ACC"],
            per_view[name]["NMI"],
        ),
    )

    best_single = dict(
        per_view[
            best_view_name
        ]
    )

    mean_single = {
        metric_name: float(
            np.mean(
                [
                    metrics[
                        metric_name
                    ]
                    for metrics
                    in per_view.values()
                ]
            )
        )
        for metric_name in (
            "ACC",
            "NMI",
            "ARI",
            "PUR",
        )
    }

    return {
        "representation": (
            representation_name
        ),
        "multi": multi_metrics,
        "best_single_view": (
            best_view_name
        ),
        "best_single": best_single,
        "mean_single": mean_single,
        "per_view": per_view,
        "_multi_features": (
            multi_features
        ),
    }


def valid(
    model,
    device,
    dataset,
    view,
    data_size,
    class_num,
    eval_h=True,
    eval_z=True,
    test=True,
    eval_lhz=True,
    eval_hhz=True,
    max_records=None,
    stage="all",
    return_details=False,
    seed=42,
    n_init=100,
    latent_scaling="none",
):
    """Evaluate H, Z, LHZ and HHZ, then select the best representation by ACC.

    This restores the requested H/Z/LHZ/HHZ evaluation system while also
    writing every representation's metrics into the returned details object.
    """
    del test, stage

    if max_records is None:
        max_records = [
            [0.0, None, None, None],
            [0.0, None, None, None],
            [0.0, None, None, None],
        ]

    loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=False,
        num_workers=0,
    )

    inferred = inference(
        loader,
        model,
        device,
        view,
        data_size,
    )

    enabled = []

    if eval_h:
        enabled.append("H")
    if eval_z:
        enabled.append("Z")
    if eval_lhz:
        enabled.append("LHZ")
    if eval_hhz:
        enabled.append("HHZ")

    if not enabled:
        raise ValueError(
            "At least one representation must be enabled"
        )

    representations = {}

    for offset, name in enumerate(
        enabled
    ):
        representations[
            name
        ] = _evaluate_representation(
            representation_name=name,
            view_features=inferred[name],
            labels=inferred["labels"],
            class_num=class_num,
            seed=(
                int(seed)
                + 10000 * offset
            ),
            n_init=n_init,
            latent_scaling=latent_scaling,
        )

    # Main multi-view result: best H/Z/LHZ/HHZ by ACC.
    selected_name = max(
        representations,
        key=lambda name: (
            representations[name]["multi"]["ACC"],
            representations[name]["multi"]["NMI"],
            representations[name]["multi"]["ARI"],
        ),
    )

    selected = representations[
        selected_name
    ]

    # Single-view result: best single view across all representations.
    single_rep_name = max(
        representations,
        key=lambda name: (
            representations[name]["best_single"]["ACC"],
            representations[name]["best_single"]["NMI"],
        ),
    )

    selected_single = representations[
        single_rep_name
    ]["best_single"]

    # Mean-single result: best mean-single representation.
    mean_rep_name = max(
        representations,
        key=lambda name: (
            representations[name]["mean_single"]["ACC"],
            representations[name]["mean_single"]["NMI"],
        ),
    )

    selected_mean = representations[
        mean_rep_name
    ]["mean_single"]

    save_flag = False
    res_fea = None

    if (
        selected["multi"]["ACC"]
        > max_records[0][0]
    ):
        max_records[0] = _record(
            selected["multi"]
        )

        save_flag = True

        res_fea = selected[
            "_multi_features"
        ]

    if (
        selected_single["ACC"]
        > max_records[1][0]
    ):
        max_records[1] = _record(
            selected_single
        )

    if (
        selected_mean["ACC"]
        > max_records[2][0]
    ):
        max_records[2] = _record(
            selected_mean
        )

    # Remove raw feature arrays before JSON serialization.
    json_representations = {}

    for name, values in representations.items():
        json_representations[
            name
        ] = {
            key: value
            for key, value
            in values.items()
            if key != "_multi_features"
        }

    details = {
        "selection_rule": (
            "Evaluate H, Z, LHZ and HHZ separately; "
            "select the multi-view representation with the highest ACC. "
            "NMI/ARI/PUR are from that same representation."
        ),
        "selected_representation": (
            selected_name
        ),
        "selected_single_representation": (
            single_rep_name
        ),
        "selected_mean_representation": (
            mean_rep_name
        ),
        "latent_scaling": (
            latent_scaling
        ),
        "kmeans_n_init": int(
            n_init
        ),
        "multi": dict(
            selected["multi"]
        ),
        "best_single": dict(
            selected_single
        ),
        "mean_single": dict(
            selected_mean
        ),
        "representations": (
            json_representations
        ),
        "view_quality": (
            inferred["view_quality"]
        ),
    }

    print(
        "[EVAL] selected={} | "
        "ACC={:.4f} NMI={:.4f} "
        "ARI={:.4f} PUR={:.4f}".format(
            selected_name,
            selected["multi"]["ACC"],
            selected["multi"]["NMI"],
            selected["multi"]["ARI"],
            selected["multi"]["PUR"],
        ),
        flush=True,
    )

    for name in REPRESENTATION_ORDER:
        if name not in representations:
            continue

        m = representations[
            name
        ]["multi"]

        print(
            "       {:>3s}: "
            "ACC={:.4f} NMI={:.4f} "
            "ARI={:.4f} PUR={:.4f}".format(
                name,
                m["ACC"],
                m["NMI"],
                m["ARI"],
                m["PUR"],
            ),
            flush=True,
        )

    result = (
        selected["multi"]["ACC"],
        selected["multi"]["PUR"],
        save_flag,
        max_records,
        res_fea,
        inferred["labels"],
    )

    if return_details:
        return result + (
            details,
        )

    return result
