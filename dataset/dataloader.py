from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import scipy.io
import scipy.sparse
import torch
from sklearn.preprocessing import StandardScaler, normalize
from torch.utils.data import Dataset


DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
SKIPPED_FILES = {
    "BBC4view.mat": "damaged MAT payload",
    "scene15.mat": "variable-size raw image cells, not aligned feature views",
}


def discover_dataset_files():
    files = {
        path.stem: path.name
        for path in sorted(DATA_ROOT.glob("*.mat"), key=lambda item: item.name.lower())
        if path.name not in SKIPPED_FILES
    }
    ccv_parts = ("STIP.npy", "SIFT.npy", "MFCC.npy", "label.npy")
    if all((DATA_ROOT / name).is_file() for name in ccv_parts):
        files["CCV"] = "<STIP.npy,SIFT.npy,MFCC.npy,label.npy>"
    return files


DATASET_FILES = discover_dataset_files()


def _dense(matrix):
    return matrix.toarray() if scipy.sparse.issparse(matrix) else np.asarray(matrix)


def _remap_labels(values):
    values = np.asarray(values).reshape(-1)
    classes = np.unique(values)
    mapping = {value: index for index, value in enumerate(classes.tolist())}
    return np.asarray([mapping[value] for value in values], dtype=np.int64)


def _normalize_view(values):
    values = _dense(values).reshape((values.shape[0], -1)).astype(np.float32)
    sparse_nonnegative = (
        values.shape[1] >= 1000
        and np.nanmin(values) >= 0
        and float(np.mean(values == 0)) >= 0.75
    )
    if sparse_nonnegative:
        return normalize(values, norm="l2", axis=1).astype(np.float32), None
    scaler = StandardScaler()
    return scaler.fit_transform(values).astype(np.float32), scaler


def _label_key(mapping):
    aliases = {str(key).lower(): key for key in mapping}
    for candidate in ("y", "gt", "labels", "label", "truth"):
        if candidate in aliases:
            return aliases[candidate]
    raise KeyError("no label variable found (expected y/Y, gt, labels, or label)")


def _materialize_view(value, sample_count):
    value = _dense(value)
    if value.dtype == object:
        cells = value.reshape(-1)
        if cells.size == sample_count:
            shapes = {np.asarray(cell).shape for cell in cells}
            if len(shapes) != 1:
                raise ValueError("variable-size sample cells are not aligned feature views")
            value = np.stack([np.asarray(cell).reshape(-1) for cell in cells])
        elif cells.size == 1:
            value = _dense(cells[0])
        else:
            raise ValueError("nested object cells cannot be interpreted as one view")
    value = np.asarray(value)
    if value.shape[0] != sample_count and value.ndim == 2 and value.shape[1] == sample_count:
        value = value.T
    if value.shape[0] != sample_count:
        raise ValueError(
            f"view has {value.shape[0]} samples but labels contain {sample_count}"
        )
    return value.reshape((sample_count, -1)).astype(np.float32)


def _load_v5_mat(path):
    raw = scipy.io.loadmat(path)
    labels = np.asarray(raw[_label_key(raw)]).reshape(-1)
    sample_count = labels.size
    preserve_raw = False
    if "X" in raw and np.asarray(raw["X"]).dtype == object:
        values = np.asarray(raw["X"], dtype=object).reshape(-1)
    elif "fea" in raw and np.asarray(raw["fea"]).dtype == object:
        values = np.asarray(raw["fea"], dtype=object).reshape(-1)
        preserve_raw = True
    elif "data" in raw and np.asarray(raw["data"]).dtype == object:
        values = np.asarray(raw["data"], dtype=object).reshape(-1)
    else:
        keys = sorted(
            (key for key in raw if key.startswith("X") and key[1:].isdigit()),
            key=lambda key: int(key[1:]),
        )
        if not keys:
            raise KeyError("no multiview variables found")
        values = [raw[key] for key in keys]
    views = [_materialize_view(value, sample_count) for value in values]
    return views, labels, preserve_raw


def _load_hdf_mat(path):
    with h5py.File(path, "r") as raw:
        label_name = _label_key(raw)
        labels = np.asarray(raw[label_name]).reshape(-1)
        sample_count = labels.size
        container_name = "X" if "X" in raw else "data" if "data" in raw else None
        if container_name is None:
            raise KeyError("no X or data view-reference container found")
        references = np.asarray(raw[container_name]).reshape(-1)
        values = [np.asarray(raw[reference]) for reference in references]
    views = [_materialize_view(value, sample_count) for value in values]
    return views, labels, False


def load_mat_views(path):
    try:
        return _load_v5_mat(path)
    except NotImplementedError:
        return _load_hdf_mat(path)


class MultiViewDataset(Dataset):
    def __init__(self, name, views, labels, preserve_raw=False):
        sizes = {view.shape[0] for view in views}
        if len(sizes) != 1:
            raise ValueError("all views must contain the same samples")
        self.raw_multi_view = [np.asarray(view, dtype=np.float32) for view in views]
        if preserve_raw:
            self.multi_view = self.raw_multi_view
            self.scalers = [None] * len(views)
        else:
            normalized = [_normalize_view(view) for view in self.raw_multi_view]
            self.multi_view = [item[0] for item in normalized]
            self.scalers = [item[1] for item in normalized]
        self.labels = _remap_labels(labels)
        self.view = len(self.multi_view)
        self.dims = [int(view.shape[1]) for view in self.multi_view]
        self.data_size = int(self.multi_view[0].shape[0])
        self.class_num = int(np.unique(self.labels).size)
        self.use_raw_reconstruction_eval = name in {"ORL", "Prokaryotic"}
        self.raw_reconstruction_eval_mode = (
            "raw_forward" if name == "Prokaryotic" else "inverse"
        )

    def __len__(self):
        return self.data_size

    def __getitem__(self, index):
        features = [torch.from_numpy(view[index]) for view in self.multi_view]
        return features, torch.as_tensor(self.labels[index]), torch.as_tensor(index)


def _load_ccv():
    views = [
        np.load(DATA_ROOT / "STIP.npy", mmap_mode="r"),
        np.load(DATA_ROOT / "SIFT.npy", mmap_mode="r"),
        np.load(DATA_ROOT / "MFCC.npy", mmap_mode="r"),
    ]
    labels = np.load(DATA_ROOT / "label.npy")
    return MultiViewDataset("CCV", views, labels, preserve_raw=True)


def load_data(name):
    current_files = discover_dataset_files()
    if name not in current_files:
        supported = ", ".join(current_files)
        raise ValueError(f"unsupported dataset {name!r}; choose one of: {supported}")
    if name == "CCV":
        dataset = _load_ccv()
    else:
        path = DATA_ROOT / current_files[name]
        views, labels, preserve_raw = load_mat_views(path)
        dataset = MultiViewDataset(name, views, labels, preserve_raw)
    print(
        f"dims: {dataset.dims}, view: {dataset.view}, "
        f"data_size: {dataset.data_size}, classnum: {dataset.class_num}"
    )
    return (
        dataset,
        dataset.dims,
        dataset.view,
        dataset.data_size,
        dataset.class_num,
    )

