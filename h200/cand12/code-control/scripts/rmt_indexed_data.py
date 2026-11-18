"""Map-style image dataset backed by large indexed byte shards."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import imagenet1k_runtime as runtime
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

if TYPE_CHECKING:
    import torch

# pyright: reportExplicitAny=false, reportPrivateUsage=false
# pyright: reportImplicitRelativeImport=false


class IndexedByteShardImageFolder(Dataset[tuple[Any, int]]):
    """Preserve ImageFolder sample order without per-image filesystem opens."""

    def __init__(self, root: Path, split: str, *, transform: Any = None) -> None:
        self.root = root.expanduser().resolve()
        self.split = split
        self.transform = transform
        index_root = self.root / split
        self.classes = json.loads((index_root / "classes.json").read_text())
        if not isinstance(self.classes, list) or not all(
            isinstance(name, str) for name in self.classes
        ):
            raise TypeError(f"invalid classes index: {index_root}")
        self.class_to_idx = {name: index for index, name in enumerate(self.classes)}
        self.shards = np.load(index_root / "shard.npy", mmap_mode="r", allow_pickle=False)
        self.offsets = np.load(index_root / "offset.npy", mmap_mode="r", allow_pickle=False)
        self.lengths = np.load(index_root / "length.npy", mmap_mode="r", allow_pickle=False)
        self.targets_array = np.load(index_root / "target.npy", mmap_mode="r", allow_pickle=False)
        array_lengths = {
            len(self.shards),
            len(self.offsets),
            len(self.lengths),
            len(self.targets_array),
        }
        if len(array_lengths) != 1:
            raise RuntimeError(f"inconsistent indexed shard arrays: {index_root}")
        self._handles: dict[int, Any] = {}

    def __len__(self) -> int:
        return len(self.targets_array)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        shard = int(self.shards[index])
        handle = self._handles.get(shard)
        if handle is None:
            handle = (self.root / f"{self.split}-{shard:04d}.bin").open("rb")
            self._handles[shard] = handle
        handle.seek(int(self.offsets[index]))
        length = int(self.lengths[index])
        payload = handle.read(length)
        if len(payload) != length:
            raise EOFError(f"short indexed-shard read: {self.split}/{index}")
        with Image.open(io.BytesIO(payload)) as encoded:
            image = encoded.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(self.targets_array[index])

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_handles"] = {}
        return state

    def __del__(self) -> None:
        for handle in getattr(self, "_handles", {}).values():
            handle.close()


def indexed_root() -> Path | None:
    """Select the optional read backend without affecting the default loader."""
    import os

    value = os.environ.get("RMT_INDEX_ROOT")
    return Path(value).expanduser().resolve() if value else None


def source_binding() -> dict[str, str]:
    """Bind code, array content, and existing manifest to checkpoint identity."""
    import hashlib
    import inspect

    root = indexed_root()
    if root is None:
        return {}
    files = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for split in ("train", "val")
        for filename in ("classes.json", "shard.npy", "offset.npy", "length.npy", "target.npy")
        for path in (root / split / filename,)
    }
    for filename in ("manifest.json", "dataset-manifest.json"):
        path = root / filename
        if path.is_file():
            files[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "indexed_adapter": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "indexed_reader": hashlib.sha256(
            inspect.getsource(IndexedByteShardImageFolder).encode()
        ).hexdigest(),
        "indexed_metadata": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
        "indexed_root": hashlib.sha256(str(root).encode()).hexdigest(),
    }


def build_loaders(task: runtime.BaselineTask, device: torch.device) -> runtime.LoaderBundle:
    """Reuse the exact runtime transforms, sampling and worker configuration.

    ImageFolder is replaced only during single-threaded loader construction.
    Workers receive the resulting top-level dataset class, never this factory or
    a persistent monkeypatch. No torchvision state survives this context.
    """
    root = indexed_root()
    if root is None:
        return runtime._build_loaders(task, device)
    from unittest.mock import patch

    import torchvision.datasets

    data_root = task.data_root.resolve()

    def image_folder(
        split_root: str | Path, *, transform: Any = None
    ) -> IndexedByteShardImageFolder:
        split_path = Path(split_root).resolve()
        if split_path.parent != data_root or split_path.name not in ("train", "val"):
            raise ValueError("indexed adapter received an unexpected ImageFolder root")
        return IndexedByteShardImageFolder(root, split_path.name, transform=transform)

    with patch.object(torchvision.datasets, "ImageFolder", image_folder):
        bundle = runtime._build_loaders(task, device)
    runtime._atomic_json(
        task.output_dir / "data-backend.json",
        {
            "backend": "indexed_byte_shards",
            "index_root": str(root),
            "logical_data_root": str(data_root),
            "source_binding": source_binding(),
            "note": "Only image bytes reader changes; transforms and loader recipe are unchanged.",
        },
    )
    return bundle
