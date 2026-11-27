# ruff: noqa: ANN001, ANN201, ANN202, ANN204, PTH108, PTH118, PTH123
"""ImageFolder ordering with immutable contiguous storage for forked workers."""

import mmap
import os
import tempfile
import weakref
from array import array

import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets.folder import IMG_EXTENSIONS, find_classes, has_file_allowed_extension


class CompactImageFolder(Dataset):
    def __init__(self, root, transform, loader, *, mapped=False):
        self.root = os.fspath(root)
        self.transform = transform
        self.loader = loader
        self.classes, self.class_to_idx = find_classes(self.root)
        path_bytes = bytearray()
        offsets = array("Q", [0])
        labels = array("H")
        for name, label in self.class_to_idx.items():
            class_root = os.path.join(self.root, name)
            start = len(labels)
            for directory, _, filenames in sorted(os.walk(class_root, followlinks=True)):
                for filename in sorted(filenames):
                    path = os.path.join(directory, filename)
                    if not has_file_allowed_extension(path, IMG_EXTENSIONS):
                        continue
                    relative = path[len(self.root.rstrip(os.sep)) + 1 :]
                    path_bytes.extend(os.fsencode(relative))
                    offsets.append(len(path_bytes))
                    labels.append(label)
            if len(labels) == start:
                raise FileNotFoundError(f"No valid images for class {name!r} in {self.root}")
        self._mapped_path = None
        if mapped:
            fd, path = tempfile.mkstemp(prefix="imagenet-manifest-", suffix=".bin")
            try:
                with os.fdopen(fd, "wb") as binary:
                    binary.write(path_bytes)
                    offsets.tofile(binary)
                    labels.tofile(binary)
            except BaseException:
                os.unlink(path)
                raise
            self._mapped_path = path
            self._path_end = len(path_bytes)
            self._offset_end = self._path_end + len(offsets) * offsets.itemsize
            self._label_count = len(labels)
            self._remove_map = weakref.finalize(self, os.unlink, path)
            self._map_arrays()
        else:
            self.path_bytes = bytes(path_bytes)
            self.offsets = np.frombuffer(offsets.tobytes(), dtype=np.uint64)
            self.labels = np.frombuffer(labels.tobytes(), dtype=np.uint16)

    def _map_arrays(self):
        with open(self._mapped_path, "rb") as binary:
            self._mapping = mmap.mmap(binary.fileno(), 0, access=mmap.ACCESS_READ)
        self.path_bytes = memoryview(self._mapping)[: self._path_end]
        self.offsets = np.frombuffer(
            self._mapping, dtype=np.uint64, count=self._label_count + 1, offset=self._path_end
        )
        self.labels = np.frombuffer(
            self._mapping, dtype=np.uint16, count=self._label_count, offset=self._offset_end
        )

    def __getstate__(self):
        state = vars(self).copy()
        if self._mapped_path is not None:
            for key in ("path_bytes", "offsets", "labels", "_mapping", "_remove_map"):
                state.pop(key)
        return state

    def __setstate__(self, state):
        vars(self).update(state)
        if self._mapped_path is not None:
            self._map_arrays()

    def __len__(self):
        return len(self.labels)

    def sample(self, index):
        start = self.offsets[index]
        end = self.offsets[index + 1]
        return os.path.join(self.root, os.fsdecode(bytes(self.path_bytes[start:end]))), int(
            self.labels[index]
        )

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        path, label = self.sample(index)
        image = self.loader(path)
        return self.transform(image) if self.transform is not None else image, label
