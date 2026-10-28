"""Read/write the .vol format used by the C++ code (see include/volume.hpp).

Arrays are returned with shape (nz, ny, nx) so that x is the fastest axis,
matching the C++ memory layout.
"""

import struct

import numpy as np

_MAGIC = {np.dtype(np.float32): b"VOLF", np.dtype(np.uint8): b"VOLB"}


def read_vol(path):
    with open(path, "rb") as f:
        magic = f.read(4)
        nx, ny, nz = struct.unpack("<3i", f.read(12))
        dtype = {b"VOLF": np.float32, b"VOLB": np.uint8}.get(magic)
        if dtype is None:
            raise ValueError(f"{path}: unknown magic {magic!r}")
        data = np.fromfile(f, dtype=dtype, count=nx * ny * nz)
    if data.size != nx * ny * nz:
        raise ValueError(f"{path}: truncated payload")
    return data.reshape(nz, ny, nx)


def write_vol(path, arr):
    arr = np.ascontiguousarray(arr)
    if arr.dtype not in _MAGIC:
        arr = arr.astype(np.float32)
    nz, ny, nx = arr.shape
    with open(path, "wb") as f:
        f.write(_MAGIC[arr.dtype])
        f.write(struct.pack("<3i", nx, ny, nz))
        arr.tofile(f)
