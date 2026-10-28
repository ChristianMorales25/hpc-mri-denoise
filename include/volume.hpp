#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// Dense 3D volume stored x-fastest: idx = (z * ny + y) * nx + x.
template <typename T>
struct Volume {
    int nx = 0, ny = 0, nz = 0;
    std::vector<T> data;

    Volume() = default;
    Volume(int nx_, int ny_, int nz_) : nx(nx_), ny(ny_), nz(nz_), data(size_t(nx_) * ny_ * nz_) {}

    size_t size() const { return data.size(); }
    size_t idx(int x, int y, int z) const { return (size_t(z) * ny + y) * nx + x; }
    T& at(int x, int y, int z) { return data[idx(x, y, z)]; }
    const T& at(int x, int y, int z) const { return data[idx(x, y, z)]; }

    // Replicate-border access (coordinates clamped into the volume).
    const T& clamped(int x, int y, int z) const {
        x = x < 0 ? 0 : (x >= nx ? nx - 1 : x);
        y = y < 0 ? 0 : (y >= ny ? ny - 1 : y);
        z = z < 0 ? 0 : (z >= nz ? nz - 1 : z);
        return at(x, y, z);
    }
};

using VolumeF = Volume<float>;
using VolumeU8 = Volume<uint8_t>;

// File format (little endian), shared with scripts/volio.py:
//   char[4] magic   "VOLF" (float32 data) or "VOLB" (uint8 data, e.g. labels)
//   int32   nx, ny, nz
//   payload nx*ny*nz elements, x-fastest
VolumeF read_volume_f32(const std::string& path);
void write_volume_f32(const std::string& path, const VolumeF& v);
VolumeU8 read_volume_u8(const std::string& path);
void write_volume_u8(const std::string& path, const VolumeU8& v);
