#include "volume.hpp"

#include <cstring>
#include <fstream>
#include <stdexcept>

namespace {

template <typename T>
Volume<T> read_impl(const std::string& path, const char* magic) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    char m[4];
    int32_t dims[3];
    f.read(m, 4);
    f.read(reinterpret_cast<char*>(dims), sizeof(dims));
    if (!f || std::memcmp(m, magic, 4) != 0)
        throw std::runtime_error(path + ": expected magic " + std::string(magic, 4));
    Volume<T> v(dims[0], dims[1], dims[2]);
    f.read(reinterpret_cast<char*>(v.data.data()), std::streamsize(v.size() * sizeof(T)));
    if (!f) throw std::runtime_error(path + ": truncated payload");
    return v;
}

template <typename T>
void write_impl(const std::string& path, const Volume<T>& v, const char* magic) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot write " + path);
    int32_t dims[3] = {v.nx, v.ny, v.nz};
    f.write(magic, 4);
    f.write(reinterpret_cast<const char*>(dims), sizeof(dims));
    f.write(reinterpret_cast<const char*>(v.data.data()), std::streamsize(v.size() * sizeof(T)));
}

}  // namespace

VolumeF read_volume_f32(const std::string& path) { return read_impl<float>(path, "VOLF"); }
void write_volume_f32(const std::string& path, const VolumeF& v) { write_impl(path, v, "VOLF"); }
VolumeU8 read_volume_u8(const std::string& path) { return read_impl<uint8_t>(path, "VOLB"); }
void write_volume_u8(const std::string& path, const VolumeU8& v) { write_impl(path, v, "VOLB"); }
