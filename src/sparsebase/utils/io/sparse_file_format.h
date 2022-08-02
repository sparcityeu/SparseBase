/*******************************************************
 * Copyright (c) 2022 SparCity, Amro Alabsi Aljundi, Taha Atahan Akyildiz, Arda Sener
 * All rights reserved.
 *
 * This file is distributed under MIT license.
 * The complete license agreement can be obtained at:
 * https://sparcityeu.github.io/sparsebase/pages/license.html
 ********************************************************/
#ifndef SPARSEBASE_SPARSEBASE_UTILS_IO_SPARSE_FILE_FORMAT_H_
#define SPARSEBASE_SPARSEBASE_UTILS_IO_SPARSE_FILE_FORMAT_H_

#ifdef USE_PIGO
#include "sparsebase/external/pigo/pigo.hpp"
#endif

#include "sparsebase/external/json/json.hpp"
#include <climits>
#include <iostream>
#include <string>
#include <type_traits>

namespace sparsebase {

namespace utils {

namespace io {

#ifdef USE_PIGO

class SbffWriteFile {
private:
  pigo::WFile file;

public:
  SbffWriteFile(std::string filename, size_t size) : file(filename, size) {}

  void Write(char *data, size_t size) { file.parallel_write(data, size); }
};

class SbffReadOnlyFile {
private:
  pigo::ROFile file;

public:
  explicit SbffReadOnlyFile(std::string filename) : file(filename) {}

  void Read(char *buffer, size_t size) { file.parallel_read(buffer, size); }
};

#else

class SbffWriteFile {
private:
  std::ofstream ofs;

public:
  SbffWriteFile(std::string filename, size_t size) {
    ofs.open(filename, std::ios::out | std::ios::binary);
  }
  ~SbffWriteFile() { ofs.close(); }
  void Write(char *data, size_t size) { ofs.write(data, size); }
};

class SbffReadOnlyFile {
private:
  std::ifstream ifs;

public:
  explicit SbffReadOnlyFile(std::string filename) {
    ifs.open(filename, std::ios::in | std::ios::binary);
  }
  ~SbffReadOnlyFile() { ifs.close(); }
  void Read(char *buffer, size_t size) { ifs.read(buffer, size); }
};

#endif

class SbffArray {
private:
  std::string name;
  size_t array_size;
  size_t type_size;
  std::string type;
  char *data;
  std::string endian;

  SbffArray() = default;

  friend class SbffObject;

public:
  template <typename T>
  static SbffArray Create(std::string name, T *arr, size_t size) {
    SbffArray sbas_arr;
    sbas_arr.name = name;
    sbas_arr.data = (char *)arr;
    sbas_arr.array_size = size;
    sbas_arr.type_size = sizeof(T);
    sbas_arr.endian = GetEndian();

    if constexpr (std::is_floating_point_v<T>) {
      sbas_arr.type = "float";
    } else if constexpr (std::is_signed_v<T>) {
      sbas_arr.type = "signed";
    } else if constexpr (std::is_unsigned_v<T>) {
      sbas_arr.type = "unsigned";
    } else {
      throw sparsebase::utils::WriterException(std::string("Type ") +
                                               typeid(T).name() +
                                               " is not supported by SBFF");
    }

    return sbas_arr;
  }

  static nlohmann::json ReadHeader(SbffReadOnlyFile &file) {
    char header_bytes[1024];
    file.Read((char *)&header_bytes, 1024);
    nlohmann::json header = nlohmann::json::parse(header_bytes);
    return header;
  }

  static SbffArray ReadArray(SbffReadOnlyFile &file, std::string endian) {
    try {
      auto header = ReadHeader(file);

      SbffArray sbas_arr;
      sbas_arr.array_size = header.at("array_size");
      sbas_arr.type_size = header.at("type_size");
      sbas_arr.type = header.at("type");
      sbas_arr.name = header.at("name");
      sbas_arr.endian = endian;

      sbas_arr.data = new char[sbas_arr.array_size * sbas_arr.type_size];
      file.Read((char *)sbas_arr.data,
                sbas_arr.array_size * sbas_arr.type_size);

      return sbas_arr;

    } catch (sparsebase::utils::ReaderException &e) {
      throw sparsebase::utils::ReaderException(e.what());
    } catch (...) {
      throw sparsebase::utils::ReaderException("Unknown SBFF ReadArray Error");
    }
  }

  static std::vector<char> HeaderToBytes(const nlohmann::json &header) {
    std::string header_str = header.dump();
    std::vector<char> header_bytes(header_str.begin(), header_str.end());

    // Headers should have a maximum size of 1024 bytes
    if (header_bytes.size() > 1024) {
      throw sparsebase::utils::WriterException("Header size exceeds 1 KB");
    }

    // Pad the header to exactly 1024 bytes
    while (header_bytes.size() < 1024) {
      header_bytes.push_back(' ');
    }

    return header_bytes;
  }

  void WriteArray(SbffWriteFile &file) {

    nlohmann::json header;
    header["name"] = name;
    header["type"] = type;
    header["type_size"] = type_size;
    header["array_size"] = array_size;

    file.Write((char *)HeaderToBytes(header).data(), 1024);
    file.Write((char *)data, array_size * type_size);
  }

  // This will fail if sizeof(int) == 1
  // which might be the case on some embedded systems
  static std::string GetEndian() {
    const int value{0x01};
    const void *address = static_cast<const void *>(&value);
    const auto *least_significant_address =
        static_cast<const unsigned char *>(address);
    return (*least_significant_address == 0x01) ? "little" : "big";
  }

  template <typename T> static T SwapEndian(T u) {
    static_assert(CHAR_BIT == 8, "CHAR_BIT != 8");

    union {
      T u;
      unsigned char u8[sizeof(T)];
    } source, dest;

    source.u = u;

    for (size_t k = 0; k < sizeof(T); k++)
      dest.u8[k] = source.u8[sizeof(T) - k - 1];

    return dest.u;
  }
};

struct SbffObject {
private:
  std::string name;
  std::unordered_map<std::string, SbffArray> arrays;
  std::vector<int> dimensions;
  size_t total_size = 1024;

public:
  explicit SbffObject(std::string name) : name(name) {}

  void AddDimensions(const std::vector<format::DimensionType> &dims) {
    dimensions.insert(dimensions.end(), dims.begin(), dims.end());
  }

  template <typename T> void AddArray(std::string name, T *arr, size_t size) {
    auto sbas_arr = SbffArray::Create(name, arr, size);
    arrays.emplace(name, sbas_arr);
    total_size += 1024 + sizeof(T) * size;
  }

  void AddArray(SbffArray sbas_arr) {
    arrays.emplace(sbas_arr.name, sbas_arr);
    total_size += 1024 + sbas_arr.array_size * sbas_arr.type_size;
  }

  template <typename T> size_t GetArray(std::string name, T *&ptr) {

    try {

      SbffArray &arr = arrays.at(name);

      if (arr.type == "float" && !std::is_floating_point_v<T>) {
        throw sparsebase::utils::ReaderException(
            "Type mismatch, array type is float");
      }

      if (arr.type == "signed" && !std::is_signed_v<T>) {
        throw sparsebase::utils::ReaderException(
            "Type mismatch, array type is signed");
      }

      if (arr.type == "unsigned" && !std::is_unsigned_v<T>) {
        throw sparsebase::utils::ReaderException(
            "Type mismatch, array type is unsigned");
      }

      if (arr.type_size != sizeof(T)) {
        throw sparsebase::utils::ReaderException(
            std::string("Type mismatch, array type has size ") +
            std::to_string(arr.type_size));
      }

      ptr = (T *)arr.data;

      if (arr.endian != SbffArray::GetEndian()) {
#pragma omp parallel for shared(ptr, arr) default(none)
        for (size_t i = 0; i < arr.array_size; i++) {
          ptr[i] = SbffArray::SwapEndian(ptr[i]);
        }
      }

      return arr.array_size;
    } catch (sparsebase::utils::ReaderException &e) {
      throw sparsebase::utils::ReaderException(e.what());
    } catch (...) {
      throw sparsebase::utils::ReaderException("Unknown SBFF ReadArray Error");
    }
  }

  void WriteObject(std::string filename) {
    SbffWriteFile file(filename, total_size);
    WriteObject(file);
  }

  void WriteObject(SbffWriteFile &file) {
    nlohmann::json header;
    header["name"] = name;
    header["array_count"] = arrays.size();
    header["dimensions"] = dimensions;
    header["endian"] = SbffArray::GetEndian();

    file.Write((char *)SbffArray::HeaderToBytes(header).data(), 1024);

    for (auto arr : arrays) {
      arr.second.WriteArray(file);
    }
  }

  static SbffObject ReadObject(SbffReadOnlyFile &file) {
    try {
      auto header = SbffArray::ReadHeader(file);

      SbffObject obj("temp");
      obj.name = header.at("name");
      std::string endian = header.at("endian");
      size_t array_count = header.at("array_count");
      auto dims = header.at("dimensions");
      obj.dimensions.insert(obj.dimensions.end(), dims.begin(), dims.end());

      for (size_t i = 0; i < array_count; i++)
        obj.AddArray(SbffArray::ReadArray(file, endian));

      return obj;

    } catch (sparsebase::utils::ReaderException &e) {
      throw sparsebase::utils::ReaderException(e.what());
    } catch (...) {
      throw sparsebase::utils::ReaderException("Unknown SBFF ReadArray Error");
    }
  }

  static SbffObject ReadObject(std::string filename) {
    SbffReadOnlyFile file(filename);
    return ReadObject(file);
  }

  std::string get_name() { return name; }

  size_t get_array_count() { return arrays.size(); }

  std::vector<int> get_dimensions() { return dimensions; }
};

} // namespace io

} // namespace utils

} // namespace sparsebase

#endif // SPARSEBASE_SPARSEBASE_UTILS_IO_SPARSE_FILE_FORMAT_H_
