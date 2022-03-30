#ifndef _SPARSEREADER_HPP
#define _SPARSEREADER_HPP

#include "config.h"
#include "sparse_format.h"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <vector>


namespace sparsebase {

namespace utils {

class Reader {
public:
  virtual ~Reader() = default;
};

template <typename IDType, typename NNZType, typename ValueType>
class ReadsSparseFormat {
public:
  virtual format::Format *ReadSparseFormat() const = 0;
};

template <typename IDType, typename NNZType, typename ValueType> class ReadsCSR {
public:
  virtual format::CSR<IDType, NNZType, ValueType> *ReadCSR() const = 0;
};

template <typename IDType, typename NNZType, typename ValueType> class ReadsCOO {
public:
  virtual format::COO<IDType, NNZType, ValueType> *ReadCOO() const = 0;
};

template <typename T> class ReadsArray {
public:
  virtual format::Array<T> *ReadCOO() const = 0;
};

template <typename IDType, typename NNZType, typename ValueType>
class UedgelistReader : public Reader,
                        public ReadsCSR<IDType, NNZType, ValueType>,
                        public ReadsSparseFormat<IDType, NNZType, ValueType> {
public:
  explicit UedgelistReader(std::string filename, bool weighted = false);
  format::CSR<IDType, NNZType, ValueType> *ReadCSR() const;
  format::Format *ReadSparseFormat() const;
  ~UedgelistReader() override;

private:
  static bool SortEdge(const std::pair<IDType, IDType> &a,
                       const std::pair<IDType, IDType> &b);
  std::string filename_;
  bool weighted_;
};

template <typename IDType, typename NNZType, typename ValueType>
class MTXReader : public Reader,
                  public ReadsCOO<IDType, NNZType, ValueType>,
                  public ReadsSparseFormat<IDType, NNZType, ValueType> {
public:
  explicit MTXReader(std::string filename, bool weighted = false);
  format::COO<IDType, NNZType, ValueType> *ReadCOO() const;
  format::Format *ReadSparseFormat() const;
  ~MTXReader() override;

private:
  std::string filename_;
  bool weighted_;
};

#ifdef USE_PIGO
template <typename IDType, typename NNZType, typename ValueType>
class PigoMTXReader : public Reader,
                  public ReadsCOO<IDType, NNZType, ValueType>,
                      public ReadsCSR<IDType, NNZType, ValueType>,
                  public ReadsSparseFormat<IDType, NNZType, ValueType> {
public:
  PigoMTXReader(std::string filename, bool weighted = false, bool convert_to_zero_index = true);
  format::COO<IDType, NNZType, ValueType> *ReadCOO() const;
  format::CSR<IDType, NNZType, ValueType> *ReadCSR() const;
  format::Format *ReadSparseFormat() const;
  virtual ~PigoMTXReader() = default;

private:
  std::string filename_;
  bool weighted_;
  bool convert_to_zero_index_;
};

// Add ValueTypeed option with contexpr
template <typename IDType, typename NNZType, typename ValueType>
class PigoEdgeListReader : public Reader,
                        public ReadsCSR<IDType, NNZType, ValueType>,
                           public ReadsCOO<IDType, NNZType, ValueType>,
                        public ReadsSparseFormat<IDType, NNZType, ValueType> {
public:
  PigoEdgeListReader(std::string filename, bool weighted = false);
  format::CSR<IDType, NNZType, ValueType> *ReadCSR() const;
  format::COO<IDType, NNZType, ValueType> *ReadCOO() const;
  format::Format *ReadSparseFormat() const;
  virtual ~PigoEdgeListReader() = default;

private:
  std::string filename_;
  bool weighted_;
};

#endif



template <typename IDType, typename NNZType, typename ValueType>
class BinaryReaderOrderTwo : public Reader,
  public ReadsCSR<IDType, NNZType, ValueType>,
  public ReadsCOO<IDType, NNZType, ValueType> {
public:
  explicit BinaryReaderOrderTwo(std::string filename);
  ~BinaryReaderOrderTwo() override = default;
  format::COO<IDType, NNZType, ValueType> *ReadCOO() const;
  format::CSR<IDType, NNZType, ValueType> *ReadCSR() const;

private:
  std::string filename_;
};

template <typename T>
class BinaryReaderOrderOne : public Reader,
                             public ReadsArray<T> {
public:
  explicit BinaryReaderOrderOne(std::string filename);
  ~BinaryReaderOrderOne() override = default;
  format::Array<T> *ReadArray() const;
  
private:
  std::string filename_;
};


} // namespace utils

} // namespace sparsebase

#ifdef _HEADER_ONLY
#include "sparse_reader.cc"
#endif
#endif
