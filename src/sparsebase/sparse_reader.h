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

template <typename IDType, typename NNZType, typename ValueType>
class Reader {
public:
  virtual ~Reader();
};

template <class IDType, typename NNZType, typename ValueType>
class ReadsSparseFormat {
public:
  virtual format::Format *ReadSparseFormat() const = 0;
};

template <class IDType, typename NNZType, typename ValueType> class ReadsCSR {
public:
  virtual format::CSR<IDType, NNZType, ValueType> *ReadCSR() const = 0;
};

template <class IDType, typename NNZType, typename ValueType> class ReadsCOO {
public:
  virtual format::COO<IDType, NNZType, ValueType> *ReadCOO() const = 0;
};
// Add weighted option with contexpr
template <typename IDType, typename NNZType, typename ValueType>
class UedgelistReader : public Reader<IDType, NNZType, ValueType>,
                        public ReadsCSR<IDType, NNZType, ValueType>,
                        public ReadsSparseFormat<IDType, NNZType, ValueType> {
public:
  UedgelistReader(std::string filename, bool _weighted = false);
  format::CSR<IDType, NNZType, ValueType> *ReadCSR() const;
  format::Format *ReadSparseFormat() const;
  virtual ~UedgelistReader();

private:
  static bool SortEdge(const std::pair<IDType, IDType> &a,
                       const std::pair<IDType, IDType> &b);
  std::string filename_;
  bool weighted_;
};

template <typename IDType, typename NNZType, typename ValueType>
class MTXReader : public Reader<IDType, NNZType, ValueType>,
                  public ReadsCOO<IDType, NNZType, ValueType>,
                  public ReadsSparseFormat<IDType, NNZType, ValueType> {
public:
  MTXReader(std::string filename, bool _weighted = false);
  format::COO<IDType, NNZType, ValueType> *ReadCOO() const;
  format::Format *ReadSparseFormat() const;
  virtual ~MTXReader();

private:
  std::string filename_;
  bool weighted_;
};

template <typename IDType, typename NNZType, typename ValueType>
class PigoMTXReader : public Reader<IDType, NNZType, ValueType>,
                  public ReadsCOO<IDType, NNZType, ValueType>,
                      public ReadsCSR<IDType, NNZType, ValueType>,
                  public ReadsSparseFormat<IDType, NNZType, ValueType> {
public:
  PigoMTXReader(std::string filename, bool _weighted = false, bool _convert_to_zero_index = false);
  format::COO<IDType, NNZType, ValueType> *ReadCOO() const;
  format::CSR<IDType, NNZType, ValueType> *ReadCSR() const;
  format::Format *ReadSparseFormat() const;
  virtual ~PigoMTXReader() = default;

private:
  std::string filename_;
  bool weighted_;
  bool convert_to_zero_index_;
};

// Add weighted option with contexpr
template <typename IDType, typename NNZType, typename ValueType>
class PigoEdgeListReader : public Reader<IDType, NNZType, ValueType>,
                        public ReadsCSR<IDType, NNZType, ValueType>,
                           public ReadsCOO<IDType, NNZType, ValueType>,
                        public ReadsSparseFormat<IDType, NNZType, ValueType> {
public:
  PigoEdgeListReader(std::string filename, bool _weighted = false);
  format::CSR<IDType, NNZType, ValueType> *ReadCSR() const;
  format::COO<IDType, NNZType, ValueType> *ReadCOO() const;
  format::Format *ReadSparseFormat() const;
  virtual ~PigoEdgeListReader() = default;

private:
  std::string filename_;
  bool weighted_;
};

} // namespace utils

} // namespace sparsebase

#ifdef _HEADER_ONLY
#include "sparse_reader.cc"
#endif
#endif
