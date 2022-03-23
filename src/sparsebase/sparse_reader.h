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

template <class VertexID, typename NumEdges, typename Weight>
class ReadsSparseFormat {
public:
  virtual format::Format *ReadSparseFormat() const = 0;
};

template <class VertexID, typename NumEdges, typename Weight> class ReadsCSR {
public:
  virtual format::CSR<VertexID, NumEdges, Weight> *ReadCSR() const = 0;
};

template <class VertexID, typename NumEdges, typename Weight> class ReadsCOO {
public:
  virtual format::COO<VertexID, NumEdges, Weight> *ReadCOO() const = 0;
};
// Add weighted option with contexpr
template <typename VertexID, typename NumEdges, typename Weight>
class UedgelistReader : public Reader<VertexID, NumEdges, Weight>,
                        public ReadsCSR<VertexID, NumEdges, Weight>,
                        public ReadsSparseFormat<VertexID, NumEdges, Weight> {
public:
  UedgelistReader(std::string filename, bool _weighted = false);
  format::CSR<VertexID, NumEdges, Weight> *ReadCSR() const;
  format::Format *ReadSparseFormat() const;
  virtual ~UedgelistReader();

private:
  static bool SortEdge(const std::pair<VertexID, VertexID> &a,
                       const std::pair<VertexID, VertexID> &b);
  std::string filename_;
  bool weighted_;
};

template <typename VertexID, typename NumEdges, typename Weight>
class MTXReader : public Reader<VertexID, NumEdges, Weight>,
                  public ReadsCOO<VertexID, NumEdges, Weight>,
                  public ReadsSparseFormat<VertexID, NumEdges, Weight> {
public:
  MTXReader(std::string filename, bool _weighted = false);
  format::COO<VertexID, NumEdges, Weight> *ReadCOO() const;
  format::Format *ReadSparseFormat() const;
  virtual ~MTXReader();

private:
  std::string filename_;
  bool weighted_;
};

#ifdef USE_PIGO
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

#endif



template <typename IDType, typename NNZType, typename ValueType>
class BinaryReader: public Reader<IDType, NNZType, ValueType>,
  public ReadsCSR<IDType, NNZType, ValueType>,
  public ReadsCOO<IDType, NNZType, ValueType> {
public:
  BinaryReader(std::string filename);
  format::COO<IDType, NNZType, ValueType> *ReadCOO() const;
  format::CSR<IDType, NNZType, ValueType> *ReadCSR() const;

private:
  std::string filename_;
};

} // namespace utils

} // namespace sparsebase

#ifdef _HEADER_ONLY
#include "sparse_reader.cc"
#endif
#endif
