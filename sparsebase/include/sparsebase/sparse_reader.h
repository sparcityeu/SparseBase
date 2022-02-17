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
class SparseReader {
public:
  virtual ~SparseReader();
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
class UedgelistReader : public SparseReader<VertexID, NumEdges, Weight>,
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
class MTXReader : public SparseReader<VertexID, NumEdges, Weight>,
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

} // namespace utils

} // namespace sparsebase

#ifdef _HEADER_ONLY
#include "../../src/sparse_reader.cc"
#endif
#endif
