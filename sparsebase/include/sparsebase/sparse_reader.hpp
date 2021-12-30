#ifndef _SPARSEREADER_HPP
#define _SPARSEREADER_HPP

#include "sparse_format.hpp"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <vector>

namespace sparsebase {

template <typename ID, typename NumNonZeros, typename Value> class SparseReader {
public:
  virtual ~SparseReader();
};

template <class VertexID, typename NumEdges, typename Weight> class ReadsSparseFormat {
public:
  virtual SparseFormat<VertexID, NumEdges, Weight> *read_sparseformat() const = 0;
};

template <class VertexID, typename NumEdges, typename Weight> class ReadsCSR {
public:
  virtual CSR<VertexID, NumEdges, Weight> *read_csr() const = 0;
};

template <class VertexID, typename NumEdges, typename Weight> class ReadsCOO {
public:
  virtual COO<VertexID, NumEdges, Weight> *read_coo() const = 0;
};
// Add weighted option with contexpr
template <typename VertexID, typename NumEdges, typename Weight>
class UedgelistReader : public SparseReader<VertexID, NumEdges, Weight>,
                        public ReadsCSR<VertexID, NumEdges, Weight>,
                        public ReadsSparseFormat<VertexID, NumEdges, Weight> {
public:
  UedgelistReader(std::string filename, bool _weighted = false);
  CSR<VertexID, NumEdges, Weight> *read_csr() const;
  SparseFormat<VertexID, NumEdges, Weight> *read_sparseformat() const;
  virtual ~UedgelistReader();

private:
  static bool sortedge(const std::pair<VertexID, VertexID> &a, const std::pair<VertexID, VertexID> &b);
  std::string filename;
  bool weighted;
};

template <typename VertexID, typename NumEdges, typename Weight>
class MTXReader : public SparseReader<VertexID, NumEdges, Weight>,
                  public ReadsCOO<VertexID, NumEdges, Weight>,
                  public ReadsSparseFormat<VertexID, NumEdges, Weight> {
public:
  MTXReader(std::string filename, bool _weighted = false);
  COO<VertexID, NumEdges, Weight> *read_coo() const;
  SparseFormat<VertexID, NumEdges, Weight> *read_sparseformat() const;
  virtual ~MTXReader();

private:
  std::string filename;
  bool weighted;
};

} // namespace sparsebase

#endif
