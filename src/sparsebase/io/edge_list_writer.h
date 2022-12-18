#ifndef SPARSEBASE_PROJECT_EDGE_LIST_WRITER_H
#define SPARSEBASE_PROJECT_EDGE_LIST_WRITER_H
#include <string>
#include "sparsebase/config.h"
#include "sparsebase/io/writer.h"

namespace sparsebase::io {

//! Writes files by encoding them in edge list format
/*!
 * - Each line contains 2 ids (vertices for a graph, cols/rows for a matrix)
 * followed by an optional weight
 */
template <typename IDType, typename NNZType, typename ValueType>
class EdgeListWriter : public Writer,
                             public WritesCOO<IDType, NNZType, ValueType>,
                             public WritesCSR<IDType, NNZType, ValueType> {
 public:
  /*!
   * Constructor for the EdgeListReader class
   * @param filename to be written
   * @param directed set to true if edge (u,v) is different from edge (v,u)
   */
  explicit EdgeListWriter(std::string filename, bool directed = false);
  ~EdgeListWriter() override = default;
  void WriteCOO(format::COO<IDType, NNZType, ValueType> *coo) const override;
  void WriteCSR(format::CSR<IDType, NNZType, ValueType> *csr) const override;
 private:
  std::string filename_;
  bool directed_;
};

}  // namespace sparsebase::io
#ifdef _HEADER_ONLY
#include "edge_list_writer.cc"
#endif
#endif  // SPARSEBASE_PROJECT_EDGE_LIST_WRITER_H
