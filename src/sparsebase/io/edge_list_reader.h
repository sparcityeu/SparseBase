#ifndef SPARSEBASE_PROJECT_EDGE_LIST__READER_H
#define SPARSEBASE_PROJECT_EDGE_LIST__READER_H
#include "sparsebase/config.h"
#include "sparsebase/io/reader.h"

#include <string>

namespace sparsebase::io {
//! Reader for the Edge List file format
/*!
 * Reads files of the following format:
 * - Each line contains 2 ids (vertices for a graph, cols/rows for a matrix)
 * followed by an optional weight
 * - Delimiters should be spaces or tabs
 * - Each line represents a connection between the specified ids with the given
 * weight
 */
template <typename IDType, typename NNZType, typename ValueType>
class EdgeListReader : public Reader,
                       public ReadsCSR<IDType, NNZType, ValueType>,
                       public ReadsCOO<IDType, NNZType, ValueType> {
 public:
  /*!
   * Constructor for the EdgeListReader class
   * @param filename path to the file to be read
   * @param weighted should be set to true if the file contains weights
   * @param remove_duplicates if set to true duplicate connections will be
   * removed
   * @param remove_self_edges if set to true connections from any vertex to
   * itself will be removed
   * @param read_undirected_ if set to true for any entry (u,v) both (u,v) and
   * (v,u) will be read
   */
  explicit EdgeListReader(std::string filename, bool weighted = false,
                          bool remove_duplicates = false,
                          bool remove_self_edges = false,
                          bool read_undirected_ = true, bool square = false);
  format::CSR<IDType, NNZType, ValueType> *ReadCSR() const override;
  format::COO<IDType, NNZType, ValueType> *ReadCOO() const override;
  ~EdgeListReader() override;

 private:
  std::string filename_;
  bool weighted_;
  bool remove_duplicates_;
  bool remove_self_edges_;
  bool read_undirected_;
  bool square_;
};

}
#ifdef _HEADER_ONLY
#include "edge_list_reader.cc"
#endif
#endif  // SPARSEBASE_PROJECT_EDGE_LIST__READER_H
