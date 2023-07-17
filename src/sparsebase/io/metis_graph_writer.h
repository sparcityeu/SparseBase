#ifndef SPARSEBASE_PROJECT_METIS_GRAPH_WRITER_H
#define SPARSEBASE_PROJECT_METIS_GRAPH_WRITER_H
#include <string>
#include "sparsebase/config.h"
#include "sparsebase/io/writer.h"
#include "sparsebase/object/object.h"

namespace sparsebase::io {

//! Writer for the METIS GRAPH file format
/*!
 * File format: https://people.sc.fsu.edu/~jburkardt/data/metis_graph/metis_graph.html
 */
template <typename IDType, typename NNZType, typename ValueType>
class MetisGraphWriter : public Writer,
                             public WritesGraph<IDType, NNZType, ValueType> {
 public:
  /*!
   * Constructor for the MetisGraphWriter class
   * @param filename to be written
   * @param edgeWeighted set to true if edge weights exists and wanted in output
   * @param vertexWeighted set to true if vertex weights exists and wanted in output
   * @param zero_indexed set to true if given graph vertices are 0 indexed
   */
  explicit MetisGraphWriter(std::string filename, 
                          bool edgeWeighted = false, bool vertexWeighted = false,
                            bool zero_indexed = false);
  ~MetisGraphWriter() override = default;
  void WriteGraph(object::Graph<IDType, NNZType, ValueType> *graph) const override;
 private:
  std::string filename_;
  bool edgeWeighted_;
  bool vertexWeighted_;
  bool zero_indexed_;
};

}  // namespace sparsebase::io
#ifdef _HEADER_ONLY
#include "metis_graph_writer.cc"
#endif
#endif  // SPARSEBASE_PROJECT_METIS_GRAPH_WRITER_H
